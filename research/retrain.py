"""Переобучение модели на правильном таргете и правильной сетке.

Что здесь делается и почему:

* **таргет** — `close >= open` стандартной часовой свечи Binance, то есть буквально
  правило резолва рынка (старая модель училась на другом и теряла 0.9 п.п.);
* **признаки** — по обрезанной свече, ровно как они будут доступны в бою за 2 минуты
  до начала часа;
* **валидация** — схема из `validation.py`: модель назначена заранее (логрег), окно
  и баланс из пространства поиска убраны, отбор признаков делает elastic net,
  шесть внутренних фолдов вместо трёх;
* **сравнение** — с контрарным правилом на том же входе. Если модель его не обыгрывает,
  честнее торговать правилом: оно проще и не ломается.

Запуск:
    python research/retrain.py --cutoff 2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import features as F  # noqa: E402
import stats as st  # noqa: E402
import validation as V  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

MODEL = "logreg"        # назначена заранее — никакого выбора лучшей из шести
N_TRIALS = 50
START = "2023-07-28"


def contrarian(feats: pd.DataFrame) -> np.ndarray:
    """Базовая линия: ставим против знака предыдущей (обрезанной) свечи."""
    return 1 - feats["y_prev"].values.astype(int)


def run(cutoff: int, seed: int = 42) -> dict:
    feats = F.load(start=START, cutoff=cutoff)
    X, y = F.make_xy(feats)

    t0 = time.time()
    res = V.run_walk_forward(X, y, [MODEL], groups=F.FEATURE_GROUPS,
                             n_trials=N_TRIALS, seeds=(seed,))
    res["cutoff"] = cutoff
    res["feats_index"] = feats.index
    res["contra"] = pd.Series(contrarian(feats), index=feats.index)
    print(f"  walk-forward за {(time.time()-t0)/60:.1f} мин")
    (CACHE / f"wf_cutoff{cutoff}.pkl").write_bytes(pickle.dumps(res))
    return res


def evaluate(res: dict) -> dict:
    """Точность, интервалы и сравнение с контрарным правилом на одних и тех же часах."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    d = res["oos"].sort_index()
    idx = pd.DatetimeIndex(d.index)
    hit = (d["pred"].values == d["y"].values)
    contra = res["contra"].reindex(idx).values
    hit_contra = (contra == d["y"].values)
    always_up = (d["y"].values == 1)

    lo, hi = st.binom_ci(int(hit.sum()), len(hit))
    bb = st.block_bootstrap_ci(hit, idx, block="D", n_boot=4000, seed=2)
    paired = st.paired_block_bootstrap(hit, hit_contra, idx, seed=3)
    mc = st.mcnemar(hit, hit_contra)

    out = {
        "cutoff": res["cutoff"],
        "часов": len(hit),
        "точность": hit.mean() * 100,
        "ci_binom": (lo * 100, hi * 100),
        "ci_block": (bb["ci_low"] * 100, bb["ci_high"] * 100),
        "контрарное правило": hit_contra.mean() * 100,
        "всегда Up": always_up.mean() * 100,
        "перевес над правилом": paired["разница"] * 100,
        "перевес CI": (paired["ci_low"] * 100, paired["ci_high"] * 100),
        "перевес p": paired["p_two_sided"],
        "mcnemar p": mc["p"],
        "ставил Up, %": d["pred"].mean() * 100,
        "log_loss": log_loss(d["y"], d["proba"], labels=[0, 1]),
        "brier": brier_score_loss(d["y"], d["proba"]),
        "roc_auc": roc_auc_score(d["y"], d["proba"]),
    }

    # Наклон калибровки: <1 означает переуверенность (в старой модели было 0.554).
    from sklearn.linear_model import LogisticRegression
    p = np.clip(d["proba"].values, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    out["наклон калибровки"] = float(
        LogisticRegression(C=1e6, max_iter=1000).fit(z, d["y"].values).coef_[0, 0])

    # По кварталам — обыгрываем ли базовую линию стабильно.
    q = pd.DataFrame({"y": d["y"].values, "hit": hit, "contra": hit_contra,
                      "fold": d["fold"].values}, index=idx)
    out["по фолдам"] = q.groupby("fold").agg(
        часов=("y", "size"), Up=("y", lambda s: s.mean() * 100),
        модель=("hit", lambda s: s.mean() * 100),
        правило=("contra", lambda s: s.mean() * 100)).round(2)
    return out


def report(ev: dict) -> None:
    c = ev["cutoff"]
    when = "в HH:00:00 (полная свеча)" if c == 0 else f"за {c} мин до конца часа"
    print(f"\n{'='*70}\nОБРЕЗКА {c} МИН — решение {when}\n{'='*70}")
    print(f"часов вне обучения : {ev['часов']:,}")
    print(f"ТОЧНОСТЬ           : {ev['точность']:.2f}%")
    print(f"  интервал биномиальный : [{ev['ci_binom'][0]:.2f}, {ev['ci_binom'][1]:.2f}]")
    print(f"  интервал блочный      : [{ev['ci_block'][0]:.2f}, {ev['ci_block'][1]:.2f}]")
    print(f"контрарное правило : {ev['контрарное правило']:.2f}%")
    print(f"всегда Up          : {ev['всегда Up']:.2f}%")
    print(f"ПЕРЕВЕС над правилом: {ev['перевес над правилом']:+.2f} п.п.  "
          f"CI [{ev['перевес CI'][0]:+.2f}, {ev['перевес CI'][1]:+.2f}]  "
          f"p={ev['перевес p']:.4f} (McNemar {ev['mcnemar p']:.4f})")
    print(f"ставил Up          : {ev['ставил Up, %']:.1f}%")
    print(f"log loss {ev['log_loss']:.5f}  Brier {ev['brier']:.5f}  AUC {ev['roc_auc']:.4f}")
    print(f"наклон калибровки  : {ev['наклон калибровки']:.3f} "
          f"({'переуверена' if ev['наклон калибровки'] < 1 else 'недоуверена'})")
    print("\nпо кварталам:")
    print(ev["по фолдам"].to_string())


def fit_final(cutoff: int, seed: int = 42) -> None:
    """Обучение боевой модели на всей истории тем же порядком, что при проверке."""
    feats = F.load(start=START, cutoff=cutoff, verbose=False)
    X, y = F.make_xy(feats)
    params, _ = V.tune(MODEL, X, y, F.FEATURE_GROUPS, n_trials=N_TRIALS, seed=seed)
    cols = params["features"]
    model = V._make_model(MODEL, params)
    model.fit(X[cols], y)

    in_sample = float((model.predict(X[cols]) == y).mean())
    if in_sample > 0.60:
        raise SystemExit(f"точность на обучении {in_sample:.3f} — подозрительно, проверьте утечку")

    coefs = dict(zip(cols, model[-1].coef_[0]))
    nz = {k: round(float(v), 5) for k, v in coefs.items() if abs(v) > 1e-8}
    card = {
        "таргет": "close >= open стандартной часовой свечи Binance BTC/USDT (правило рынка)",
        "обрезка_минут": cutoff,
        "модель": "StandardScaler + LogisticRegression (elastic net)",
        "гиперпараметры": {k: params.get(k) for k in ("C", "l1_ratio", "class_weight")},
        "признаков_всего": len(cols),
        "признаков_ненулевых": len(nz),
        "коэффициенты": dict(sorted(nz.items(), key=lambda kv: -abs(kv[1]))),
        "обучено": {"строк": int(len(X)),
                    "с": f"{X.index[0]:%Y-%m-%d %H:%M}Z",
                    "по": f"{X.index[-1]:%Y-%m-%d %H:%M}Z",
                    "доля_Up": float(y.mean()),
                    "точность_на_обучении": in_sample},
        "собрано": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path = MODELS / (f"model_cutoff{cutoff}.joblib")
    joblib.dump({"model": model, "features": cols, "cutoff": cutoff, "card": card}, path)
    (MODELS / f"model_cutoff{cutoff}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {path.name} — {len(nz)} ненулевых признаков из {len(cols)}, "
          f"точность на обучении {in_sample*100:.2f}%")
    print("ненулевые коэффициенты:")
    for k, v in list(card["коэффициенты"].items())[:12]:
        print(f"   {k:26s} {v:+.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2)
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--no-fit", action="store_true")
    a = ap.parse_args()

    path = CACHE / f"wf_cutoff{a.cutoff}.pkl"
    if a.cached and path.exists():
        res = pickle.loads(path.read_bytes())
        print(f"кэш: {path.name}")
    else:
        res = run(a.cutoff)
    report(evaluate(res))
    if not a.no_fit:
        fit_final(a.cutoff)


if __name__ == "__main__":
    main()
