"""Переобучение модели одной командой.

    python -m train.retrain                 # посчитать и положить в model_new/
    python -m train.retrain --replace       # сразу заменить model/ и пересобрать фикстуры

Схема, гиперпараметры и признаки — те же самые, что у нынешней модели. Ничего
не подбирается заново: подбор гиперпараметров на тех же данных, на которых потом
меряется результат, — самый простой способ обмануть себя.

Что происходит по шагам:

1.  Грузятся минутки (кэш на диске; первый раз это около 180 МБ и несколько минут).
2.  Собираются обрезанные свечи (:00..:57) — вход признаков, и полные свечи —
    по ним считается исход рынка `close >= open`.
3.  Проверяются инварианты обрезки. Не сошлись — стоп, дальше считать нечего.
4.  Нарезаются фолды 1460/7/30 и проверяется целостность: каждый час периода
    оценки обязан быть предсказан ровно один раз.
5.  Прогон walk-forward: в каждом фолде модель обучается с нуля.
6.  Боевая модель обучается заново на последних 1460 днях и сохраняется
    с паспортом.
7.  Печатается сравнение «было / стало» рядом со старым паспортом.

По умолчанию ничего не заменяется: новая модель кладётся в model_new/, и решение
остаётся за вами. Так пересчёт нельзя случайно принять за замену.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

import lightgbm as lgb                                        # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score           # noqa: E402

from kit import _featload as fl                               # noqa: E402
from kit import candles as kc                                 # noqa: E402
from train import binance_1m, metrics as mt, phases as ph     # noqa: E402
from train import validation as val                           # noqa: E402

W = 78
HOUR = pd.Timedelta(hours=1)

#: Гиперпараметры LightGBM. Ровно те, на которых обучена нынешняя модель.
#: Подбирать их заново нельзя: измеренные числа относятся именно к этим.
PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, max_depth=3,
              min_child_samples=250, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=42)

#: Пороги, из которых выбирается порог уверенности во внутреннем контуре.
#: На рынок он не выносится (см. README), но в паспорте сохраняется.
THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
MIN_INNER_SELECTED = 100

#: Экономика часового рынка Polymarket.
PRICE, SHARES, FEE_RATE = 0.50, 5, 0.07

#: С какого дня грузить минутки: первому обучающему окну нужно
#: eval_start - embargo - train_days, плюс разгон 168 баров.
LOAD_START = "2019-08-01"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def num(x) -> str:
    return f"{int(x):,}".replace(",", " ")


# --------------------------------------------------------------------------
# данные
# --------------------------------------------------------------------------

def market_target(full: pd.DataFrame) -> pd.Series:
    """Исход рынка: close >= open ПОЛНОЙ часовой свечи. Равенство = Up.

    Именно так написано в правиле резолва Polymarket. Обратите внимание: исход
    считается по полной свече, а признаки — по обрезанной. Это не рассогласование,
    а суть задачи: в момент решения полной свечи ещё не существует.
    """
    f = full.set_index("open_time")
    return (f["close"] >= f["open"]).astype(float)


def forward_return(full: pd.DataFrame) -> pd.Series:
    """Движение внутри полной свечи, open -> close. Нужно для «б.п. за сделку»."""
    f = full.set_index("open_time")
    return f["close"] / f["open"] - 1.0


def check_cutoff_invariants(m1: pd.DataFrame, trunc: pd.DataFrame,
                            full: pd.DataFrame) -> dict:
    """Дешёвые проверки, ловящие перепутанную обрезку.

    Обрезанная свеча обязана начинаться там же, где полная, не выходить за её
    границы и закрываться минуткой :57.
    """
    a = trunc.set_index("open_time")
    b = full.set_index("open_time")
    common = a.index.intersection(b.index)

    open_bad = int((a.loc[common, "open"] != b.loc[common, "open"]).sum())
    high_bad = int((a.loc[common, "high"] > b.loc[common, "high"] + 1e-8).sum())
    low_bad = int((a.loc[common, "low"] < b.loc[common, "low"] - 1e-8).sum())

    last_min = m1[m1.index.minute == (59 - kc.CUTOFF_MINUTES)]["close"].astype("float64")
    last_min.index = last_min.index.floor("h")
    aligned = last_min.reindex(common)
    cmp_ok = aligned.notna()
    close_bad = int(((a.loc[common, "close"][cmp_ok] - aligned[cmp_ok]).abs() > 1e-8).sum())

    return {"n": len(common), "open": open_bad, "high": high_bad, "low": low_bad,
            "close": close_bad, "close_compared": int(cmp_ok.sum()),
            "ok": open_bad == 0 and high_bad == 0 and low_bad == 0 and close_bad == 0}


# --------------------------------------------------------------------------
# прогон
# --------------------------------------------------------------------------

def walk_forward(X, y, fwd, folds, inner):
    """Walk-forward: в каждом фолде модель обучается с нуля.

    Ничего не переносится между фолдами. Порог подбирается на внутренней
    валидации — последних 90 днях обучающего окна, которых проверочная часть
    не видела.
    """
    proba = pd.Series(np.nan, index=X.index)
    thr = pd.Series(np.nan, index=X.index)
    rows = []
    t0 = time.time()

    for f in folds:
        i_fit, i_val = inner(f)
        best_th, best_edge = 0.50, -9e9
        if int(i_fit.sum()) >= 200 and int(i_val.sum()) >= MIN_INNER_SELECTED:
            m_in = lgb.LGBMClassifier(**PARAMS).fit(X[i_fit], y[i_fit])
            p_in = m_in.predict_proba(X[i_val])[:, 1]
            f_in = fwd[i_val].to_numpy()
            for th in THRESHOLDS:
                sel = (p_in > th) | (p_in < 1 - th)
                if sel.sum() < MIN_INNER_SELECTED:
                    continue
                pr = (p_in[sel] > 0.5).astype(float)
                e = float((np.where(pr == 1, 1.0, -1.0) * f_in[sel]).mean() * 1e4)
                if e > best_edge:
                    best_edge, best_th = e, th

        model = lgb.LGBMClassifier(**PARAMS).fit(X[f.train], y[f.train])
        p = model.predict_proba(X[f.val])[:, 1]
        idx = X.index[f.val.to_numpy()]
        proba.loc[idx] = p
        thr.loc[idx] = best_th

        pred = (p > 0.5).astype(float)
        yy = y[f.val].to_numpy()
        rows.append(dict(n=f.n + 1, start=f.val_start, end=f.val_end,
                         bars=int(f.val.sum()), acc=float((pred == yy).mean()),
                         th=best_th))
        if (f.n + 1) % 5 == 0 or f.n + 1 == len(folds):
            log(f"    фолд {f.n + 1:2d}/{len(folds)}: {rows[-1]['start']:%Y-%m-%d}, "
                f"точность {rows[-1]['acc'] * 100:.2f}%   [{time.time() - t0:.0f} c]")
    return proba, thr, rows


def naive_rule(d: pd.DataFrame) -> pd.Series:
    """Планка: «против предыдущей свечи».

    ВНИМАНИЕ: здесь комплект ETH отличается от комплекта BTC, и намеренно.
    У BTC лучшее простое правило — «против 2 одинаковых подряд». У ETH на его
    собственных барах лучшим оказалось «против предыдущей»: 52.64% против
    52.27% у правила BTC. Планка обязана быть лучшим простым правилом ИМЕННО
    этой монеты, иначе сравнение льстит модели: против правила BTC разрыв
    вышел бы +1.77 п.п. вместо честных +1.40 п.п.

    Считается по ОБРЕЗАННОЙ свече и по рыночному правилу close >= open. Иначе
    было бы нечестно: в момент решения исход полной свечи H-1 ещё неизвестен.
    """
    up = (d["close"] >= d["open"]).astype(float)
    return pd.Series(1.0 - up.to_numpy(), index=d.index, dtype=float)


# --------------------------------------------------------------------------
# отчёт
# --------------------------------------------------------------------------

def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if kind == "pct":
        return f"{value:.2f}%"
    if kind == "pp":
        return f"{value:+.2f} п.п."
    if kind == "p":
        return f"{value:.4g}"
    if kind == "bp":
        return f"{value:+.2f}"
    return f"{value:.4f}"


def comparison_table(old: dict, new: dict) -> list[str]:
    """Таблица «было / стало». Старые числа берутся из паспорта прежней модели."""
    spec = [
        ("Точность модели", "model_pct", "pct", "pp"),
        ("Точность простого правила", "best_rule_pct", "pct", "pp"),
        ("Разрыв", "gap_pp", "pp", "pp"),
        ("McNemar p", "mcnemar_p", "p", None),
        ("Движение, б.п. за сделку", "edge_bp", "bp", "bp"),
        ("Balanced accuracy", "balanced_accuracy", "pct", "pp"),
        ("ROC-AUC", "roc_auc", "num", "num"),
        ("Log loss", "log_loss", "num", "num"),
        ("Brier", "brier", "num", "num"),
        ("Наклон калибровки", "calibration_slope", "num", "num"),
        ("Баров в проверке", "bars", "int", None),
        ("Окон выше 50%", "windows_above_50", "str", None),
    ]
    L = ["| Показатель | Было | Стало | Изменение |", "|---|---|---|---|"]
    for label, key, kind, dkind in spec:
        o, n = old.get(key), new.get(key)
        if kind == "str":
            L.append(f"| {label} | {o or '—'} | {n or '—'} | |")
            continue
        if kind == "int":
            L.append(f"| {label} | {num(o) if o else '—'} | {num(n) if n else '—'} | |")
            continue
        delta = ""
        if dkind and isinstance(o, (int, float)) and isinstance(n, (int, float)):
            d = n - o
            delta = f"{d:+.2f} п.п." if dkind == "pp" else f"{d:+.4f}"
            if dkind == "bp":
                delta = f"{d:+.2f}"
        L.append(f"| {label} | {_fmt(o, kind)} | {_fmt(n, kind)} | {delta} |")
    return L


# --------------------------------------------------------------------------
# главное
# --------------------------------------------------------------------------

def run(a) -> int:
    t_start = time.time()
    print("-" * W)
    print("  ПЕРЕОБУЧЕНИЕ ЧАСОВОЙ МОДЕЛИ ПОД ПРАВИЛО POLYMARKET")
    print("-" * W)

    scheme = val.Scheme()
    out_dir = Path(a.out) if a.out else (ROOT / "model_new")
    old_json = ROOT / "model" / "poly_hourly_eth.json"
    old = json.loads(old_json.read_text(encoding="utf-8")) if old_json.exists() else {}
    old_wf = old.get("walkforward", {})
    if old:
        log(f"  прежняя модель: {old.get('model_id')}, "
            f"граница знаний {old.get('data_cutoff_utc')}")

    # ---------- данные ----------
    m1 = binance_1m.load_1m(start=LOAD_START, end=a.data_end, cache_dir=a.cache_dir)
    last_minute = m1.index[-1]

    full = kc.hourly(m1, cutoff=0)
    full = full[full["open_time"] + pd.Timedelta(minutes=59) <= last_minute]
    full = full.reset_index(drop=True)

    trunc = kc.hourly(m1, cutoff=kc.CUTOFF_MINUTES)
    # Строке T нужна закрытая полная свеча T+1ч, то есть прожитая минутка T+1:59.
    trunc = trunc[trunc["open_time"] <= last_minute - pd.Timedelta(hours=1, minutes=59)]
    trunc = trunc.reset_index(drop=True)
    log(f"  свечей: обрезанных {num(len(trunc))}, полных {num(len(full))}")

    inv = check_cutoff_invariants(m1, trunc, full)
    log(f"  инварианты обрезки: open {inv['open']}, high {inv['high']}, "
        f"low {inv['low']}, close {inv['close']} нарушений "
        f"(close сверен на {num(inv['close_compared'])} часах)")
    if not inv["ok"]:
        log("  СТОП: обрезанная свеча не согласуется с полной. Дальше считать нечего.")
        return 1

    # ---------- признаки и таргет ----------
    b = fl.build(trunc, "1h")
    d, X = b["df"], b["X"]
    t = d["open_time"]
    nxt = pd.DatetimeIndex(t + HOUR)
    y = pd.Series(market_target(full).reindex(nxt).to_numpy(), index=d.index)
    fwd = pd.Series(forward_return(full).reindex(nxt).to_numpy(), index=d.index)
    usable = b["valid"] & y.notna() & b["ready"]

    corr = val.future_correlations(X, fwd, usable)
    log(f"  максимальная корреляция признака с будущим: "
        f"{float(corr.iloc[0]):.4f} ({corr.index[0]})"
        + ("   ЧИСТО" if float(corr.iloc[0]) < 0.15 else "   ТРЕВОГА: похоже на утечку"))
    if float(corr.iloc[0]) >= 0.15:
        log("  СТОП: признак слишком похож на будущее. Это утечка, а не сигнал.")
        return 1

    folds = val.make_folds(t, usable, scheme)
    integrity = val.check_folds(folds, t, usable, scheme)
    log(f"  период оценки {integrity['eval_start']:%Y-%m-%d} .. "
        f"{integrity['eval_end']:%Y-%m-%d}, баров {integrity['n_eval_bars']}, "
        f"фолдов {len(folds)}")
    log(f"  целостность: пересечений {integrity['train_val_overlap']}, зазор "
        f"{integrity['min_embargo_days']:.0f} д, предсказано ровно один раз "
        f"{integrity['predicted_once']} из {integrity['n_eval_bars']}")
    if not folds:
        log("  СТОП: не построено ни одного фолда.")
        return 1
    if integrity["predicted_once"] != integrity["n_eval_bars"]:
        log("  СТОП: не каждый час периода оценки предсказан ровно один раз.")
        return 1

    # ---------- прогон ----------
    log("")
    log("  walk-forward:")
    proba, thr, rows = walk_forward(X, y, fwd, folds,
                                    lambda f: val.inner_split(t, f.train, scheme, f.train_end))
    ev = usable & proba.notna()
    pred = pd.Series(np.where(proba.notna(), (proba > 0.5).astype(float), np.nan),
                     index=d.index)

    # ---------- метрики ----------
    rule = naive_rule(d)
    mod = mt.cell(pred, y, fwd, ev)
    nai = mt.cell(rule, y, fwd, ev)
    yy = y[ev].to_numpy()
    pr = proba[ev].to_numpy()
    pp = pred[ev].to_numpy()

    rec_up = float(((pp == 1) & (yy == 1)).sum() / max(((yy == 1)).sum(), 1))
    rec_dn = float(((pp == 0) & (yy == 0)).sum() / max(((yy == 0)).sum(), 1))
    _, slope = mt.calibration(pr, yy)
    accs = np.array([r["acc"] for r in rows])

    new_wf = {
        "eval_start": integrity["eval_start"].isoformat(),
        "eval_end": integrity["eval_end"].isoformat(),
        "bars": int(ev.sum()),
        "folds": len(folds),
        "best_rule_pct": round(nai["acc"] * 100, 4),
        "model_pct": round(mod["acc"] * 100, 4),
        "gap_pp": round((mod["acc"] - nai["acc"]) * 100, 4),
        "mcnemar_p": mt.mcnemar(mod["hit"], nai["hit"]),
        "balanced_accuracy": round((rec_up + rec_dn) / 2 * 100, 4),
        "roc_auc": round(float(roc_auc_score(yy, pr)), 6),
        "log_loss": round(float(log_loss(yy, pr)), 6),
        "brier": round(float(np.mean((pr - yy) ** 2)), 6),
        "edge_bp": round(mod["bp"], 4),
        "calibration_slope": round(slope, 4),
        "windows_above_50": f"{int((accs > 0.5).sum())}/{len(accs)}",
    }

    # ---------- фазы ----------
    phase_rows = []
    lab = ph.phase_label_causal(d)
    for name in ph.PHASES:
        mask = ev & (lab == name)
        cm, cn = mt.cell(pred, y, fwd, mask), mt.cell(rule, y, fwd, mask)
        if cm and cn:
            phase_rows.append((name, cm["n"], cn["acc"] * 100, cm["acc"] * 100,
                               (cm["acc"] - cn["acc"]) * 100,
                               mt.mcnemar(cm["hit"], cn["hit"]), cm["bp"]))

    # ---------- боевая модель ----------
    cutoff = t[usable].max()
    win_start = cutoff - pd.Timedelta(days=scheme.train_days)
    train = usable & (t > win_start) & (t <= cutoff)
    i_fit, i_val = val.inner_split(t, train, scheme, cutoff)

    best_th, best_edge = 0.50, -9e9
    if int(i_fit.sum()) >= 200 and int(i_val.sum()) >= MIN_INNER_SELECTED:
        m_in = lgb.LGBMClassifier(**PARAMS).fit(X[i_fit], y[i_fit])
        p_in = m_in.predict_proba(X[i_val])[:, 1]
        f_in = fwd[i_val].to_numpy()
        for th in THRESHOLDS:
            sel = (p_in > th) | (p_in < 1 - th)
            if sel.sum() < MIN_INNER_SELECTED:
                continue
            prr = (p_in[sel] > 0.5).astype(float)
            e = float((np.where(prr == 1, 1.0, -1.0) * f_in[sel]).mean() * 1e4)
            if e > best_edge:
                best_edge, best_th = e, th

    model = lgb.LGBMClassifier(**PARAMS).fit(X[train], y[train])
    out_dir.mkdir(parents=True, exist_ok=True)
    txt, js = out_dir / "poly_hourly_eth.txt", out_dir / "poly_hourly_eth.json"
    model.booster_.save_model(str(txt))
    sha = hashlib.sha256(txt.read_bytes()).hexdigest()

    hits = (pp == yy)
    pnl = np.where(hits, SHARES * (1 - PRICE), -SHARES * PRICE)
    fee = SHARES * FEE_RATE * PRICE * (1 - PRICE)

    meta = {
        "model_id": sha[:12],
        "sha256": sha,
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "bar_minutes": 60,
        "market": "Polymarket, ETH Up or Down (часовой)",
        "target_rule": "close >= open полной часовой свечи Binance ETHUSDT; равенство = Up",
        "feature_cutoff_minutes": kc.CUTOFF_MINUTES,
        "feature_source": "часовая свеча H-1 из минуток (H-1):00 .. (H-1):57",
        "decision_time": "H - 2 минуты",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_cutoff_utc": cutoff.isoformat(),
        "train_start_utc": win_start.isoformat(),
        "train_rows": int(train.sum()),
        "train_days": scheme.train_days,
        "features": fl.FEATURES,
        "params": dict(PARAMS),
        "threshold": best_th,
        "phase_coverage": ph.coverage(ph.phase_label_report(d)[train]),
        "walkforward": new_wf,
        "market_economics": {
            "price": PRICE, "shares": SHARES, "fee_rate": FEE_RATE,
            "fee_per_bet_usd": round(fee, 6),
            "breakeven_maker_pct": round(PRICE * 100, 2),
            "breakeven_taker_pct": round((PRICE + fee / SHARES) * 100, 2),
            "ev_per_bet_usd": round(float(pnl.mean()), 4),
            "ev_total_usd": round(float(pnl.mean()) * len(pnl), 2),
        },
        "note": ("Переобучена комплектом poly_hourly_kit_eth. Схема, признаки и "
                 "гиперпараметры те же. Порог подобран на внутренней валидации и "
                 "на рынок не выносится: ставим на каждый час."),
    }
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- сравнение ----------
    L = ["# Переобучение: было / стало", "",
         f"Прогнано {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, "
         f"{time.time() - t_start:.0f} с.", "",
         f"Прежняя модель `{old.get('model_id', '—')}` "
         f"(граница знаний {old.get('data_cutoff_utc', '—')})", "",
         f"Новая модель `{meta['model_id']}` "
         f"(граница знаний {meta['data_cutoff_utc']}, {num(meta['train_rows'])} баров)", ""]
    L += comparison_table(old_wf, new_wf)
    L += ["", "## Фазы рынка (причинная разметка)", "",
          "| Фаза | Баров | Правило | Модель | Разрыв | McNemar p | б.п. |",
          "|---|---|---|---|---|---|---|"]
    for name, n_, r_, m_, g_, p_, bp_ in phase_rows:
        L.append(f"| {name} | {n_} | {r_:.2f}% | {m_:.2f}% | {g_:+.2f} п.п. | "
                 f"{p_:.4g} | {bp_:+.2f} |")
    weak = [name for name, _, r_, m_, *_ in phase_rows if m_ <= r_]
    L += ["", ("Модель обгоняет правило во всех трёх фазах." if not weak else
               "**ПРОВАЛ в фазах: " + ", ".join(weak) + ".** Модель, которая хороша "
               "в целом, но проваливается в фазе, к использованию не годится."), ""]
    (out_dir / "retrain_report.md").write_text("\n".join(L), encoding="utf-8")

    print()
    print("=" * W)
    print("  БЫЛО / СТАЛО")
    print("=" * W)
    for line in comparison_table(old_wf, new_wf):
        print("  " + line)
    print()
    print("  Фазы рынка (причинная разметка):")
    for name, n_, r_, m_, g_, p_, bp_ in phase_rows:
        print(f"    {name:<10} {n_:>6} баров   правило {r_:.2f}%   модель {m_:.2f}%   "
              f"{g_:+.2f} п.п.   p={p_:.4g}")
    if weak:
        print(f"    ПРОВАЛ в фазах: {', '.join(weak)}")
    print()
    print(f"  новая модель: {out_dir / 'poly_hourly_eth.txt'}   id={meta['model_id']}")
    print(f"  отчёт:        {out_dir / 'retrain_report.md'}")
    print(f"  время:        {time.time() - t_start:.0f} c")

    if a.replace:
        for name in ("poly_hourly_eth.txt", "poly_hourly_eth.json"):
            shutil.copyfile(out_dir / name, ROOT / "model" / name)
        print()
        print("  --replace: модель в model/ заменена. Пересобираю фикстуры...")
        print()
        from train import make_fixtures
        rc = make_fixtures.main(["--cache-dir", str(a.cache_dir)])
        if rc:
            print("  Фикстуры пересобрать не удалось — самопроверка не пройдёт.")
            return rc
    else:
        print()
        print("  Модель НЕ заменена. Чтобы поставить её в работу:")
        print("    python -m train.retrain --replace")
        print("  либо скопируйте оба файла в model/ и выполните")
        print("    python -m train.make_fixtures")
    print()
    print("  Затем обязательно:  python -m kit.selfcheck")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Переобучить часовую модель по той же схеме 1460/7/30.")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "binance_1m"),
                    help="кэш минуток; можно указать на готовый, тогда качать нечего")
    ap.add_argument("--out", default=None,
                    help="куда положить новую модель (по умолчанию model_new/)")
    ap.add_argument("--replace", action="store_true",
                    help="сразу заменить model/ и пересобрать фикстуры")
    ap.add_argument("--data-end", default=None,
                    help="конец данных, UTC. По умолчанию — до текущего момента. "
                         "Переданные с комплектом числа получены при "
                         "--data-end '2026-09-07 00:00': без этой заморозки "
                         "каждый запуск приносит на минуту больше данных, и числа плывут")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
