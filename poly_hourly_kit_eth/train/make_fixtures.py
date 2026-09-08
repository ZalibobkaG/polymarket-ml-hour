"""Пересборка фикстур самопроверки: fixtures/minutes.csv.gz и fixtures/reference.csv.

Запускать нужно ровно в одном случае — когда в model/ положили другую модель.
Эталонные вероятности привязаны к конкретной модели, и после её замены старый
эталон обязан не сойтись. Это не поломка, а именно то, ради чего он существует.

    python -m train.make_fixtures

Что делает:

1.  Грузит минутки за длинный отрезок истории.
2.  Считает признаки и вероятности по ВСЕЙ этой истории — это «настоящий прогон».
3.  Вырезает короткий кусок минуток под фикстуру и пересчитывает те же часы
    заново, уже через `kit.predict.predict`, то есть ровно тем путём, которым
    пойдёт напарник.
4.  Сверяет два расчёта. Расходятся — фикстуры не пишутся вовсе: эталон, который
    сам себе не равен, хуже отсутствующего.
5.  Пишет фикстуры и обновляет паспорт комплекта.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kit import _featload as fl                       # noqa: E402
from kit import candles as kc                         # noqa: E402
from kit import predict as kp                         # noqa: E402
from train import binance_1m                          # noqa: E402

FIXTURES = ROOT / "fixtures"
PASSPORT = ROOT / "passport.json"

#: Сколько часов кладём в эталон.
DEFAULT_ROWS = 200

#: Сколько часов минуток идёт в фикстуру перед первым эталонным часом.
#: Жёсткий минимум predict() — 240 ч; здесь берётся заметный запас сверх него.
DEFAULT_WARMUP_HOURS = 336

#: Сколько истории грузится для «настоящего прогона». Год с лишним — заведомо
#: больше любого окна признаков.
DEFAULT_HISTORY_DAYS = 400

HOUR = pd.Timedelta(hours=1)
FLOAT_FMT = "%.17g"      # столько знаков хватает, чтобы float64 читался обратно точно


def _write_csv(df: pd.DataFrame, path: Path, gz: bool = False) -> None:
    text = df.to_csv(index=False, float_format=FLOAT_FMT, lineterminator="\n")
    if gz:
        # mtime=0 — чтобы одинаковые данные давали одинаковый файл и одинаковую
        # контрольную сумму, а не менялись от одного времени запуска к другому.
        with gzip.GzipFile(path, "wb", mtime=0) as fh:
            fh.write(text.encode("utf-8"))
    else:
        path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Пересобрать фикстуры самопроверки под текущую модель.")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "binance_1m"),
                    help="куда класть кэш минуток (можно указать на готовый)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help=f"сколько часов в эталоне (по умолчанию {DEFAULT_ROWS})")
    ap.add_argument("--warmup-hours", type=int, default=DEFAULT_WARMUP_HOURS,
                    help=f"разгон в фикстуре, часов (по умолчанию {DEFAULT_WARMUP_HOURS})")
    ap.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS,
                    help="сколько истории грузить для «настоящего прогона»")
    ap.add_argument("--end", default=None,
                    help="последний час, UTC; по умолчанию — граница знаний модели")
    a = ap.parse_args(argv)

    meta = json.loads((ROOT / "model" / "poly_hourly_eth.json").read_text(encoding="utf-8"))
    end = pd.Timestamp(a.end or meta["data_cutoff_utc"])
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    start = end - pd.Timedelta(days=a.history_days)

    print("-" * 78)
    print("  ПЕРЕСБОРКА ФИКСТУР")
    print("-" * 78)
    print(f"  модель {meta['model_id']}, граница знаний {meta['data_cutoff_utc']}")

    m1 = binance_1m.load_1m(start=start, end=end + HOUR, cache_dir=a.cache_dir)

    # --- «настоящий прогон»: признаки по всей загруженной истории -------------
    trunc = kc.hourly(m1, cutoff=kc.CUTOFF_MINUTES)
    built = fl.build(trunc, "1h")
    X, t, ready = built["X"], built["df"]["open_time"], built["ready"]
    booster = kp._load_model()["booster"]

    idx = np.flatnonzero(ready.to_numpy())
    if len(idx) < a.rows:
        print(f"  СТОП: пригодных часов {len(idx)}, а нужно {a.rows}. "
              "Возьмите больше истории (--history-days).")
        return 1
    pick = idx[-a.rows:]
    p_full = booster.predict(X.iloc[pick][fl.FEATURES].to_numpy(dtype="float64"))

    # Час рынка H = час признаков + 1. Именно его получает predict().
    market_hours = pd.DatetimeIndex(t.iloc[pick]) + HOUR
    print(f"  эталонных часов: {len(pick)}, "
          f"{market_hours[0]:%Y-%m-%d %H:%M} .. {market_hours[-1]:%Y-%m-%d %H:%M} UTC")

    # --- фикстура: короткий кусок минуток ------------------------------------
    lo = market_hours[0] - pd.Timedelta(hours=a.warmup_hours)
    hi = market_hours[-1] - kc.CUTOFF_MINUTES * pd.Timedelta(minutes=1)
    fx = m1.loc[(m1.index >= lo) & (m1.index < hi), kc.REQUIRED_MINUTE_COLS].copy()
    print(f"  фикстура минуток: {len(fx):,} строк, "
          f"{fx.index[0]:%Y-%m-%d %H:%M} .. {fx.index[-1]:%Y-%m-%d %H:%M} UTC "
          f"(разгон {a.warmup_hours} ч)".replace(",", " "))

    # --- пересчёт тем же путём, которым пойдёт напарник -----------------------
    rows, probs = [], []
    for h in market_hours:
        row, p = kp._features_row(fx, h)
        rows.append(row.to_numpy(dtype="float64"))
        probs.append(p)
    probs = np.asarray(probs, dtype="float64")

    dp = np.abs(probs - p_full)
    n_diff = int((probs != p_full).sum())
    print(f"  сверка с полным прогоном: расхождений {n_diff} из {len(probs)}, "
          f"максимум {dp.max():.3e}")
    if n_diff:
        print("  СТОП: расчёт по фикстуре не совпал с расчётом по полной истории.")
        print("  Увеличьте --warmup-hours и повторите. Эталон с расхождением не пишется.")
        return 1

    # --- запись ---------------------------------------------------------------
    FIXTURES.mkdir(exist_ok=True)
    minutes_out = FIXTURES / "minutes.csv.gz"
    ref_out = FIXTURES / "reference.csv"

    fx_csv = fx.reset_index().rename(columns={fx.index.name or "index": "open_time"})
    fx_csv["open_time"] = pd.DatetimeIndex(fx_csv["open_time"]).strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_csv(fx_csv, minutes_out, gz=True)

    ref = pd.DataFrame(np.vstack(rows), columns=fl.FEATURES)
    ref.insert(0, "hour_utc", [h.strftime("%Y-%m-%dT%H:%M:%S%z") for h in market_hours])
    ref["p_up"] = probs
    _write_csv(ref, ref_out)

    # --- проверка обратного чтения: CSV обязан читаться бит в бит -------------
    # float_precision="round_trip": разбор по умолчанию быстрее, но иногда ошибается
    # на последний бит. Здесь допуск ноль, поэтому нужен точный разбор.
    back = pd.read_csv(ref_out, float_precision="round_trip")
    bad = int((back["p_up"].to_numpy() != probs).sum())
    bad += int((back[fl.FEATURES].to_numpy() != np.vstack(rows)).sum())
    if bad:
        print(f"  СТОП: эталон не читается обратно точно ({bad} значений разошлись).")
        return 1

    passport = json.loads(PASSPORT.read_text(encoding="utf-8"))
    passport["model_id"] = meta["model_id"]
    passport["fixtures"] = {
        "minutes_csv_gz_sha256": hashlib.sha256(minutes_out.read_bytes()).hexdigest(),
        "reference_csv_sha256": hashlib.sha256(ref_out.read_bytes()).hexdigest(),
        "rows": int(len(ref)),
        "minute_rows": int(len(fx_csv)),
        "warmup_hours": int(a.warmup_hours),
        "first_hour_utc": market_hours[0].isoformat(),
        "last_hour_utc": market_hours[-1].isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    PASSPORT.write_text(json.dumps(passport, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"  записано: fixtures/minutes.csv.gz ({minutes_out.stat().st_size / 1024:.0f} КБ)")
    print(f"            fixtures/reference.csv  ({ref_out.stat().st_size / 1024:.0f} КБ)")
    print("            passport.json обновлён")
    print()
    print("  Теперь запустите самопроверку:  python -m kit.selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
