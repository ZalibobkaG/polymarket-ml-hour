"""Признаки в реальном времени — одна строка на один рынок.

Здесь легче всего испортить всё незаметно. Модель обучена на признаках, посчитанных
по свече H−1, обрезанной на CUTOFF_MINUTES минут; если в бою в них попадёт хоть одна
лишняя минутка, вход перестанет соответствовать обучению, и честные 52.9% превратятся
в неизвестно что. Поэтому:

* расчёт идёт тем же кодом, что и при обучении (`research/features.py`) — не копией;
* минутки жёстко обрезаются моментом решения перед расчётом;
* `verify()` сверяет живой путь с обучающим на исторических часах, допуск — ноль.

Тайминг. Рынок с началом в H — это свеча H:00:00–H:59:59. Решение принимается
в (H−1):(60−cutoff):00, то есть за `cutoff` минут до H. Признаки берутся из свечи H−1,
собранной из минуток [(H−1):00 … (H−1):(59−cutoff)].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from app.log import get

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT.parent / "daily_btc_notebook"))

import btc_data as bd  # noqa: E402
import features as train_features  # noqa: E402  — тот же модуль, что при обучении

log = get("features")


def decision_time(hour_start: pd.Timestamp, cutoff: int | None = None) -> pd.Timestamp:
    """Момент, когда признаки для рынка `hour_start` становятся известны."""
    cutoff = config.CUTOFF_MINUTES if cutoff is None else cutoff
    return pd.Timestamp(hour_start) - pd.Timedelta(minutes=cutoff)


def load_minutes(hour_start: pd.Timestamp, cutoff: int | None = None,
                 days: int = config.HISTORY_DAYS) -> pd.DataFrame:
    """Минутки для расчёта, обрезанные ровно моментом решения.

    Обрезка — главная защита от подглядывания: что бы ни отдал источник, ничего
    новее момента решения в расчёт не попадёт.
    """
    cut = decision_time(hour_start, cutoff)
    start = cut - pd.Timedelta(days=days)
    m1 = bd.load_1m(start=start, verbose=False)
    return m1[m1.index < cut]


def row(m1: pd.DataFrame, hour_start: pd.Timestamp,
        cutoff: int | None = None) -> pd.Series:
    """Признаки одной строки для рынка, начинающегося в `hour_start`.

    m1 должен быть уже обрезан (см. `load_minutes`); здесь обрезка не делается,
    чтобы функцию можно было честно проверить на исторических данных.
    """
    cutoff = config.CUTOFF_MINUTES if cutoff is None else cutoff
    hour_start = pd.Timestamp(hour_start)
    prev = hour_start - pd.Timedelta(hours=1)

    hist = train_features.hourly_candles(m1, cutoff=cutoff)
    if prev not in hist.index:
        raise ValueError(f"нет свечи {prev:%Y-%m-%d %H:%M}Z — не хватает данных")

    ind = train_features._indicators(hist, ma="ema")
    r = ind.loc[prev]
    if r.isna().any():
        missing = list(r.index[r.isna()])
        raise ValueError(f"признаки не досчитаны на {prev:%Y-%m-%d %H:%M}Z "
                         f"(мало истории): {missing}")

    out = {f"{c}_prev": float(r[c]) for c in ind.columns}
    # Календарь берётся от предсказываемого часа: известен заранее, утечкой не является.
    et = hour_start.tz_convert(bd.ET)
    out["hour_et"] = et.hour
    out["hour_sin"] = np.sin(2 * np.pi * et.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * et.hour / 24)
    out["dow"] = et.dayofweek
    return pd.Series(out, name=hour_start)


def for_market(hour_start: pd.Timestamp, cutoff: int | None = None) -> pd.Series:
    """Скачать минутки и посчитать признаки для одного рынка."""
    m1 = load_minutes(hour_start, cutoff)
    if m1.empty:
        raise ValueError("не получены минутки")
    last = m1.index[-1]
    expected = decision_time(hour_start, cutoff) - pd.Timedelta(minutes=1)
    if last < expected:
        # Данные отстают: биржа или сеть не успели. Лучше отказаться от ставки,
        # чем считать признаки по дырявому ряду.
        raise ValueError(f"минутки отстают: последняя {last:%H:%M}Z, "
                         f"ожидалась {expected:%H:%M}Z")
    return row(m1, hour_start, cutoff)


# --------------------------------------------------------------------------
# Самопроверка
# --------------------------------------------------------------------------

def verify(n: int = 6, cutoff: int | None = None) -> float:
    """Сверяет живой путь с обучающим на случайных исторических часах.

    Обучающий путь считает всю таблицу разом, живой — одну строку по обрезанным
    минуткам. Любое расхождение означает, что в бою модель получит не то, на чём
    училась. Допуск — ноль.
    """
    cutoff = config.CUTOFF_MINUTES if cutoff is None else cutoff
    m1 = bd.load_1m(start="2025-01-01", verbose=False)
    batch = train_features.build(m1, cutoff=cutoff)
    cols = [c for c in batch.columns if c != "y"]

    rng = np.random.default_rng(0)
    sample = pd.DatetimeIndex(rng.choice(batch.index[800:], size=n, replace=False))
    worst = 0.0
    for ts in sample.sort_values():
        cut = decision_time(ts, cutoff)
        live = row(m1[m1.index < cut], ts, cutoff)
        diff = float((live[cols] - batch.loc[ts, cols]).abs().max())
        worst = max(worst, diff)
        log.debug("%s max|разница| = %.2e", ts, diff)
    log.info("самопроверка признаков: худшее расхождение %.2e на %d часах", worst, n)
    return worst
