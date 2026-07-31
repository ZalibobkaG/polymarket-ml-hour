"""Признаки на сетке, по которой рынок реально резолвится.

Отличие от `daily_btc_notebook/btc_features_hourly.py` — в двух вещах, и обе
принципиальны для торговли:

1.  **Сетка часов.** Там час собирался из минуток HH:01 … HH+1:00 (сдвиг на минуту).
    Часовой рынок Polymarket резолвится по стандартной свече Binance HH:00:00–HH:59:59:

        > resolve to "Up" if the close price is greater than or equal to the open price
        > for the BTC/USDT 1 hour candle that begins on the time specified in the title

2.  **Таргет.** Там сравнивались close текущего и предыдущего часа, ничьи выбрасывались.
    Здесь — `close >= open` одной и той же свечи, и **равенство считается Up**,
    ровно как в правиле рынка.

Расхождение этих двух определений — 6.28% часов, и стоит оно 0.9 п.п. точности.

## Обрезка последней свечи

Решение о ставке принимается ДО начала часа (за EARLY_SECONDS). В этот момент
последняя свеча ещё не закрыта: за 2 минуты до конца известны 58 из 60 её минут.
Поэтому вся история строится из **одинаково обрезанных** свечей: у каждой берутся
минутки [:00 … :(59−cutoff)]. Такой ряд целиком доступен в момент решения,
и признаки в обучении считаются ровно так же, как будут считаться в бою.

При `cutoff=0` получается обычная полная свеча — режим «решаем в HH:00:00».
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Загрузка минуток и вспомогательные индикаторы переиспользуются из исследования.
RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "daily_btc_notebook"
sys.path.insert(0, str(RESEARCH_ROOT))

import btc_data as bd  # noqa: E402
from btc_features import _asymmetry, _dist_pct, _ma, _rsi, _streak  # noqa: E402

MA_WINDOWS = (12, 24, 168, 720)
WARMUP_DAYS = 60
VOL_WINDOWS = (24, 168)
ATR_WINDOW = 24
CHANNEL_WINDOWS = (24, 168)
SHORT_CHANNEL = 6
UP_SHARE_WINDOW = 24
VOLUME_Z_WINDOW = 168
ASYM_WINDOWS = (24, 72)
RSI_WINDOWS = (12, 24)
RET_HOURS = (3, 6, 12)

TARGET = "y"
NON_FEATURES = ["y", "y_full", "hour_end"]

FEATURE_GROUPS = {
    "тренд": ["ma168_change_pct_prev", "ma720_change_pct_prev", "ma168_dist_prev",
              "ma720_dist_prev", "ma168_vs_ma720_prev", "ma168_dist_delta24_prev",
              "ma168_dist_delta72_prev"],
    "импульс": ["ma12_change_pct_prev", "ma24_change_pct_prev", "ma12_dist_prev",
                "ma24_dist_prev", "ma12_vs_ma24_prev", "ma24_vs_ma168_prev",
                "ret3h_prev", "ret6h_prev", "ret12h_prev"],
    "волатильность": ["vol24_prev", "vol168_prev", "vol_ratio_prev", "atr24_prev",
                      "range_pct_prev"],
    "положение": ["close_pos_prev", "channel_pos24_prev", "channel_pos168_prev",
                  "up_share24_prev", "asym24_prev", "asym72_prev"],
    "возврат": ["rsi12_prev", "rsi24_prev", "channel_pos6_prev", "dist_high24_prev",
                "dist_low24_prev", "streak_prev"],
    "объём": ["volume_z168_prev", "volume_change_pct_prev"],
    "час": ["change_pct_prev", "y_prev"],
    "календарь": ["hour_et", "hour_sin", "hour_cos", "dow"],
}


# --------------------------------------------------------------------------
# Свечи
# --------------------------------------------------------------------------

def hourly_candles(m1: pd.DataFrame, cutoff: int = 0) -> pd.DataFrame:
    """Часовые свечи на сетке рынка. `cutoff` — сколько последних минут отбросить.

    Метка свечи — время её **начала** (HH:00:00), как у Binance. Свеча с меткой H
    закрывается в H:59:59.999 и полностью известна к (H+1):00:00.

    При cutoff=k свеча собирается из минуток [:00 … :(59−k)] и известна уже
    к :(60−k):00 — то есть за k минут до конца часа.
    """
    if cutoff:
        m1 = m1[m1.index.minute <= (59 - cutoff)]
    out = m1.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


def market_target(m1: pd.DataFrame) -> pd.Series:
    """Исход рынка для свечи H: close >= open той же **полной** свечи.

    Равенство даёт Up — так написано в правиле резолва ("greater than or equal").
    """
    full = hourly_candles(m1, cutoff=0)
    return (full["close"] >= full["open"]).astype("int8").rename("y")


# --------------------------------------------------------------------------
# Индикаторы
# --------------------------------------------------------------------------

def _indicators(hist: pd.DataFrame, ma: str = "ema") -> pd.DataFrame:
    """Состояние на конец каждой (возможно обрезанной) свечи."""
    close, high, low, volume = hist["close"], hist["high"], hist["low"], hist["volume"]
    ret = (close / close.shift(1) - 1) * 100
    out = pd.DataFrame(index=hist.index)

    mas = {w: _ma(close, w, ma) for w in MA_WINDOWS}
    for w in MA_WINDOWS:
        out[f"ma{w}_change_pct"] = (mas[w] / mas[w].shift(1) - 1) * 100
        out[f"ma{w}_dist"] = _dist_pct(close, mas[w])
    for short, long in ((12, 24), (24, 168), (168, 720)):
        out[f"ma{short}_vs_ma{long}"] = _dist_pct(mas[short], mas[long])
    for lag in (24, 72):
        out[f"ma168_dist_delta{lag}"] = out["ma168_dist"] - out["ma168_dist"].shift(lag)

    for n in RET_HOURS:
        out[f"ret{n}h"] = (close / close.shift(n) - 1) * 100

    for w in VOL_WINDOWS:
        out[f"vol{w}"] = ret.rolling(w, min_periods=w).std()
    out["vol_ratio"] = out[f"vol{VOL_WINDOWS[0]}"] / out[f"vol{VOL_WINDOWS[1]}"]

    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr24"] = true_range.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean() / close * 100
    out["range_pct"] = (high - low) / close * 100
    out["close_pos"] = ((close - low) / (high - low)).replace([np.inf, -np.inf], np.nan)

    for w in (SHORT_CHANNEL,) + CHANNEL_WINDOWS:
        hi = high.rolling(w, min_periods=w).max()
        lo = low.rolling(w, min_periods=w).min()
        out[f"channel_pos{w}"] = ((close - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    hi24 = high.rolling(24, min_periods=24).max()
    lo24 = low.rolling(24, min_periods=24).min()
    out["dist_high24"] = _dist_pct(close, hi24)
    out["dist_low24"] = _dist_pct(close, lo24)

    up = (ret > 0).astype("float64").where(ret.notna())
    out["up_share24"] = up.rolling(UP_SHARE_WINDOW, min_periods=UP_SHARE_WINDOW).mean()
    for w in ASYM_WINDOWS:
        out[f"asym{w}"] = _asymmetry(ret, w)

    for w in RSI_WINDOWS:
        out[f"rsi{w}"] = _rsi(close, w)
    out["streak"] = _streak(ret)

    mean_v = volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).mean()
    std_v = volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).std()
    out["volume_z168"] = (volume - mean_v) / std_v
    out["volume_change_pct"] = (volume / volume.shift(1) - 1) * 100

    out["change_pct"] = ret
    # Знак самой свечи — по её собственным open/close, как резолвится рынок.
    out["y"] = (hist["close"] >= hist["open"]).astype("float64")
    return out


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Час и день недели по нью-йоркскому времени. Известны заранее, утечкой не являются."""
    et = idx.tz_convert(bd.ET)
    return pd.DataFrame({
        "hour_et": et.hour,
        "hour_sin": np.sin(2 * np.pi * et.hour / 24),
        "hour_cos": np.cos(2 * np.pi * et.hour / 24),
        "dow": et.dayofweek,
    }, index=idx)


# --------------------------------------------------------------------------
# Сборка
# --------------------------------------------------------------------------

def build(m1: pd.DataFrame, cutoff: int = 2, ma: str = "ema") -> pd.DataFrame:
    """Таблица: строка — час H (рынок), признаки — по свече H−1, обрезанной на `cutoff`.

    Проверка на утечку простая и жёсткая: признаки строки H берутся с метки H−1
    обрезанного ряда, то есть из данных, закончившихся в (H−1):(59−cutoff):59.999.
    Ничего из самого часа H в них не попадает.
    """
    hist = hourly_candles(m1, cutoff=cutoff)
    ind = _indicators(hist, ma)

    y = market_target(m1)
    index = y.index                                   # метки часов-рынков
    prev = index - pd.Timedelta(hours=1)

    out = pd.DataFrame(index=index)
    for col in ind.columns:
        out[f"{col}_prev"] = ind[col].reindex(prev).values
    out = out.join(_calendar(pd.DatetimeIndex(index)))
    out[TARGET] = y.values
    out.index.name = "hour_end"

    out = out.dropna()
    out[TARGET] = out[TARGET].astype("int8")
    out["y_prev"] = out["y_prev"].astype("int8")
    return out


def feature_names(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in NON_FEATURES]


def make_xy(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_names(feats)
    return feats[cols].astype("float64"), feats[TARGET].astype("int8")


def load(start: str = "2023-07-28", cutoff: int = 2, ma: str = "ema",
         verbose: bool = True) -> pd.DataFrame:
    """Грузит минутки с запасом на разгон и собирает признаки."""
    first = pd.Timestamp(start).normalize()
    m1 = bd.load_1m(start=first - pd.Timedelta(days=WARMUP_DAYS), verbose=False)
    feats = build(m1, cutoff=cutoff, ma=ma)
    feats = feats[feats.index >= first.tz_localize("UTC")]
    if verbose:
        print(f"  обрезка {cutoff} мин: {len(feats):,} строк × "
              f"{len(feature_names(feats))} признаков, "
              f"{feats.index[0]:%Y-%m-%d %H:%M} → {feats.index[-1]:%Y-%m-%d %H:%M} UTC, "
              f"доля Up {feats[TARGET].mean()*100:.2f}%")
    return feats
