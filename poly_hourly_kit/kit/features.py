"""Признаки и целевая переменная. Всё считается только по прошлому.

Один и тот же набор признаков работает на всех трёх таймфреймах (5m, 15m, 1h).

ОКНА СЧИТАЮТСЯ В БАРАХ, А НЕ В ЧАСАХ. Это осознанный выбор: модель смотрит на форму
движения, а не на календарь. `r24` на пятнадцатиминутках — это 6 часов, и это правильный
перевод. Числа окон (1, 2, 3, 6, 12, 24, 48, 168, 14) для всех таймфреймов одинаковы,
меняется только цена деления.

Список FEATURES зафиксирован: ровно те 22 признака, на которых измерены
контрольные числа из reports/step1c_walkforward.md (планка 52.72%, LightGBM 54.01%).
Добавление или удаление признака делает эти числа недействительными и требует
нового полного прогона walk-forward.

Календарные признаки (час суток, день недели) остаются календарными на всех
таймфреймах: hsin/hcos считаются от целого часа. На 15m четыре бара подряд получают
одинаковое значение, на 5m — двенадцать. Так и задумано: это метка времени суток,
а не позиция внутри часа.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.data import INTERVAL_MS

FEATURES = [
    "r1", "r2", "r3", "r6", "r12", "r24", "r48",
    "z24", "z168", "vol_ratio", "atr_rel",
    "bodyn", "lower", "upper",
    "volrel", "trrel", "rsi", "streak_signed",
    "hsin", "hcos", "dsin", "dcos",
]
MAX_LOOKBACK_BARS = 168


def bar_step(timeframe: str) -> pd.Timedelta:
    """Длительность одного бара таймфрейма. Единственное место, где признаки
    зависят от таймфрейма."""
    if timeframe not in INTERVAL_MS:
        raise ValueError(
            f"Неизвестный таймфрейм '{timeframe}'. "
            f"Доступны: {', '.join(INTERVAL_MS)}."
        )
    return pd.Timedelta(milliseconds=INTERVAL_MS[timeframe])


def build(df: pd.DataFrame, timeframe: str) -> dict:
    """Возвращает X (признаки), y (таргет), fwd (доходность следующей свечи),
    valid (следующая свеча существует и идёт ровно через один бар), naive (наивное правило).

    timeframe обязателен: молчаливо посчитать 5m как 1h хуже, чем упасть с ошибкой.
    """
    step = bar_step(timeframe)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v, trd, t = df["volume"], df["trades"].astype(float), df["open_time"]

    r = {k: c.pct_change(k) for k in (1, 2, 3, 6, 12, 24, 48)}
    sma24, sma168 = c.rolling(24).mean(), c.rolling(168).mean()
    std24, std168 = r[1].rolling(24).std(), r[1].rolling(168).std()
    rng = (h - l).replace(0, np.nan)
    body = (c > o).astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    streak = body.groupby((body != body.shift()).cumsum()).cumcount() + 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    X = pd.DataFrame({
        **{f"r{k}": r[k] for k in r},
        "z24": (c / sma24 - 1) / std24,
        "z168": (c / sma168 - 1) / std168,
        "vol_ratio": std24 / std168,
        "atr_rel": atr / c,
        "bodyn": (c - o) / rng,
        "lower": (np.minimum(o, c) - l) / rng,
        "upper": (h - np.maximum(o, c)) / rng,
        # 0 и inf убираются ДО логарифма, иначе numpy сыплет предупреждениями;
        # итоговые значения те же самые (NaN), меняется только шум в консоли
        "volrel": np.log((v / v.rolling(24).mean()).replace([0, np.inf, -np.inf], np.nan)),
        "trrel": np.log((trd / trd.rolling(24).mean()).replace([0, np.inf, -np.inf], np.nan)),
        "rsi": rsi / 100,
        "streak_signed": streak * np.where(body == 1, 1, -1),
        "hsin": np.sin(2 * np.pi * t.dt.hour / 24),
        "hcos": np.cos(2 * np.pi * t.dt.hour / 24),
        "dsin": np.sin(2 * np.pi * t.dt.dayofweek / 7),
        "dcos": np.cos(2 * np.pi * t.dt.dayofweek / 7),
    })[FEATURES].replace([np.inf, -np.inf], np.nan)

    y = (c.shift(-1) > o.shift(-1)).astype(float)
    y[c.shift(-1).isna()] = np.nan
    fwd = c.shift(-1) / o.shift(-1) - 1
    valid = (t.diff().shift(-1) == step)
    naive = pd.Series(np.where(streak >= 2, 1 - body, 1.0), index=df.index, dtype=float)

    return {"df": df, "X": X, "y": y, "fwd": fwd, "valid": valid, "naive": naive,
            "ready": X.notna().all(axis=1)}
