"""Разметка фаз рынка: медвежья / боковик / бычья.

ВАЖНО про причинность (см. reports/honesty.md):
- phase_label_report() сглаживает отрезки короче 21 дня. Сглаживание смотрит на соседние дни,
  в том числе последующие. Использовать ТОЛЬКО для отчётных разбивок.
- phase_label_causal() без сглаживания. Каждая метка зависит только от прошлого.
  ТОЛЬКО эту версию допустимо использовать как признак модели.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BEAR, SIDE, BULL = "медвежья", "боковик", "бычья"
PHASES = [BEAR, SIDE, BULL]
PHASE_CODE = {BEAR: -1.0, SIDE: 0.0, BULL: 1.0}

R90_BULL, R90_BEAR, DD_BEAR, MIN_SEGMENT_DAYS = 0.20, -0.20, -0.25, 21


def _daily_close(df: pd.DataFrame) -> pd.Series:
    return df.set_index("open_time")["close"].resample("1D").last().dropna()


def _raw_labels(dc: pd.Series) -> pd.Series:
    """Сырая разметка. Все три условия смотрят только назад."""
    r90 = dc.pct_change(90)
    drawdown = dc / dc.cummax() - 1
    sma200 = dc.rolling(200).mean()
    lab = pd.Series(SIDE, index=dc.index)
    lab[(r90 > R90_BULL) & (dc > sma200)] = BULL
    lab[(r90 < R90_BEAR) | ((drawdown < DD_BEAR) & (dc < sma200))] = BEAR
    return lab


def phase_label_causal(df: pd.DataFrame) -> pd.Series:
    """Причинная разметка по часам. Пригодна как признак."""
    lab = _raw_labels(_daily_close(df))
    return pd.Series(lab.reindex(df["open_time"].dt.normalize()).values, index=df.index)


def phase_label_report(df: pd.DataFrame) -> pd.Series:
    """Сглаженная разметка. НЕ причинная. Только для отчётов и разбивок."""
    lab = _raw_labels(_daily_close(df))
    short = lab.groupby((lab != lab.shift()).cumsum()).transform("size") < MIN_SEGMENT_DAYS
    lab = lab.mask(short).ffill().bfill()
    return pd.Series(lab.reindex(df["open_time"].dt.normalize()).values, index=df.index)


def phase_code(labels: pd.Series) -> pd.Series:
    return labels.map(PHASE_CODE).astype(float)


def coverage(labels: pd.Series) -> dict:
    vc = labels.value_counts(normalize=True)
    return {p: float(vc.get(p, 0.0)) for p in PHASES}
