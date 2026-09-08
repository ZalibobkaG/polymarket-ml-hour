"""Метрики. Одна формула — одно место, чтобы отчёт и проверка не разошлись.

Главное правило: точность в одиночку не показывается никогда. Рядом всегда идёт
среднее движение за сделку в базисных пунктах. Причина простая — правило может
выигрывать часто по мелочи и проигрывать реже, но крупнее, и по одной точности
этого не видно.

Второе правило: две точности сравниваются ТОЛЬКО парным тестом Макнемара на
одних и тех же барах. Смотреть, пересекаются ли доверительные интервалы, нельзя:
они пересекаются даже при настоящей разнице.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z95 = 1.96


def acc_ci(n: int) -> float:
    """Половина 95% доверительного интервала точности в п.п. (консервативно, p=0.5)."""
    return Z95 * math.sqrt(0.25 / n) * 100 if n > 0 else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Интервал Уилсона для доли, в долях единицы.

    Не нормальное приближение: оно на краях вылезает за границы 0..1 и рисует
    отрицательную нижнюю границу. Уилсон из отрезка [0, 1] не выходит.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def edge_bp(pred: np.ndarray, fwd: np.ndarray) -> tuple[float, float]:
    """Среднее движение за сделку в б.п. и половина 95% интервала.

    Сделка: вход по открытию часа, выход по его закрытию, комиссии нет.
    """
    if len(pred) == 0:
        return float("nan"), float("nan")
    e = np.where(pred == 1, 1.0, -1.0) * fwd
    m = float(e.mean() * 1e4)
    ci = float(Z95 * e.std(ddof=1) / np.sqrt(len(e)) * 1e4) if len(e) > 1 else float("nan")
    return m, ci


def mcnemar(hit_a: np.ndarray, hit_b: np.ndarray) -> float:
    """Парный тест Макнемара с поправкой Йейтса на одних и тех же барах."""
    from scipy.stats import chi2

    b = int((hit_a & ~hit_b).sum())
    c = int((~hit_a & hit_b).sum())
    if b + c == 0:
        return 1.0
    # sf, а не 1-cdf: в далёком хвосте 1-cdf схлопывается в ровный ноль,
    # и очень маленькое p печаталось бы как невозможное p = 0.
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def cell(pred: pd.Series, y: pd.Series, fwd: pd.Series, mask: pd.Series) -> dict | None:
    """Точность и движение одного среза. None — если баров меньше 40.

    Возвращает и сам массив попаданий `hit`: он нужен парному тесту.
    """
    m = mask & pred.notna()
    n = int(m.sum())
    if n < 40:
        return None
    p, yy, ff = pred[m].to_numpy(), y[m].to_numpy(), fwd[m].to_numpy()
    bp, bp_ci = edge_bp(p, ff)
    return dict(n=n, acc=float((p == yy).mean()), ci=acc_ci(n), bp=bp, bp_ci=bp_ci,
                hit=(p == yy))


def calibration(proba: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Децильные корзины и наклон «сбылось ~ обещано».

    Корзины равного РАЗМЕРА, а не равной ширины: прогнозы лежат в узком коридоре,
    и равномерная сетка по [0, 1] схлопнула бы всё в две корзины.

    Наклон 1.0 — идеальная калибровка. Меньше единицы — модель обещает больше,
    чем делает, и тогда потолок цены входа ниже её обещанной вероятности.
    """
    ok = np.isfinite(proba) & np.isfinite(y)
    p, yy = proba[ok], y[ok]
    if p.size < n_bins * 10:
        return [], float("nan")

    order = np.argsort(p, kind="mergesort")
    rows = []
    for ch in np.array_split(order, n_bins):
        if ch.size == 0:
            continue
        rows.append({"n": int(ch.size), "lo": float(p[ch].min()), "hi": float(p[ch].max()),
                     "promised": float(p[ch].mean()), "realized": float(yy[ch].mean())})

    n = np.array([r["n"] for r in rows], dtype=float)
    xs = np.array([r["promised"] for r in rows])
    ys = np.array([r["realized"] for r in rows])
    xm, ym = np.average(xs, weights=n), np.average(ys, weights=n)
    den = float((n * (xs - xm) ** 2).sum())
    slope = float((n * (xs - xm) * (ys - ym)).sum() / den) if den > 0 else float("nan")
    return rows, slope
