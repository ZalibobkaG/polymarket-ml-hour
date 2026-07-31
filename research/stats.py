"""Интервалы и тесты, не предполагающие независимости часов (§4 из ML_REVIEW.md).

Биномиальный тест трактует 15 965 часов как независимые броски монетки. Это не так:
признаки соседних часов считаются по перекрывающимся окнам (720-часовая скользящая
у соседних часов различается одним наблюдением из 720), волатильность кластеризуется,
и попадания модели могут слипаться. Если слипаются — настоящий интервал шире
биномиального, и утверждение «нижняя граница выше 51%» может оказаться сильнее,
чем данные позволяют.

Лечение стандартное: блочный бутстреп. Ресемплим не отдельные часы, а целые блоки
подряд идущих часов (по умолчанию сутки), сохраняя тем самым внутриблоковую
зависимость какой она есть в данных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def binom_ci(hits: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Точный биномиальный интервал — как в оригинале, для сравнения."""
    lo, hi = binomtest(hits, n).proportion_ci(confidence_level=conf, method="exact")
    return float(lo), float(hi)


def _blocks(index: pd.DatetimeIndex, block: str) -> list[np.ndarray]:
    """Позиции наблюдений, сгруппированные в блоки по календарю."""
    key = pd.PeriodIndex(pd.DatetimeIndex(index).tz_convert("UTC").tz_localize(None),
                         freq=block)
    order = np.argsort(key.asi8, kind="stable")
    keys_sorted = key.asi8[order]
    cuts = np.flatnonzero(np.diff(keys_sorted)) + 1
    return np.split(order, cuts)


def block_bootstrap_ci(values: np.ndarray, index: pd.DatetimeIndex,
                       block: str = "D", n_boot: int = 2000, conf: float = 0.95,
                       seed: int = 0) -> dict:
    """Перцентильный интервал для среднего с блочным ресемплингом.

    values — что усредняем (для точности это булев вектор попаданий, для денег —
    прибыль в долларах на ставку). Блоки берутся целиком, с возвращением, пока не
    наберётся исходное число наблюдений.
    """
    values = np.asarray(values, dtype="float64")
    blocks = _blocks(pd.DatetimeIndex(index), block)
    sizes = np.array([len(b) for b in blocks])
    sums = np.array([values[b].sum() for b in blocks])
    rng = np.random.default_rng(seed)
    n_blocks, total = len(blocks), len(values)

    # Ресемплим столько блоков, сколько их было: суммарный размер выборки колеблется
    # вокруг исходного, что для среднего несущественно.
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)

    a = (1 - conf) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return {
        "оценка": float(values.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "se": float(means.std(ddof=1)),
        "блоков": n_blocks, "наблюдений": total,
        # Двусторонний бутстрепный p-value против нуля: доля ресемплов по ту сторону.
        "p_two_sided": float(2 * min((means <= 0.5).mean(), (means >= 0.5).mean()))
        if np.all((values == 0) | (values == 1)) else np.nan,
    }


def paired_block_bootstrap(a: np.ndarray, b: np.ndarray, index: pd.DatetimeIndex,
                           block: str = "D", n_boot: int = 2000, conf: float = 0.95,
                           seed: int = 0) -> dict:
    """Интервал для разницы средних на одних и тех же наблюдениях.

    Парность важна: обе стратегии предсказывают одни и те же часы, и общий для них шум
    рынка из разницы уходит. Ресемплятся те же блоки для обеих сразу.
    """
    diff = np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")
    blocks = _blocks(pd.DatetimeIndex(index), block)
    sizes = np.array([len(x) for x in blocks])
    sums = np.array([diff[x].sum() for x in blocks])
    rng = np.random.default_rng(seed)

    draws = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    a_q = (1 - conf) / 2
    lo, hi = np.quantile(means, [a_q, 1 - a_q])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"разница": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(p, 1.0))}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Точный парный тест: информацию несут только расхождения.

    a, b — булевы векторы «угадал». Считаем часы, где угадал ровно один из двух.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    if n01 + n10 == 0:
        return {"только A": 0, "только B": 0, "p": 1.0}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"только A": n01, "только B": n10, "p": float(p)}
