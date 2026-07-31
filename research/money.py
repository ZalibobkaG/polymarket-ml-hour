"""Арифметика ставок на Polymarket: из предсказаний в деньги.

Механика бинарного рынка. Шейр покупается за цену p долларов; выигравший шейр
погашается за $1, проигравший за $0. Комиссии за исполнение лимитного ордера нет.

Ставим фиксированную сумму STAKE на каждый рынок, то есть покупаем STAKE/p шейров:

    выигрыш:  STAKE/p × $1 − STAKE = STAKE × (1−p)/p
    проигрыш: −STAKE

При p = 0.50 это ровно ±STAKE. При p = 0.48 выигрыш больше проигрыша, при 0.51 — меньше.

Отсюда главное соотношение всей этой затеи. Матожидание нулевое, когда

    w × (1−p)/p = 1 − w   ⟺   w = p

то есть **точка безубыточности численно равна цене входа**. Купил по 48 центов — хватит
48% угаданных; купил по 51 — нужен 51%. Поэтому цена лимитки решает не меньше, чем модель:
между 48 и 51 центами — три процентных пункта требуемой точности, а всё преимущество
модели над монеткой составляет 2.45.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

#: Сколько долларов ставим на каждый рынок.
STAKE = 10.0


def payoff(price: float, stake: float = STAKE) -> float:
    """Сколько приносит выигравшая ставка (проигравшая всегда стоит -stake)."""
    return stake * (1 - price) / price


def pnl(win: np.ndarray, price: float, stake: float = STAKE) -> np.ndarray:
    """Прибыль по каждой ставке в долларах."""
    return np.where(np.asarray(win, dtype=bool), payoff(price, stake), -stake)


def drawdown(equity: np.ndarray) -> tuple[float, np.ndarray]:
    """Максимальная просадка от предыдущего максимума и сама кривая просадки."""
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min()), dd


def evaluate(win: np.ndarray, price: float, stake: float = STAKE,
             months: pd.PeriodIndex | None = None) -> dict:
    """Полная сводка по одной стратегии при одной цене входа."""
    win = np.asarray(win, dtype=bool)
    n, hits = len(win), int(win.sum())
    if n == 0:
        return {"ставок": 0}

    profit = pnl(win, price, stake)
    equity = np.cumsum(profit)
    max_dd, _ = drawdown(equity)
    turnover = n * stake
    # Односторонний тест: вопрос «прибыльнее ли безубыточности», а не «отличается ли».
    test = binomtest(hits, n, price, alternative="greater")

    out = {
        "ставок": n,
        "угадано, %": hits / n * 100,
        "нужно, %": price * 100,
        "запас, п.п.": hits / n * 100 - price * 100,
        "выигрыш даёт, $": payoff(price, stake),
        "итог, $": float(equity[-1]),
        "на ставку, $": float(profit.mean()),
        "ROI, %": float(profit.sum() / turnover * 100),
        "просадка, $": max_dd,
        "p-value": test.pvalue,
    }
    if months is not None:
        by_m = monthly(profit, months)
        out["месяцев в минусе"] = int((by_m < 0).sum())
        out["месяцев всего"] = int(len(by_m))
        # Насколько убыточный месяц вообще ожидаем при таком преимуществе: месячная
        # прибыль — сумма ~700 независимых ставок, то есть почти нормальна.
        from scipy.stats import norm
        out["ожидаемая доля минусовых, %"] = float(
            norm.cdf(-by_m.mean() / by_m.std(ddof=1)) * 100)
    return out


#: Месяц короче этого числа ставок — огрызок на краю периода, в помесячную разбивку
#: не идёт: границы кварталов проходят по UTC, а месяцы считаются по нью-йоркскому.
MIN_MONTH_BETS = 200


def monthly(profit: np.ndarray, months) -> pd.Series:
    """Прибыль по месяцам, без огрызков на краях периода."""
    g = pd.Series(profit).groupby(np.asarray(months))
    return g.sum()[g.size() >= MIN_MONTH_BETS]


def equity_curve(win: np.ndarray, price: float, stake: float = STAKE) -> np.ndarray:
    return np.cumsum(pnl(win, price, stake))


def streaks(flags: np.ndarray) -> np.ndarray:
    """Длины всех серий подряд идущих True."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return np.array([0])
    # Границы серий: там, где значение меняется.
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    return edges[1::2] - edges[::2]


def expected_max_streak(q: float, n: int) -> float:
    """Ожидаемая длина самой длинной серии события с вероятностью q за n попыток.

    Нужна для сравнения: без неё непонятно, двенадцать промахов подряд — это модель
    сломалась или обычная арифметика. Классическая оценка: log(n(1−q)) / log(1/q).
    """
    if not 0 < q < 1:
        return float("nan")
    return float(np.log(n * (1 - q)) / np.log(1 / q))


def breakeven_gap(accuracy: float, price: float) -> float:
    """На сколько процентных пунктов точность выше требуемой."""
    return (accuracy - price) * 100
