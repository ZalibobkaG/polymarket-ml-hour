"""Решение: на какую сторону ставим, по какой цене и стоит ли вообще.

Механика бинарного рынка. Шейр покупается за цену `p`, выигравший гасится за $1,
проигравший за $0. При `shares` шейрах:

    выигрыш:  shares × (1 − p)        проигрыш:  −shares × p

Матожидание нулевое при вероятности выигрыша ровно `p`, то есть **точка
безубыточности численно равна цене входа**. Купил по 48 центов — хватит 48%
угаданных, по 50 — нужно 50%.

Отсюда же берётся потолок цены. Модель переуверена (наклон калибровки 0.756 —
poly_hourly_kit/README.md §5): выше обещанной вероятности сбывается меньше.
Поэтому выше MAX_ENTRY_PRICE не покупаем даже при формально положительном
матожидании — там правило `q > цена` уходит в минус.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import config


@dataclass
class Decision:
    side: str | None            # "Up" / "Down" / None — не ставим
    price: float
    shares: float
    proba_up: float
    ev_usd: float
    reason: str
    #: False — ставим пассивно (биржа отклонит ордер, который взял бы комиссию).
    #: True — сознательно пересекаем стакан, потому что цена окупает комиссию.
    cross: bool = False

    @property
    def bet(self) -> bool:
        return self.side is not None


def fee_per_share(price: float) -> float:
    """Комиссия тейкера на один шейр. Максимальна у 0.50 и гаснет к краям."""
    return config.FEE_RATE * price * (1 - price)


def rebate_per_share(price: float) -> float:
    """Ребейт мейкеру на один шейр. Считаем как бонус, в матожидание не берём."""
    return config.MAKER_REBATE_SHARE * fee_per_share(price)


def expected_value(proba_up: float, price: float, side: str,
                   shares: float = config.SHARES, taker: bool = False) -> float:
    """Матожидание ставки в долларах.

    Комиссия тейкера включается, ребейт мейкера — нет: он приходит отдельной
    программой и на нашем объёме может не начислиться. Пусть будет запасом, а не
    частью расчёта.
    """
    w = proba_up if side == "Up" else 1.0 - proba_up
    ev = w * shares * (1 - price) - (1 - w) * shares * price
    return ev - shares * fee_per_share(price) if taker else ev


def round_to_tick(price: float, tick: float) -> float:
    """Округление к сетке цен биржи — вниз, чтобы не переплатить."""
    if tick <= 0:
        return price
    return math.floor(price / tick + 1e-9) * tick


def decide(proba_up: float, tick: float = 0.01,
           best_ask: float | None = None) -> Decision:
    """Что делать при данном прогнозе.

    Сторону выбираем по прогнозу, цену — фиксированную из конфига (LIMIT_PRICE).
    Фиксированная цена, а не подстройка под стакан, выбрана намеренно: весь расчёт
    доходности в отчёте сделан именно для неё.

    Отдельно решается, пересекать ли стакан. `best_ask` — лучшее предложение по
    выбранной стороне. Если оно не ниже нашей цены, ставим пассивно и комиссию не
    платим. Если ниже — мы всё равно исполнимся тейкером, поэтому проверяем, что
    матожидание с комиссией остаётся положительным; на 0.47 это выгоднее пассивной
    постановки, на 0.50 — почти ноль.
    """
    side = "Up" if proba_up >= 0.5 else "Down"
    price = round_to_tick(config.LIMIT_PRICE, tick)

    if price > config.MAX_ENTRY_PRICE + 1e-9:
        return Decision(None, price, 0, proba_up, 0.0,
                        f"цена {price:.2f} выше потолка {config.MAX_ENTRY_PRICE:.2f}")

    ev = expected_value(proba_up, price, side)
    if ev <= 0:
        # При фиксированной цене 0.50 это значит, что модель почти не отклонилась
        # от монетки: ставить не на чем.
        return Decision(None, price, 0, proba_up, ev,
                        f"матожидание {ev:+.3f}$ не положительно")

    crossing = best_ask is not None and best_ask <= price + 1e-9
    if not crossing:
        return Decision(side, price, config.SHARES, proba_up, ev,
                        f"p(Up)={proba_up:.4f}, EV={ev:+.3f}$, пассивно")

    # Сравниваем не с нулём, а с пассивным вариантом: важно, что выгоднее, а не что
    # просто прибыльно. Дешёвое предложение окупает комиссию (0.47 даёт эффективную
    # цену 0.4874 против 0.50), цена вровень с нашей — нет.
    ev_taker = expected_value(proba_up, best_ask, side, taker=True)
    if ev_taker <= ev:
        # Пересечение невыгодно. Всё равно отправляем пассивно: если стакан
        # шевельнётся, ордер ляжет в книгу, а если нет — биржа его отклонит,
        # и это правильный исход, а не потеря.
        return Decision(side, price, config.SHARES, proba_up, ev,
                        f"p(Up)={proba_up:.4f}, EV={ev:+.3f}$; ask {best_ask:.2f} "
                        f"тейкером дал бы {ev_taker:+.3f}$ — только пассивно")

    return Decision(side, price, config.SHARES, proba_up, ev_taker,
                    f"p(Up)={proba_up:.4f}, ask {best_ask:.2f}, EV тейкером "
                    f"{ev_taker:+.3f}$ против {ev:+.3f}$ пассивно — берём",
                    cross=True)


def describe(d: Decision) -> str:
    if not d.bet:
        return f"пропуск — {d.reason}"
    return (f"{d.side} × {d.shares:g} шейров по {d.price:.2f} "
            f"[{'тейкер' if d.cross else 'пассивно'}] "
            f"(p(Up)={d.proba_up:.4f}, EV={d.ev_usd:+.3f}$, риск {d.shares*d.price:.2f}$)")
