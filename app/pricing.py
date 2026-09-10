"""Решение: на какую сторону ставим, по какой цене и стоит ли вообще.

Механика бинарного рынка. Шейр покупается за цену `p`, выигравший гасится за $1,
проигравший за $0. При `shares` шейрах:

    выигрыш:  shares × (1 − p)        проигрыш:  −shares × p

Матожидание нулевое при вероятности выигрыша ровно `p`, то есть **точка
безубыточности численно равна цене входа**. Купил по 48 центов — хватит 48%
угаданных, по 50 — нужно 50%.

Отсюда же берётся потолок цены. Модель переуверена (наклон калибровки 0.756 у BTC,
0.806 у ETH —
README.md §5 комплекта): выше обещанной вероятности сбывается меньше.
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
    #: Факт, а не выбор: True — в момент решения аск уже ≤ нашей цены, и заявка
    #: исполнится сразу тейкером (целиком или частично); False — ляжет в книгу.
    #: На решение о ставке не влияет — это нормальный исход, см. `decide`.
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
           best_ask: float | None = None,
           shares: float = config.SHARES) -> Decision:
    """Что делать при данном прогнозе.

    Сторону выбираем по прогнозу, цену — фиксированную из конфига (LIMIT_PRICE).
    Фиксированная цена, а не подстройка под стакан, выбрана намеренно: весь расчёт
    доходности в отчёте сделан именно для неё.

    `shares` — размер ставки; по умолчанию `config.SHARES`, но вызывающий код
    почти всегда передаёт его явно (`storage.current_shares()` в момент решения,
    либо уже зафиксированный размер этого рынка при повторных попытках) — сам
    дефолт вычисляется один раз при импорте модуля и не подхватит `/shares`.

    Правило исполнения одно: обычная лимитная заявка по LIMIT_PRICE, без
    post-only. `best_ask` — лучшее предложение по выбранной стороне — только
    отмечается в `Decision.cross`: если аск уже не выше нашей цены, заявка
    исполнится сразу тейкером, и это нормальный исход, а не повод отказаться или
    ждать. Раньше здесь сравнивалось матожидание тейкера с пассивным, и при
    встречке по нашей цене заявка ждала пассивной постановки. Замер на настоящем
    стакане (747 часов BTC, исполнение по очереди, см. блок «исполнение» в
    config.py) показал, что ожидание стоит дороже пересечения: пассивную заявку
    наливают в основном тогда, когда цена идёт против позиции (точность на
    исполненных 50.38% против 56.22%). Итог новой схемы +0.0470 на шейр в час
    против +0.0143: цена 0.51 вместо 0.49 дала +0.0171, отказ от post-only
    +0.0081, добор тейкером на 12-й минуте (`runner.cross_remaining`) +0.0075.
    Комиссия тейкера по формуле в замеренный период фактически не списывалась,
    поэтому в матожидание решения она не входит.
    """
    side = "Up" if proba_up >= 0.5 else "Down"
    price = round_to_tick(config.LIMIT_PRICE, tick)

    if price > config.MAX_ENTRY_PRICE + 1e-9:
        return Decision(None, price, 0, proba_up, 0.0,
                        f"цена {price:.2f} выше потолка {config.MAX_ENTRY_PRICE:.2f}")

    ev = expected_value(proba_up, price, side, shares=shares)
    if ev <= 0:
        # Безубыточность равна цене входа: при 0.51 модель должна давать выбранной
        # стороне больше 51%, иначе ставить не на чем.
        return Decision(None, price, 0, proba_up, ev,
                        f"матожидание {ev:+.3f}$ не положительно")

    marketable = best_ask is not None and best_ask <= price + 1e-9
    how = (f"ask {best_ask:.2f} ≤ цены — исполнится сразу" if marketable
           else "лимиткой в книгу")
    return Decision(side, price, shares, proba_up, ev,
                    f"p(Up)={proba_up:.4f}, EV={ev:+.3f}$, {how}", cross=marketable)


def describe(d: Decision) -> str:
    if not d.bet:
        return f"пропуск — {d.reason}"
    return (f"{d.side} × {d.shares:g} шейров по {d.price:.2f} "
            f"[{'сразу тейкером' if d.cross else 'лимиткой в книгу'}] "
            f"(p(Up)={d.proba_up:.4f}, EV={d.ev_usd:+.3f}$, риск {d.shares*d.price:.2f}$)")
