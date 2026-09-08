"""Риск-лимиты и предполётные проверки.

Философия остановок взята из исследования: **останавливаться по накопленному
результату, а не по серии промахов**. Серия из 12 промахов подряд для этой модели —
статистическая норма (проверено на 16 тысячах часов), а вот минус за месяц — уже
повод разбираться.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import config
from app import storage
from app.log import get

log = get("risk")


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


def _day_start_et(now: dt.datetime) -> dt.datetime:
    from zoneinfo import ZoneInfo
    et = now.astimezone(ZoneInfo("America/New_York"))
    return et.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(dt.timezone.utc)


def _month_start_et(now: dt.datetime) -> dt.datetime:
    from zoneinfo import ZoneInfo
    et = now.astimezone(ZoneInfo("America/New_York"))
    return et.replace(day=1, hour=0, minute=0, second=0,
                      microsecond=0).astimezone(dt.timezone.utc)


def check(now: dt.datetime | None = None) -> Verdict:
    """Можно ли ставить прямо сейчас."""
    now = now or dt.datetime.now(dt.timezone.utc)

    if not config.TRADING_ENABLED:
        return Verdict(False, "торговля выключена в конфиге")

    with storage.session() as s:
        open_n = len(storage.open_positions(s))
        if open_n >= config.MAX_OPEN_POSITIONS:
            return Verdict(False, f"открытых позиций {open_n} "
                                  f"(лимит {config.MAX_OPEN_POSITIONS})")

        day = storage.pnl_since(s, _day_start_et(now))
        if day <= -abs(config.DAILY_LOSS_LIMIT_USD):
            return Verdict(False, f"дневной убыток {day:.2f}$ достиг лимита "
                                  f"{config.DAILY_LOSS_LIMIT_USD:.2f}$")

        month = storage.pnl_since(s, _month_start_et(now))
        if month <= -abs(config.MONTHLY_LOSS_LIMIT_USD):
            return Verdict(False, f"месячный убыток {month:.2f}$ достиг лимита "
                                  f"{config.MONTHLY_LOSS_LIMIT_USD:.2f}$ — "
                                  "требуется ручная проверка модели")

    return Verdict(True)


def preflight() -> list[str]:
    """Проверки при старте. Непустой список — бот не должен торговать."""
    problems: list[str] = []

    if config.SHARES < 5:
        problems.append("SHARES < 5: биржа не примет ордер меньше orderMinSize")
    if not 0 < config.LIMIT_PRICE < 1:
        problems.append(f"LIMIT_PRICE={config.LIMIT_PRICE} вне (0,1)")
    if config.LIMIT_PRICE > config.MAX_ENTRY_PRICE:
        problems.append("LIMIT_PRICE выше потолка MAX_ENTRY_PRICE")

    try:
        from app import model
        card = model.card()
        # Сверка обрезки и монеты уже происходит при импорте app/model.py
        # (падает сразу, если комплект обучен не на том cutoff или не на той
        # паре, что в конфиге) — здесь достаточно убедиться, что модель вообще
        # загрузилась и не пустая.
        if not card.get("model_id"):
            problems.append("модель загрузилась, но в паспорте нет model_id")
    except Exception as e:                                   # noqa: BLE001
        problems.append(f"модель не загружается: {e}")

    if not config.DRY_RUN and not config.PRIVATE_KEY:
        problems.append("боевой режим без POLYMARKET_PRIVATE_KEY в .env")

    return problems
