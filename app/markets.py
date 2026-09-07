"""Поиск часового рынка и чтение стакана.

Рынок находится по слагу, который строится из времени: часовой рынок
`bitcoin-up-or-down-july-29-2026-11am-et` — это свеча, начинающаяся в 11:00 ET
(15:00 UTC летом) и заканчивающаяся в 16:00 UTC.

Валидация обязательна перед каждым ордером. Polymarket уже переводил часть рынков
BTC Up/Down на Chainlink (семейство `btc-updown-5m/15m/4h`), а модель обучена
на Binance BTC/USDT — торговать «похожим» рынком нельзя. Поэтому проверяется
источник резолва, формула, исходы и то, что endDate совпадает с ожидаемым часом.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import httpx
import pandas as pd

import config
from app.log import get

log = get("markets")

_ET = "America/New_York"
_TIMEOUT = 20.0
_RETRIES = 3

#: Признаки правильного рынка. Меняются вместе с правилами Polymarket — если
#: валидация начнёт падать на всех рынках, сначала смотреть сюда, а не отключать её.
REQUIRED_SOURCE = "binance.com"
REQUIRED_PAIR = "BTC/USDT"
REQUIRED_FORMULA = "greater than or equal to the open price"
EXPECTED_OUTCOMES = ["Up", "Down"]


class MarketError(RuntimeError):
    """Рынок не найден или не соответствует тому, на чём обучена модель."""


@dataclass
class MarketInfo:
    hour_start: pd.Timestamp
    slug: str
    condition_id: str
    token_up: str
    token_down: str
    end_date: pd.Timestamp
    accepting_orders: bool
    min_size: float
    tick_size: float
    #: Рынок из связки «взаимоисключающие исходы». У часовых Up/Down всегда False,
    #: но сборщик ордеров подписывает их другим контрактом, поэтому берём из API.
    neg_risk: bool
    best_bid_up: float | None
    best_ask_up: float | None

    def token(self, side: str) -> str:
        return self.token_up if side == "Up" else self.token_down


def slug_for(hour_start: pd.Timestamp) -> str:
    """Слаг рынка для часа, начинающегося в `hour_start` (UTC)."""
    et = pd.Timestamp(hour_start).tz_convert(_ET)
    hour12 = et.hour % 12 or 12
    suffix = "am" if et.hour < 12 else "pm"
    month = et.strftime("%B").lower()
    return (f"bitcoin-up-or-down-{month}-{et.day}-{et.year}-{hour12}{suffix}-et")


def _get(url: str, params: dict | None = None) -> list | dict:
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            r = httpx.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:                      # noqa: BLE001 — важен любой сбой
            last = e
            log.warning("request {} failed ({}/{}): {}", url, attempt + 1, _RETRIES, e)
    raise MarketError(f"не удалось получить {url}: {last}")


def fetch(hour_start: pd.Timestamp, closed: bool | None = None) -> dict:
    """Рынок по часу.

    Закрытые Gamma отдаёт только с явным `closed=true`, поэтому по умолчанию
    (`closed=None`) сначала спрашиваем открытые, а потом закрытые: иначе всё, что
    работает с уже прошедшими часами — сверка исполнения, резолв, разбор истории —
    падает на «рынок не найден».
    """
    slug = slug_for(hour_start)
    attempts = (False, True) if closed is None else (closed,)
    for want_closed in attempts:
        params = {"slug": slug}
        if want_closed:
            params["closed"] = "true"
        data = _get(f"{config.GAMMA_HOST}/markets", params)
        if data:
            return data[0]
    raise MarketError(f"рынок {slug} не найден")


def validate(raw: dict, hour_start: pd.Timestamp) -> None:
    """Проверяет, что найденный рынок — тот самый, под который обучена модель."""
    problems: list[str] = []

    src = (raw.get("resolutionSource") or "").lower()
    desc = raw.get("description") or ""
    if REQUIRED_SOURCE not in src and REQUIRED_SOURCE not in desc.lower():
        problems.append(f"источник резолва не Binance: {src!r}")
    if REQUIRED_PAIR not in desc:
        problems.append(f"в описании нет пары {REQUIRED_PAIR}")
    if REQUIRED_FORMULA not in desc:
        # Именно эта формула — close >= open одной свечи. Модель обучена на ней.
        problems.append("формула резолва отличается от close >= open")

    outcomes = raw.get("outcomes")
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if outcomes != EXPECTED_OUTCOMES:
        problems.append(f"исходы {outcomes} вместо {EXPECTED_OUTCOMES}")

    end = pd.Timestamp(raw.get("endDate"))
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    expected_end = pd.Timestamp(hour_start) + pd.Timedelta(hours=1)
    if end != expected_end:
        problems.append(f"endDate {end} вместо {expected_end}")

    if not raw.get("enableOrderBook"):
        problems.append("у рынка нет стакана")

    if problems:
        raise MarketError(f"{raw.get('slug')}: " + "; ".join(problems))


def book(token_id: str) -> dict:
    """Лучшие цены по одному токену. Пустой стакан — не ошибка, а состояние рынка."""
    try:
        data = _get(f"{config.CLOB_HOST}/book", {"token_id": token_id})
    except MarketError:
        return {"bid": None, "ask": None}
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    # CLOB отдаёт лестницу от худшей цены к лучшей.
    best_bid = max((float(b["price"]) for b in bids), default=None)
    best_ask = min((float(a["price"]) for a in asks), default=None)
    return {"bid": best_bid, "ask": best_ask}


#: Когда правило резолва проверялось в последний раз. Проверка не нужна перед каждым
#: ордером — правила рынка меняются раз в месяцы, а не раз в час. Но совсем без неё
#: нельзя: Polymarket уже переводил семейство BTC Up/Down (5m/15m/4h) на Chainlink,
#: и модель, обученная на Binance, торговала бы не тот рынок.
_validated_at: pd.Timestamp | None = None
VALIDATE_EVERY = pd.Timedelta(hours=24)


def ensure_valid(hour_start: pd.Timestamp, force: bool = False) -> None:
    """Проверить правило резолва, но не чаще раза в сутки.

    Вызывается при старте и периодически из цикла, а не из горячего пути ордера.
    """
    global _validated_at
    now = pd.Timestamp.now(tz="UTC")
    if not force and _validated_at is not None and now - _validated_at < VALIDATE_EVERY:
        return
    raw = fetch(pd.Timestamp(hour_start))
    validate(raw, pd.Timestamp(hour_start))
    _validated_at = now
    log.info("resolution rule verified on {} — Binance BTC/USDT, close >= open",
             raw.get("slug"))


def find(hour_start: pd.Timestamp, with_book: bool = True) -> MarketInfo:
    """Найти рынок для указанного часа.

    Правило резолва здесь **не** проверяется — этим занимается `ensure_valid`
    по расписанию.
    """
    hour_start = pd.Timestamp(hour_start)
    raw = fetch(hour_start)

    tokens = raw.get("clobTokenIds")
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    if not tokens or len(tokens) != 2:
        raise MarketError(f"{raw.get('slug')}: нет пары токенов")

    end = pd.Timestamp(raw["endDate"])
    info = MarketInfo(
        hour_start=hour_start,
        slug=raw["slug"],
        condition_id=raw["conditionId"],
        token_up=tokens[0],
        token_down=tokens[1],
        end_date=end if end.tzinfo else end.tz_localize("UTC"),
        accepting_orders=bool(raw.get("acceptingOrders")),
        min_size=float(raw.get("orderMinSize") or 5),
        tick_size=float(raw.get("orderPriceMinTickSize") or 0.001),
        neg_risk=bool(raw.get("negRisk")),
        best_bid_up=None, best_ask_up=None,
    )
    if with_book:
        b = book(info.token_up)
        info.best_bid_up, info.best_ask_up = b["bid"], b["ask"]
    return info


def snapshot(info: MarketInfo) -> dict:
    """Полный снимок стакана обеих сторон — для записи в БД."""
    up, down = book(info.token_up), book(info.token_down)
    spread = (up["ask"] - up["bid"]) if (up["ask"] and up["bid"]) else None
    return {"best_bid_up": up["bid"], "best_ask_up": up["ask"],
            "best_bid_down": down["bid"], "best_ask_down": down["ask"],
            "spread_up": spread}


def outcome_of(hour_start: pd.Timestamp) -> str | None:
    """Исход по данным Polymarket: "Up" / "Down" / None, если ещё не резолвлен."""
    for closed in (True, False):
        try:
            raw = fetch(hour_start, closed=closed)
        except MarketError:
            continue
        prices = raw.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices:
            continue
        up = float(prices[0])
        if up > 0.99:
            return "Up"
        if up < 0.01:
            return "Down"
    return None            # ещё торгуется или спорный


def outcome_from_binance(hour_start: pd.Timestamp) -> str | None:
    """Исход, посчитанный самостоятельно по правилу резолва: close >= open свечи.

    Нужен по двум причинам. Во-первых, он доступен сразу после конца часа, а Polymarket
    резолвит с задержкой. Во-вторых, это независимая сверка: расхождение с их резолвом
    означает, что либо у нас неверные данные, либо правило рынка изменилось — и то,
    и другое надо заметить сразу, а не через месяц по кривой P&L.
    """
    import btc_data as bd            # локально: тянет за собой pandas-тяжёлый модуль

    hour_start = pd.Timestamp(hour_start)
    end = hour_start + pd.Timedelta(hours=1)
    if pd.Timestamp.now(tz="UTC") < end:
        return None
    m1 = bd.load_1m(start=hour_start - pd.Timedelta(minutes=5),
                    cache_dir=config.CACHE_DIR, verbose=False)
    candle = m1[(m1.index >= hour_start) & (m1.index < end)]
    if candle.empty or candle.index[-1] < end - pd.Timedelta(minutes=1):
        return None                  # свеча ещё не полная
    return "Up" if float(candle["close"].iloc[-1]) >= float(candle["open"].iloc[0]) else "Down"
