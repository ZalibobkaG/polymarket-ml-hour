"""Отправка ордеров на Polymarket CLOB.

Два исполнителя с одинаковым интерфейсом:

* `PaperTrader` — ничего никуда не отправляет, а моделирует исполнение по стакану.
  Работает в DRY_RUN.
* `ClobTrader` — настоящая торговля через py-clob-client.

Заявок по часу бывает две:

* лимитка по LIMIT_PRICE типа **GTD**, живёт до `hour_start + CROSS_AFTER_SECONDS − 5 с`
  (≈ H+11:55). Это страховка: даже если бот упадёт, заявка умрёт сама до момента
  добора и не будет исполняться дальше в часе, когда налив — это уже в основном
  цена против нас;
* добор тейкером — тип **FAK**: исполняется то, что есть в стакане до лимита,
  остаток биржа сразу снимает, в книге ничего не остаётся.

Модель исполнения в бумажном режиме — «по касанию»: лимитка считается налитой, как
только лучший ask на очередном опросе (раз в TICK_SECONDS, 5 с)
опустился до нашей цены. Ошибка у неё в обе стороны: очередь перед нами не учитывается (оптимистично), а проскоки
цены между опросами не видны (пессимистично). На записи стакана 27.07–06.09.2026
она легла между двумя вариантами модели очереди из реплея (`app/replay.py`): доля
мейкера 52% против 57% (очередь съедают продажи нашего токена и покупки парного)
и 37% (только продажи). Цифрам бумаги по исполнению доверять как ориентиру, не больше.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

import config
from app.log import get
from app.markets import MarketInfo, book

log = get("trader")


@dataclass
class OrderResult:
    ok: bool
    order_id: str | None = None
    error: str | None = None
    #: Биржа отклонила ордер, потому что он исполнился бы тейкером. Не ошибка.
    post_only_rejected: bool = False


@dataclass
class Leg:
    """Один кусок исполнения нашего ордера — по данным сделки, а не заявки."""

    price: float
    shares: float
    #: "MAKER" / "TAKER" — сторона сделки на бирже.
    role: str
    #: Комиссия, которую биржа начислила за этот кусок. None — биржа не сообщила.
    fee_charged: float | None = None


def passive_expiration(info: MarketInfo) -> int:
    """Unix-время, когда лимитка умирает сама: за 5 секунд до добора тейкером."""
    end = pd.Timestamp(info.hour_start) + pd.Timedelta(
        seconds=config.CROSS_AFTER_SECONDS - 5)
    return int(end.timestamp())


def _now_ts() -> float:
    """Текущее unix-время. Отдельной функцией — чтобы реплей мог подставить свои часы."""
    return time.time()


#: Признаки отказа из-за post_only. Отказ ожидаем и означает «встречка есть».
_POST_ONLY_REJECT = ("post only", "post-only", "postonly", "marketable",
                     "would match", "would cross", "crossing")


def is_post_only_reject(err: str) -> bool:
    e = (err or "").lower()
    return any(s in e for s in _POST_ONLY_REJECT)


@dataclass
class _PaperOrder:
    token: str
    price: float
    shares: float
    #: Unix-время, после которого лимитка не исполняется; 0 — FAK, в книге не живёт.
    expires: float
    legs: list[Leg] = field(default_factory=list)
    cancelled: bool = False

    @property
    def filled(self) -> float:
        return sum(leg.shares for leg in self.legs)


class PaperTrader:
    """Бумажная торговля: решения настоящие, деньги нет.

    Заявки помнятся только в памяти процесса. После перезапуска бумажная заявка
    текущего часа считается снятой: налить её уже некому, а добор на 12-й минуте
    доберёт остаток — так же, как в бою после отмены.
    """

    live = False

    def __init__(self) -> None:
        self._orders: dict[str, _PaperOrder] = {}

    def place(self, info: MarketInfo, side: str, price: float, shares: float,
              post_only: bool = False, order_type: str = "GTD") -> OrderResult:
        kind = "cross" if order_type == "FAK" else "limit"
        oid = f"paper-{info.hour_start:%Y%m%d%H}-{side.lower()}-{kind}"
        ask = book(info.token(side)).get("ask")
        marketable = ask is not None and ask <= price + 1e-9
        # Бумажная имитация post-only: если встречка по нашей цене уже есть,
        # биржа отклонила бы ордер — воспроизводим это, иначе бумага будет
        # оптимистичнее боя.
        if post_only and marketable:
            log.warning("[paper] order rejected post-only: ask={:.3f} ≤ {:.3f}",
                       ask, price)
            return OrderResult(ok=False, post_only_rejected=True,
                               error="paper: would cross")
        order = _PaperOrder(token=info.token(side), price=price, shares=shares,
                            expires=passive_expiration(info) if order_type == "GTD" else 0,
                            cancelled=order_type != "GTD")
        if marketable:
            # По касанию: весь размер берётся по лучшему аску, глубину не смотрим.
            order.legs.append(Leg(ask, shares, "TAKER", 0.0))
        self._orders[oid] = order
        log.success("[paper] order {} {} × {:g} at {:.2f} [{}{}] ({})", info.slug,
                    side, shares, price, order_type,
                    f", taker at {ask:.3f}" if marketable else "", oid)
        return OrderResult(ok=True, order_id=oid)

    def legs(self, info: MarketInfo, order_id: str | None) -> list[Leg]:
        """Куски исполнения бумажной заявки на сейчас.

        Лимитка, лежащая в книге, наливается целиком по своей цене, как только
        лучший ask опустился до неё (мы мейкер, продавец пришёл к нам). Очередь
        перед нами не учитывается, проскоки между опросами не видны — см.
        docstring модуля.
        """
        o = self._orders.get(order_id or "")
        if o is None:
            return []
        rest = o.shares - o.filled
        if not o.cancelled and rest > 1e-9 and _now_ts() < o.expires:
            ask = book(o.token).get("ask")
            if ask is not None and ask <= o.price + 1e-9:
                log.success("[paper] resting order {} hit: ask={:.3f} ≤ {:.3f}",
                            order_id, ask, o.price)
                o.legs.append(Leg(o.price, rest, "MAKER", 0.0))
        return list(o.legs)

    def cancel_confirmed(self, order_id: str) -> tuple[bool, float | None]:
        """Снять заявку. В бумаге отмена подтверждается сразу."""
        o = self._orders.get(order_id)
        if o is not None:
            o.cancelled = True
        log.info("[paper] cancel {}", order_id)
        return True, (o.filled if o is not None else None)

    def cancel(self, order_id: str) -> bool:
        return self.cancel_confirmed(order_id)[0]

    def open_orders(self) -> list[dict]:
        return []


#: Типы подписи, которые понимает клиент v2. Третий — подпись смарт-контракта
#: по EIP-1271, именно на нём живут современные депозитные кошельки Polymarket.
#: Старый пакет py-clob-client знает только 0..2 и на тройке падает ещё на сборке
#: ордера («Invalid order inputs»), поэтому проект переведён на v2.
SIGNATURE_TYPES = {0: "EOA", 1: "POLY_PROXY (email/Magic)",
                   2: "POLY_GNOSIS_SAFE (браузерный кошелёк)",
                   3: "POLY_1271 (депозитный кошелёк)"}


class ClobTrader:
    """Боевая торговля. Требует ключей в .env и подтверждённого DRY_RUN=False."""

    live = True

    def __init__(self) -> None:
        from py_clob_client_v2.client import ClobClient

        if not config.PRIVATE_KEY:
            raise RuntimeError("нет POLY_PRIVATE_KEY в .env")
        if config.SIGNATURE_TYPE not in SIGNATURE_TYPES:
            raise RuntimeError(
                f"POLY_SIGNATURE_TYPE={config.SIGNATURE_TYPE} — неизвестный тип подписи, "
                f"допустимы {sorted(SIGNATURE_TYPES)}")

        kwargs = dict(host=config.CLOB_HOST, key=config.PRIVATE_KEY,
                      chain_id=config.CHAIN_ID,
                      signature_type=config.SIGNATURE_TYPE)
        if config.FUNDER_ADDRESS:
            kwargs["funder"] = config.FUNDER_ADDRESS
        self.client = ClobClient(**kwargs)
        log.info("signature: type {} — {}{}", config.SIGNATURE_TYPE,
                 SIGNATURE_TYPES[config.SIGNATURE_TYPE],
                 f", funder {config.FUNDER_ADDRESS[:10]}…" if config.FUNDER_ADDRESS else "")

        # L2-креды: либо из .env, либо выводим из ключа (детерминированно —
        # один и тот же ключ всегда даёт те же креды, заводить их руками не нужно).
        if config.API_KEY and config.API_SECRET and config.API_PASSPHRASE:
            from py_clob_client_v2.clob_types import ApiCreds
            self.client.set_api_creds(ApiCreds(
                api_key=config.API_KEY, api_secret=config.API_SECRET,
                api_passphrase=config.API_PASSPHRASE))
        else:
            # Клиент сначала пробует СОЗДАТЬ ключ, получает 400 «Could not create
            # api key» (он уже существует) и только потом выводит его из подписи.
            # Итог успешный, но по дороге печатается ERROR — глушим только её.
            import logging
            noisy = logging.getLogger("py_clob_client_v2.http_helpers.helpers")
            was = noisy.level
            noisy.setLevel(logging.CRITICAL)
            try:
                self.client.set_api_creds(self.client.create_or_derive_api_key())
            finally:
                noisy.setLevel(was)
        log.success("CLOB client ready (py-clob-client-v2)")
        self.check_funds()

    @property
    def creds(self):
        """L2-креды текущей сессии — нужны WebSocket-каналу своих ордеров."""
        return getattr(self.client, "creds", None)

    def check_funds(self) -> float | None:
        """Баланс USDC и разрешения на списание.

        `signature_type` обязателен: без него API отдаёт баланс EOA (нули), а не
        депозитного кошелька, где лежат деньги.
        """
        try:
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
            r = self.client.get_balance_allowance(BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=config.SIGNATURE_TYPE))
        except Exception as e:                                # noqa: BLE001
            log.warning("could not check balance: {}", e)
            return None

        balance = float(r.get("balance", 0)) / 1e6            # USDC в 6 знаках
        allowances = r.get("allowances") or {}
        approved = sum(1 for v in allowances.values() if float(v) > 0)
        log.info("USDC balance: {:.2f}$, allowances granted: {}", balance, approved)
        if not approved:
            log.error("no approval on any contract — orders will be rejected; "
                      "grant an allowance in the Polymarket UI")
        need = config.SHARES * config.LIMIT_PRICE
        if balance < need:
            log.error("balance {:.2f}$, need {:.2f}$ for the bet", balance, need)
        return balance

    #: Публичный (без ключей) Data-API Polymarket — им же пользуется их сайт для
    #: строки «Portfolio». Отдаёт суммарную стоимость открытых позиций по адресу;
    #: portfolio = эта стоимость + кэш, ровно как в интерфейсе.
    _VALUE_API = "https://data-api.polymarket.com/value"

    def account_value(self) -> tuple[float, float] | None:
        """(cash, portfolio) в долларах — как на сайте Polymarket, или None."""
        if not config.FUNDER_ADDRESS:
            return None
        cash = self.check_funds()
        if cash is None:
            return None
        try:
            import httpx
            r = httpx.get(self._VALUE_API, params={"user": config.FUNDER_ADDRESS},
                          timeout=10.0)
            r.raise_for_status()
            data = r.json()
            positions = float(data[0]["value"]) if data else 0.0
        except Exception as e:                                # noqa: BLE001
            log.warning("could not fetch portfolio value: {}", e)
            return None
        return cash, cash + positions

    def build_order(self, info: MarketInfo, side: str, price: float, shares: float,
                    expiration: int | None = None):
        """Собрать и подписать ордер. Ничего не отправляет — только локально.

        Вынесено отдельно, чтобы предполётная проверка могла пройти ровно этот путь
        и поймать отказ сборки заранее, а не за две минуты до часа.

        `expiration` — unix-время смерти заявки. По умолчанию лимитка живёт до
        `passive_expiration` (≈ H+11:55): сама умрёт до добора, даже если бот упадёт.
        Для FAK передаётся 0 — такая заявка в книге не живёт вовсе.
        """
        from py_clob_client_v2.clob_types import (OrderArgsV2,
                                                  PartialCreateOrderOptions)
        from py_clob_client_v2.order_builder.constants import BUY

        if expiration is None:
            expiration = passive_expiration(info)
        args = OrderArgsV2(token_id=info.token(side), price=round(price, 4),
                           size=float(shares), side=BUY, expiration=expiration)
        opts = PartialCreateOrderOptions(tick_size=f"{info.tick_size:g}",
                                         neg_risk=info.neg_risk)
        return self.client.create_order(args, opts)

    #: build_order только читает биржу (tick-size и т.п.) и подписывает локально —
    #: до post_order ничего не отправлено, поэтому повтор здесь ничем не рискует.
    #: Так лечится редкий транзиентный сбой сети на GET /tick-size, который
    #: py_clob_client_v2 сам не ретраит (в отличие от post()).
    _BUILD_RETRIES = 3
    _BUILD_RETRY_DELAY = 0.4

    def _build_order_retrying(self, info: MarketInfo, side: str, price: float,
                              shares: float, expiration: int | None = None):
        for attempt in range(1, self._BUILD_RETRIES + 1):
            try:
                return self.build_order(info, side, price, shares, expiration)
            except Exception as e:                                # noqa: BLE001
                if attempt == self._BUILD_RETRIES:
                    raise
                log.warning("build_order attempt {}/{} failed: {} — retrying",
                           attempt, self._BUILD_RETRIES, e)
                time.sleep(self._BUILD_RETRY_DELAY)

    def place(self, info: MarketInfo, side: str, price: float, shares: float,
              post_only: bool = False, order_type: str = "GTD") -> OrderResult:
        """Отправить заявку. `order_type`: "GTD" — лимитка в книгу до
        `passive_expiration`; "FAK" — добор: исполнить что есть до цены, остаток снять."""
        from py_clob_client_v2.clob_types import OrderType

        try:
            signed = self._build_order_retrying(
                info, side, price, shares,
                expiration=None if order_type == "GTD" else 0)
            resp = self.client.post_order(signed, getattr(OrderType, order_type),
                                          post_only)
        except Exception as e:                                # noqa: BLE001
            if post_only and is_post_only_reject(str(e)):
                # Expected and frequent while we wait for the crossing order to
                # clear — DEBUG, not INFO: `runner` logs a WARNING/SUCCESS summary
                # once the retry series ends, so this stays out of the way.
                log.debug("order rejected post-only: our price is crossed, "
                         "not going taker")
                return OrderResult(ok=False, post_only_rejected=True, error=str(e))
            log.exception("order not sent")
            return OrderResult(ok=False, error=str(e))

        if not resp or not resp.get("success", True):
            err = str(resp.get("errorMsg") or resp) if isinstance(resp, dict) else str(resp)
            if post_only and is_post_only_reject(err):
                log.debug("order rejected post-only: {}", err)
                return OrderResult(ok=False, post_only_rejected=True, error=err)
            return OrderResult(ok=False, error=err)
        oid = resp.get("orderID") or resp.get("orderId")
        log.success("order sent {} {} × {:g} at {:.2f} [{}{}] → {}",
                    info.slug, side, shares, price, order_type,
                    ", post-only" if post_only else "", oid)
        return OrderResult(ok=True, order_id=oid)

    def legs(self, info: MarketInfo, order_id: str | None) -> list[Leg]:
        """Наши куски исполнения по ордеру — из сделок, а не из заявки.

        Единственный надёжный источник цены. `get_order` отдаёт цену **заявки**, а
        исполниться она могла лучше: наш лимит 0.51 при живом предложении 0.47
        покупается по 0.47. Из сделки берём именно нашу ногу — при пассивном
        исполнении верхнеуровневые `price`/`size` принадлежат тейкеру, а наша доля
        лежит в `maker_orders[]`.

        Комиссия по факту. Тейкерская нога: ставка сделки `fee_rate_bps` — 0
        значит «биржа не взяла», иначе сумма по той же форме, что FEE_RATE
        (ставка·p·(1−p)); поля нет — None, неизвестно. Мейкер по правилам
        биржи комиссию не платит: `fee_rate_bps` в `maker_orders[]` — это
        потолок, подписанный в заявке, а не списание. Цифру по тейкеру стоит один
        раз сверить с `usdcSize` из Data API на первых боевых сделках.
        """
        if not order_id:
            return []
        from py_clob_client_v2.clob_types import TradeParams

        trades = self.client.get_trades(TradeParams(market=info.condition_id))
        trades = trades if isinstance(trades, list) else (trades or {}).get("data", [])
        legs: list[Leg] = []
        for t in trades:
            if str(t.get("status", "")).upper() not in ("MATCHED", "CONFIRMED"):
                continue
            if t.get("taker_order_id") == order_id:
                p, s = float(t["price"]), float(t["size"])
                bps = t.get("fee_rate_bps")
                fee = (None if bps in (None, "") else
                       s * float(bps) / 1e4 * p * (1 - p))
                legs.append(Leg(p, s, "TAKER", fee))
            for m in t.get("maker_orders") or []:
                if m.get("order_id") == order_id:
                    legs.append(Leg(float(m["price"]),
                                    float(m.get("matched_amount") or 0), "MAKER", 0.0))
        return [leg for leg in legs if leg.shares > 0]

    #: Сколько ждать, пока биржа подтвердит отмену, прежде чем отложить добор на
    #: следующий тик. Добор без подтверждения мог бы купить сверх размера ставки.
    _CANCEL_CONFIRM_SECONDS = 5.0

    def cancel_confirmed(self, order_id: str) -> tuple[bool, float | None]:
        """Снять заявку и убедиться, что она больше не может исполниться.

        Возвращает (подтверждено, исполнено по заявке шейров). Подтверждение —
        статус заявки у биржи уже не LIVE: снята, исполнена целиком или истекла.
        """
        try:
            self.client.cancel_orders([order_id])
        except Exception as e:                                # noqa: BLE001
            # Не приговор: заявка могла уже истечь или исполниться — это и
            # проверяется ниже по её статусу.
            log.warning("cancel request for {} failed: {}", order_id, e)
        until = time.monotonic() + self._CANCEL_CONFIRM_SECONDS
        while True:
            try:
                o = self.client.get_order(order_id) or {}
                status = str(o.get("status", "")).upper()
                if status and status not in ("LIVE", "OPEN", "UNMATCHED", "DELAYED"):
                    matched = float(o.get("size_matched") or 0)
                    log.info("order {} is {} — matched {:g}", order_id, status, matched)
                    return True, matched
            except Exception as e:                            # noqa: BLE001
                log.warning("could not read order {}: {}", order_id, e)
            if time.monotonic() >= until:
                log.warning("order {}: cancel not confirmed in {:.0f}s",
                            order_id, self._CANCEL_CONFIRM_SECONDS)
                return False, None
            time.sleep(0.5)

    def cancel(self, order_id: str) -> bool:
        try:
            self.client.cancel_orders([order_id])
            log.info("order {} cancelled", order_id)
            return True
        except Exception as e:                                # noqa: BLE001
            log.warning("could not cancel {}: {}", order_id, e)
            return False

    def open_orders(self) -> list[dict]:
        try:
            r = self.client.get_open_orders()
            return r if isinstance(r, list) else (r or {}).get("data", []) or []
        except Exception as e:                                # noqa: BLE001
            log.warning("could not fetch open orders: {}", e)
            return []


def build() -> PaperTrader | ClobTrader:
    """Исполнитель по режиму из конфига."""
    if config.DRY_RUN:
        log.info("PAPER mode: orders are not sent to the exchange")
        return PaperTrader()
    log.warning("LIVE mode: orders are sent to the exchange for real")
    return ClobTrader()
