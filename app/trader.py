"""Отправка ордеров на Polymarket CLOB.

Два исполнителя с одинаковым интерфейсом:

* `PaperTrader` — ничего никуда не отправляет, а моделирует исполнение по стакану.
  Работает в DRY_RUN и нужен не «для галочки»: он отвечает на главный незакрытый
  вопрос исследования — как часто лимитку по нашей цене вообще налили бы.
* `ClobTrader` — настоящая торговля через py-clob-client.

Ордер выставляется типом **GTD** с истечением в конце часа. Это главная страховка:
даже если бот упадёт, ордер умрёт сам и не повиснет до следующего дня.

Модель исполнения в бумажном режиме консервативная: считаем ордер исполненным только
если рынок реально давал нашу цену, то есть лучший ask ≤ нашей цены покупки. Это
пессимистичнее, чем «стоял в стакане час — значит налили», и ближе к правде: в реальности
нас нальют тем охотнее, чем меньше нам это выгодно.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import config
from app.log import get
from app.markets import MarketInfo, book

log = get("trader")


@dataclass
class OrderResult:
    ok: bool
    order_id: str | None = None
    filled_shares: float = 0.0
    fill_price: float | None = None
    error: str | None = None
    #: "MAKER" / "TAKER" / None — по факту исполнения, а не по нашему намерению.
    role: str | None = None
    #: Комиссия тейкера в долларах (положительное число = мы заплатили).
    fee_usd: float = 0.0
    #: Биржа отклонила ордер, потому что он исполнился бы тейкером. Не ошибка.
    post_only_rejected: bool = False


#: Признаки отказа из-за post_only. Отказ ожидаем и означает «встречка есть».
_POST_ONLY_REJECT = ("post only", "post-only", "postonly", "marketable",
                     "would match", "would cross", "crossing")


def is_post_only_reject(err: str) -> bool:
    e = (err or "").lower()
    return any(s in e for s in _POST_ONLY_REJECT)


class PaperTrader:
    """Бумажная торговля: решения настоящие, деньги нет."""

    live = False

    def place(self, info: MarketInfo, side: str, price: float,
              shares: float, post_only: bool = False) -> OrderResult:
        oid = f"paper-{info.hour_start:%Y%m%d%H}-{side.lower()}"
        # Бумажная имитация post-only: если встречка по нашей цене уже есть,
        # биржа отклонила бы ордер — воспроизводим это, иначе бумага будет
        # оптимистичнее боя.
        ask = book(info.token(side)).get("ask")
        if post_only and ask is not None and ask <= price + 1e-9:
            log.warning("[paper] order rejected post-only: ask={:.3f} ≤ {:.3f}",
                       ask, price)
            return OrderResult(ok=False, post_only_rejected=True,
                               error="paper: would cross")
        log.success("[paper] order {} {} × {:g} at {:.2f} [{}] ({})", info.slug, side,
                    shares, price, "passive" if post_only else "can go taker", oid)
        return OrderResult(ok=True, order_id=oid)

    def poll_fill(self, info: MarketInfo, side: str, price: float,
                  shares: float) -> OrderResult:
        """Исполнился бы ордер по текущему стакану?

        Покупка исполняется, когда кто-то готов продать не дороже нашей цены,
        то есть best_ask ≤ price. Цену считаем по предложению, а не по своей
        заявке — так же, как в бою.
        """
        b = book(info.token(side))
        ask = b.get("ask")
        if ask is not None and ask <= price + 1e-9:
            log.success("[paper] filled {}: ask={:.3f} ≤ {:.3f}", info.slug, ask, price)
            fee = shares * config.FEE_RATE * ask * (1 - ask)
            return OrderResult(ok=True, filled_shares=shares, fill_price=ask,
                               role="TAKER", fee_usd=fee)
        return OrderResult(ok=True, filled_shares=0.0)

    def cancel(self, order_id: str) -> bool:
        log.info("[paper] cancel {}", order_id)
        return True

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

    def build_order(self, info: MarketInfo, side: str, price: float, shares: float):
        """Собрать и подписать ордер. Ничего не отправляет — только локально.

        Вынесено отдельно, чтобы предполётная проверка могла пройти ровно этот путь
        и поймать отказ сборки заранее, а не за две минуты до часа.
        """
        from py_clob_client_v2.clob_types import (OrderArgsV2,
                                                  PartialCreateOrderOptions)
        from py_clob_client_v2.order_builder.constants import BUY

        # Ордер живёт до конца часа минус запас: сам умрёт, даже если бот упадёт.
        expiration = int((info.end_date - dt.timedelta(seconds=30)).timestamp())
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
                              shares: float):
        for attempt in range(1, self._BUILD_RETRIES + 1):
            try:
                return self.build_order(info, side, price, shares)
            except Exception as e:                                # noqa: BLE001
                if attempt == self._BUILD_RETRIES:
                    raise
                log.warning("build_order attempt {}/{} failed: {} — retrying",
                           attempt, self._BUILD_RETRIES, e)
                time.sleep(self._BUILD_RETRY_DELAY)

    def place(self, info: MarketInfo, side: str, price: float,
              shares: float, post_only: bool = False) -> OrderResult:
        from py_clob_client_v2.clob_types import OrderType

        try:
            signed = self._build_order_retrying(info, side, price, shares)
            resp = self.client.post_order(signed, OrderType.GTD, post_only)
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
        log.success("order sent {} {} × {:g} at {:.2f} [{}] → {}",
                    info.slug, side, shares, price,
                    "passive" if post_only else "can go taker", oid)
        return OrderResult(ok=True, order_id=oid)

    def our_legs(self, info: MarketInfo, order_id: str) -> list[tuple[float, float, str]]:
        """Наши куски исполнения по ордеру: [(цена, шейры, роль)].

        Единственный надёжный источник цены. `get_order` отдаёт цену **заявки**, а
        исполниться она могла лучше: наш лимит 0.50 при живом предложении 0.47
        покупается по 0.47. Из сделки берём именно нашу ногу — при пассивном
        исполнении верхнеуровневые `price`/`size` принадлежат тейкеру, а наша доля
        лежит в `maker_orders[]`.
        """
        from py_clob_client_v2.clob_types import TradeParams

        trades = self.client.get_trades(TradeParams(market=info.condition_id))
        trades = trades if isinstance(trades, list) else (trades or {}).get("data", [])
        legs: list[tuple[float, float, str]] = []
        for t in trades:
            if str(t.get("status", "")).upper() not in ("MATCHED", "CONFIRMED"):
                continue
            if t.get("taker_order_id") == order_id:
                legs.append((float(t["price"]), float(t["size"]), "TAKER"))
            for m in t.get("maker_orders") or []:
                if m.get("order_id") == order_id:
                    legs.append((float(m["price"]),
                                 float(m.get("matched_amount") or 0), "MAKER"))
        return [(p, s, r) for p, s, r in legs if s > 0]

    def poll_fill(self, info: MarketInfo, side: str, price: float,
                  shares: float, order_id: str | None = None) -> OrderResult:
        if not order_id:
            return OrderResult(ok=True, filled_shares=0.0)
        try:
            legs = self.our_legs(info, order_id)
        except Exception as e:                                # noqa: BLE001
            log.warning("could not read trades for order {}: {}", order_id, e)
            return OrderResult(ok=False, error=str(e))
        if not legs:
            return OrderResult(ok=True, filled_shares=0.0)

        filled = sum(s for _, s, _ in legs)
        vwap = sum(p * s for p, s, _ in legs) / filled
        roles = {r for _, _, r in legs}
        role = roles.pop() if len(roles) == 1 else "MIXED"
        # Комиссию платит только тейкер, и считается она по каждой ноге отдельно.
        fee = sum(s * config.FEE_RATE * p * (1 - p)
                  for p, s, r in legs if r == "TAKER")
        return OrderResult(ok=True, filled_shares=filled, fill_price=vwap,
                           role=role, fee_usd=fee)

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
