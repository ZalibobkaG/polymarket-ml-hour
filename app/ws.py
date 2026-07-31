"""WebSocket-подписки Polymarket CLOB.

Два канала:

* **user** — свои ордера и сделки. Требует L2-кредов и **непустого** списка рынков:
  с пустым сервер соединение держит, но событий не шлёт.
* **market** — стакан по конкретным токенам, без авторизации.

Работает в отдельном потоке-демоне: торговый цикл синхронный, и заворачивать его
в asyncio ради одной подписки смысла нет. Поток сам переподключается с растущей
паузой; если сокет умрёт совсем, опрос по REST в `runner` останется страховкой —
поэтому падение WebSocket не должно останавливать торговлю.

Сознательное решение: событие сделки здесь **не** используется как источник
истины по объёму. Оно лишь говорит «по этому ордеру что-то произошло», а сколько
именно налили — читается по REST. Причина в форме события: при пассивном
исполнении наш объём лежит в `maker_orders[].matched_amount`, при агрессивном —
в верхнеуровневом `size`, и верхнеуровневый `size` в первом случае означает объём
всей сделки, а не нашу долю. Разбирать это в двух местах — лишний источник ошибок.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Iterable

import websocket

import config
from app.log import get

log = get("ws")

USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

#: Пауза перед переподключением растёт до этого предела.
MAX_BACKOFF = 60.0
#: Polymarket закрывает молчащее соединение; шлём пинг чаще.
PING_INTERVAL = 20


class _Channel(threading.Thread):
    """Общая механика: подключиться, подписаться, читать, переподключаться."""

    daemon = True

    def __init__(self, url: str, subscribe: dict, on_event: Callable[[dict], None],
                 name: str, needs_markets: bool = False) -> None:
        super().__init__(name=name)
        self.url = url
        self.subscribe = subscribe
        self.on_event = on_event
        self.needs_markets = needs_markets
        self._stop = threading.Event()
        self._sock: websocket.WebSocket | None = None
        self._lock = threading.Lock()
        self.connected = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self._drop()

    def _drop(self) -> None:
        """Закрыть сокет из чужого потока, чтобы recv разблокировался."""
        with self._lock:
            sock, self._sock = self._sock, None
        if sock:
            try:
                sock.close()
            except Exception:                      # noqa: BLE001
                pass

    def update_markets(self, markets: Iterable[str]) -> None:
        """Сменить набор рынков и переподписаться.

        Подписка задаётся при установке соединения, поэтому смена набора означает
        переподключение. Для часового бота это раз в час — дёшево.
        """
        new = sorted({m for m in markets if m})
        if new == self.subscribe.get("markets"):
            return
        self.subscribe["markets"] = new
        log.info("%s: подписка на %d рынков, переподключение", self.name, len(new))
        self._drop()

    def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            if self.needs_markets and not self.subscribe.get("markets"):
                # Подписка без рынков молча не приносит событий — ждём первый ордер.
                time.sleep(1.0)
                continue
            try:
                self._session()
                backoff = 1.0                      # успешная сессия сбрасывает паузу
            except Exception as e:                 # noqa: BLE001
                if not self._stop.is_set():
                    log.warning("%s: сокет отвалился (%s), переподключение через %.0fс",
                                self.name, e, backoff)
            finally:
                self.connected.clear()
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    def _session(self) -> None:
        sock = websocket.create_connection(self.url, timeout=30)
        with self._lock:
            self._sock = sock
        try:
            sock.send(json.dumps(self.subscribe))
            self.connected.set()
            log.info("%s: подписка отправлена (%d рынков)",
                     self.name, len(self.subscribe.get("markets") or []))
            sock.settimeout(PING_INTERVAL)
            while not self._stop.is_set():
                with self._lock:
                    if self._sock is not sock:     # нас переподписали
                        return
                try:
                    raw = sock.recv()
                except websocket.WebSocketTimeoutException:
                    sock.ping()                    # держим соединение живым
                    continue
                if not raw:
                    continue
                self._dispatch(sock, raw)
        finally:
            try:
                sock.close()
            except Exception:                      # noqa: BLE001
                pass

    def _dispatch(self, sock, raw) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            if isinstance(raw, str) and raw.strip().upper() == "PING":
                try:
                    sock.send("PONG")
                except Exception:                  # noqa: BLE001
                    pass
            return
        events = data if isinstance(data, list) else [data]
        for e in events:
            if not isinstance(e, dict):
                continue
            try:
                self.on_event(e)
            except Exception:                      # noqa: BLE001
                log.exception("%s: обработчик события упал", self.name)


def user_channel(on_event: Callable[[dict], None],
                 markets: list[str] | None = None,
                 creds=None) -> _Channel | None:
    """Подписка на свои ордера и сделки.

    Креды берутся из уже созданного CLOB-клиента (он выводит их из приватного ключа
    детерминированно), а если его нет — из .env. Заводить их руками не требуется.
    Канал стартует сразу, но молчит, пока `update_markets` не задаст рынки.
    """
    api_key = getattr(creds, "api_key", None) or config.API_KEY
    secret = getattr(creds, "api_secret", None) or config.API_SECRET
    passphrase = getattr(creds, "api_passphrase", None) or config.API_PASSPHRASE

    if not (api_key and secret and passphrase):
        log.warning("нет L2-кредов — канал user не запущен, "
                    "исполнение отслеживается опросом по REST")
        return None
    sub = {
        "type": "user",
        "markets": sorted(markets or []),
        # Сервер в разных версиях читает разное написание ключа — шлём оба.
        "auth": {"apiKey": api_key, "apikey": api_key,
                 "secret": secret, "passphrase": passphrase},
    }
    ch = _Channel(USER_URL, sub, on_event, name="ws-user", needs_markets=True)
    ch.start()
    return ch


def market_channel(token_ids: list[str],
                   on_event: Callable[[dict], None]) -> _Channel:
    """Подписка на стакан по токенам. Авторизация не нужна."""
    sub = {"type": "market", "assets_ids": token_ids}
    ch = _Channel(MARKET_URL, sub, on_event, name="ws-market")
    ch.start()
    return ch


def order_ids_in(event: dict) -> set[str]:
    """Идентификаторы ордеров, упомянутые в событии сделки.

    Агрессивное исполнение кладёт наш ордер в `taker_order_id`, пассивное — в
    `maker_orders[].order_id`. Возвращаем всех кандидатов; чьи они, решает
    вызывающий по своему реестру.
    """
    if (event.get("event_type") or event.get("type")) != "trade":
        return set()
    if str(event.get("status") or "").upper() not in ("MATCHED", "CONFIRMED"):
        return set()
    ids = {event.get("taker_order_id"), event.get("order_id")}
    for mo in event.get("maker_orders") or []:
        if isinstance(mo, dict):
            ids.add(mo.get("order_id") or mo.get("orderID"))
    return {str(i) for i in ids if i}
