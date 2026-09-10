"""Оркестрация: цикл одного часа и планировщик.

Расписание для рынка, начинающегося в H (всё в UTC):

    H − 120с … H − 25с  окно решения: первый тик внутри него ставит лимитку по
                        LIMIT_PRICE; если аск уже не выше — она исполнится сразу
    ... каждые 30 сек   опрос заполнения
    H + 11:55           лимитка истекает сама (GTD) — страховка на случай падения
    H + 12:00           первый тик после CROSS_AFTER_SECONDS: снять остаток лимитки,
                        дождаться подтверждения и добрать недостающее тейкером (FAK),
                        не дороже TAKER_MAX
    H + 1ч + 60 сек     резолв, P&L, запись, уведомление

Устойчивость к падениям держится на четырёх вещах: лимитка истекает сама до
момента добора, на каждый час в БД ровно одна строка (повторный запуск не удвоит
ставку), отметка о доборе пишется до отправки (второго добора не будет), а при
старте незакрытые рынки дорезолвливаются.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Callable

import pandas as pd
from sqlalchemy import or_

import config
from app import features, markets, model, notify, pricing, risk, storage, trader, ws
from app.log import get

log = get("runner")

#: Как часто перепроверять правило резолва рынка (не перед каждым ордером).
VALIDATE_EVERY = dt.timedelta(hours=24)

#: Сколько секунд до начала часа нужно оставить, чтобы успеть подписать и отправить
#: ордер. Это не стратегия, а технический запас, поэтому живёт здесь, а не в конфиге.
#: Признаки от момента решения не зависят — они жёстко обрезаны на CUTOFF_MINUTES,
#: так что решение в любой точке окна даёт один и тот же вектор.
MIN_LEAD_SECONDS = 25


def _now() -> pd.Timestamp:
    """Текущее время UTC. Одна точка на весь модуль — реплей подставляет свои часы."""
    return pd.Timestamp.now(tz="UTC")


def cross_at(hour_start) -> pd.Timestamp:
    """Конец пассивной фазы: лимитка снимается, остаток добирается тейкером."""
    return pd.Timestamp(hour_start) + pd.Timedelta(seconds=config.CROSS_AFTER_SECONDS)


def next_hour(now: dt.datetime | None = None) -> pd.Timestamp:
    """Ближайший час, по которому мы ещё успеваем принять решение.

    Порог здесь — MIN_LEAD_SECONDS, а не EARLY_SECONDS: «поздно» значит «не успеем
    отправить ордер», а не «не попали в начало окна». Если сравнивать с EARLY_SECONDS,
    условие входа в `tick` становится недостижимым — окно схлопывается в точку.
    """
    now = pd.Timestamp(now) if now is not None else _now()
    h = now.ceil("h")
    if h - now < pd.Timedelta(seconds=MIN_LEAD_SECONDS):
        h += pd.Timedelta(hours=1)          # на этот час опоздали
    return h


class Bot:
    def __init__(self, stopping: "Callable[[], bool] | None" = None) -> None:
        storage.init()
        self.trader = trader.build()
        self._last_validation: dt.datetime | None = None
        #: Пауза по команде: начатое доводим до резолва, новых ставок не делаем.
        self.paused = False
        #: Признак «пора останавливаться» — чтобы прервать ожидание между попытками.
        self._stopping = stopping or (lambda: False)
        #: Часы, по которым заявка уже уходила на биржу. Нужен, чтобы повтор после
        #: сбоя не поставил второй ордер на тот же рынок.
        self._dispatched: set[dt.datetime] = set()
        # Канал своих ордеров: событие о сделке приходит сразу, а не через полминуты.
        # Креды берём у торгового клиента — он выводит их из приватного ключа сам.
        # Если сокет упадёт, страховкой остаётся опрос по REST.
        self.user_ws = ws.user_channel(self._on_user_event,
                                       creds=getattr(self.trader, "creds", None))

    # ------------------------------------------------------------------
    # WebSocket: исполнение в реальном времени
    # ------------------------------------------------------------------

    def _on_user_event(self, e: dict) -> None:
        """Событие своей сделки — повод немедленно перечитать ордер.

        Объём из события здесь не берётся: при пассивном исполнении наша доля лежит
        в `maker_orders[].matched_amount`, а верхнеуровневый `size` — объём всей
        сделки. Вместо разбора форм спрашиваем биржу о самом ордере; это тот же
        путь, что и в опросе по расписанию, просто на полминуты раньше.
        """
        ids = ws.order_ids_in(e)
        if not ids:
            return
        with storage.session() as s:
            ours = [storage.as_utc(m.hour_start)
                    for m in s.query(storage.Market)
                    .filter(or_(storage.Market.order_id.in_(ids),
                                storage.Market.cross_order_id.in_(ids)))
                    .filter(storage.Market.status.in_([storage.ORDERED,
                                                       storage.FILLED])).all()]
        for hour_start in ours:
            log.debug("ws: trade event for our order at {} — syncing via REST",
                     hour_start)
            self._sync_fill(hour_start)

    def _revalidate(self) -> None:
        """Периодическая проверка правила резолва — раз в сутки, не перед ордером."""
        now = _now().to_pydatetime()
        if self._last_validation and now - self._last_validation < VALIDATE_EVERY:
            return
        try:
            markets.ensure_valid(next_hour(now), force=True)
            self._last_validation = now
        except Exception as e:                                # noqa: BLE001
            msg = f"resolution rule changed or market not found: {e}"
            log.error(msg)
            storage.log_event("ERROR", "validation", msg)
            notify.error("Проверка рынка", msg)

    def close(self) -> None:
        if self.user_ws:
            self.user_ws.stop()

    def _account(self) -> tuple[float, float] | None:
        """(cash, portfolio) для уведомлений — только в бою, в бумаге денег нет."""
        return self.trader.account_value() if self.trader.live else None

    # ------------------------------------------------------------------
    # Решение и ордер
    # ------------------------------------------------------------------

    def handle_market(self, hour_start: pd.Timestamp) -> None:
        """Принять решение по одному рынку и выставить ордер. Идемпотентно."""
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            if m.status != storage.NEW:
                # В окно попадает несколько тиков подряд — это норма, не шумим в лог.
                log.debug("{} already handled ({}) — skipping", hour_start, m.status)
                return

        try:
            self._decide_and_order(hour_start)
        except Exception as e:                                # noqa: BLE001
            self._handle_failure(hour_start, e)

    def _handle_failure(self, hour_start: pd.Timestamp, exc: Exception) -> None:
        """Разобрать сбой при выставлении: пробовать ещё или потерять час.

        Окно решения длиной EARLY_SECONDS − MIN_LEAD_SECONDS вмещает несколько
        тиков, поэтому единичный сбой сети не должен стоить целого часа: строка
        остаётся в NEW, и следующий тик зайдёт сюда снова. Признаки от момента
        решения не зависят (обрезка на CUTOFF_MINUTES), так что повтор внутри
        окна — та же самая ставка, а не новая. Окончательным сбой становится,
        когда окно закрылось: это доделывает `_close_stale`.

        Исключение — час, по которому заявка уже уходила на биржу: повтор мог бы
        поставить второй ордер. Неважно, чем кончилась отправка: таймаут не
        означает, что биржа заявку не приняла.
        """
        left = (pd.Timestamp(hour_start) - _now()).total_seconds()
        if left > MIN_LEAD_SECONDS and hour_start.to_pydatetime() not in self._dispatched:
            log.warning("{}: {} — {:.0f}s left in window, retrying next tick",
                        hour_start, exc, left)
            with storage.session() as s:
                m = storage.get_or_create(s, hour_start.to_pydatetime())
                m.reason = str(exc)[:2000]
            return

        log.exception("error on market {}", hour_start)
        self._mark_error(hour_start, str(exc))
        notify.error("Ошибка при выставлении ордера",
                     f"{hour_start:%d.%m %H:%M}Z: {exc}")

    def _close_stale(self, now: pd.Timestamp) -> None:
        """Зафиксировать часы, по которым решение так и не состоялось.

        Сбой внутри окна оставляет строку в NEW ради повтора. Когда окно
        закрылось, повторять уже негде — час становится ошибкой, чтобы не висел
        в журнале «новым» и чтобы о потере пришло уведомление.

        Граница — окно РЕШЕНИЯ (до H − MIN_LEAD_SECONDS), а не пассивной фазы:
        момент решения H−2 новая схема исполнения не меняет.
        """
        cutoff = (now + pd.Timedelta(seconds=MIN_LEAD_SECONDS)).to_pydatetime()
        with storage.session() as s:
            rows = [(storage.as_utc(m.hour_start), m.reason)
                    for m in s.query(storage.Market)
                    .filter(storage.Market.status == storage.NEW)
                    .filter(storage.Market.hour_start < cutoff).all()]

        for hour_start, reason in rows:
            hs = pd.Timestamp(hour_start)
            reason = reason or "решение не состоялось до конца окна"
            self._mark_error(hs, reason)
            log.error("{}: hour lost — {}", hs, reason)
            notify.error("Час пропущен из-за сбоя", f"{hs:%d.%m %H:%M}Z: {reason}")

    def _decide_and_order(self, hour_start: pd.Timestamp) -> None:
        if self.paused:
            log.info("paused — not betting on {}", hour_start)
            self._mark_skipped(hour_start, "пауза по команде из Telegram")
            return

        verdict = risk.check()
        if not verdict.allowed:
            log.warning("bet skipped: {}", verdict.reason)
            self._mark_skipped(hour_start, verdict.reason)
            return

        info = markets.find(hour_start)
        if not info.accepting_orders:
            self._mark_skipped(hour_start, "рынок не принимает ордера")
            return

        minutes = features.for_market(hour_start)
        proba = model.predict(minutes, hour_start)["p_up"]
        # Стакан по стороне ставки: по нему отмечается, исполнится ли лимитка
        # сразу (аск уже не выше нашей цены) — это журнал, а не условие ставки.
        side = "Up" if proba >= 0.5 else "Down"
        side_book = markets.book(info.token(side))
        with storage.session() as s:
            shares = storage.current_shares(s)
        d = pricing.decide(proba, tick=info.tick_size, best_ask=side_book.get("ask"),
                           shares=shares)
        log.info("{} | {} | {}", hour_start, info.slug, pricing.describe(d))

        book = markets.snapshot(info)
        now = _now()
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            m.slug, m.condition_id = info.slug, info.condition_id
            m.token_up, m.token_down = info.token_up, info.token_down
            m.decided_at = now.to_pydatetime()
            m.proba_up, m.ev_usd, m.reason = d.proba_up, d.ev_usd, d.reason
            storage.add_quote(s, hour_start.to_pydatetime(), "decision", book,
                              at=now.to_pydatetime())

            if not d.bet:
                m.status = storage.SKIPPED
                notify.skipped(hour_start, d.reason)
                return
            m.side, m.price, m.shares = d.side, d.price, d.shares
            m.place_ts = now.to_pydatetime()

        # Быстрая серия попыток до начала часа: это самый ценный момент входа,
        # ради него стоит проверять стакан каждые несколько секунд.
        res, d, tries = self._place_until_resting(info, d, hour_start)
        self._record_placement(hour_start, info, d, res, tries,
                               bid=side_book.get("bid"), ask=side_book.get("ask"))

    def _record_placement(self, hour_start, info, d, res, tries,
                          bid=None, ask=None) -> None:
        """Записать итог попытки постановки и уведомить."""
        with storage.session() as s:
            m = storage.get_market(s, hour_start.to_pydatetime())
            m.price, m.ev_usd, m.reason = d.price, d.ev_usd, d.reason
            m.side, m.shares = d.side, d.shares
            m.place_attempts = tries
            if res.post_only_rejected:
                # Не ошибка и не отказ от часа: встречка по нашей цене ещё стоит.
                # Ждём и пробуем снова на каждом тике до конца пассивной фазы,
                # дальше час перейдёт к добору тейкером.
                m.status = storage.PENDING
                m.reason = (f"жду пассивной постановки, попыток {tries}: "
                            f"встречка по {d.price:.2f} держится")
                return
            if not res.ok:
                m.status, m.reason = storage.ERROR, res.error or "ордер отклонён"
                notify.error("Ордер отклонён", f"{hour_start:%d.%m %H:%M}Z: {m.reason}")
                return
            m.status = storage.ORDERED
            m.order_id = res.order_id
            m.ordered_at = _now().to_pydatetime()

        # Канал своих сделок слушает конкретные рынки: пока список пуст, событий нет.
        if self.user_ws:
            self.user_ws.update_markets(self._live_conditions())
        notify.order_placed(hour_start, d.side, d.price, d.shares, d.proba_up,
                            ev=d.ev_usd, cross=d.cross, bid=bid, ask=ask, tries=tries,
                            account=self._account())
        # Маркетабельная лимитка исполняется в момент постановки — не ждём тика.
        self._sync_fill(pd.Timestamp(hour_start).to_pydatetime())

    def retry_pending(self) -> None:
        """Продолжать ставить ордера, которые ещё не легли в книгу.

        Бывает только при POST_ONLY=True (сейчас выключено): биржа отклоняет
        заявку, которая исполнилась бы сразу. Потолка по числу попыток нет:
        пробуем на каждом тике до конца пассивной фазы — hour_start +
        CROSS_AFTER_SECONDS. Дальше строка остаётся в PENDING, и её подбирает
        `cross_remaining`: добирает весь размер тейкером или отказывается.

        Почему не крутить это в одном длинном цикле внутри часа: он заблокировал бы
        основной поток, и бот на этот час перестал бы проверять исполнение, резолвить
        прошлый рынок и решать по следующему.
        """
        now = _now()
        with storage.session() as s:
            rows = [(storage.as_utc(m.hour_start), m.side, m.price, m.shares,
                     m.proba_up, m.place_attempts or 0)
                    for m in s.query(storage.Market)
                    .filter(storage.Market.status == storage.PENDING).all()]

        for hour_start, side, price, shares, proba, done in rows:
            hs = pd.Timestamp(hour_start)
            if now >= cross_at(hs):
                continue                    # пассивная фаза кончилась — это дело добора

            # Короткая серия внутри тика: плотнее ловим момент, когда встречка
            # уходит, но не превращаем тик в бесконечное ожидание.
            until = time.monotonic() + config.RETRY_BURST_SECONDS
            while True:
                try:
                    info = markets.find(hs, with_book=False)
                    book = markets.book(info.token(side))
                    # shares — уже зафиксированный при первом решении размер этого
                    # рынка (из его же строки в БД), а не текущий /shares: если
                    # ставку поменяли посреди часа, начатая ставка размер не меняет.
                    d = pricing.decide(proba, tick=info.tick_size,
                                       best_ask=book.get("ask"), shares=shares)
                    if not d.bet:
                        break
                    res = self.trader.place(info, d.side, d.price, d.shares,
                                            post_only=config.POST_ONLY)
                except Exception as e:                        # noqa: BLE001
                    log.warning("retry placement {} failed: {}", hs, e)
                    break

                done += 1
                if not res.post_only_rejected:
                    if res.ok:
                        log.success("{}: order landed on attempt {}", hs, done)
                    self._record_placement(hs, info, d, res, done,
                                           bid=book.get("bid"), ask=book.get("ask"))
                    break
                if (time.monotonic() >= until or self._stopping()
                        or _now() >= cross_at(hs)):
                    with storage.session() as s:
                        storage.get_market(s, hour_start).place_attempts = done
                    break
                time.sleep(config.RETRY_SECONDS)

    def _place_until_resting(self, info, d, hour_start
                             ) -> tuple[trader.OrderResult, "pricing.Decision", int]:
        """Поставить лимитку; при отказе post-only — повторять до конца пассивной фазы.

        Дедлайн — hour_start + CROSS_AFTER_SECONDS: до него заявка может стоять
        в книге, после него остаток добирается тейкером (`cross_remaining`).

        При POST_ONLY=False (сейчас так) отказа post-only не бывает, и серия
        кончается на первой попытке: если аск уже не выше нашей цены, заявка
        исполнится сразу тейкером — это нормальный исход. Ждать пассивной
        постановки невыгодно: по замеру на настоящем стакане ожидание стоит
        дороже пересечения — пассивную заявку наливают в основном, когда цена
        идёт против позиции, а комиссия тейкера в замеренный период фактически
        не списывалась. Отказ от post-only сам по себе дал +0.0081 на шейр в час.

        Если POST_ONLY включат обратно, отказ означает «по нашей цене есть
        встречка»: стакан перечитывается и попытка повторяется. Одна серия не
        держит тик дольше, чем до начала часа (или RETRY_BURST_SECONDS, если час
        уже идёт) — длинное ожидание заблокировало бы весь остальной цикл; дальше
        строка уходит в PENDING, и `retry_pending` продолжает попытки на тиках.
        """
        deadline = cross_at(hour_start)
        stop_series = min(deadline, max(
            pd.Timestamp(hour_start),
            _now() + pd.Timedelta(seconds=config.RETRY_BURST_SECONDS)))
        tries = 0
        while True:
            tries += 1
            # Отмечаем до отправки, а не после: если вызов оборвётся таймаутом,
            # ордер всё равно мог дойти, и повторять этот час уже нельзя.
            self._dispatched.add(pd.Timestamp(hour_start).to_pydatetime())
            res = self.trader.place(info, d.side, d.price, d.shares,
                                    post_only=config.POST_ONLY)
            if res.ok or not res.post_only_rejected:
                if tries > 1:
                    log.success("{}: landed passively on attempt {}", hour_start, tries)
                return res, d, tries

            left = (stop_series - _now()).total_seconds()
            if left <= config.RETRY_SECONDS or self._stopping():
                log.warning("{}: not resting after {} fast attempts, will keep "
                            "trying on ticks until {:%H:%M:%S}", hour_start, tries,
                            deadline)
                return res, d, tries
            time.sleep(config.RETRY_SECONDS)

            # Стакан мог измениться — решение пересчитываем, а не повторяем старое.
            try:
                ask = markets.book(info.token(d.side)).get("ask")
            except Exception as e:                            # noqa: BLE001
                log.warning("book read failed on attempt {}: {}", tries, e)
                continue
            # shares берём из уже принятого решения d, а не заново из /shares —
            # серия попыток внутри одного захода на один и тот же рынок должна
            # оставаться с тем размером, с которым начиналась.
            fresh = pricing.decide(d.proba_up, tick=info.tick_size, best_ask=ask,
                                   shares=d.shares)
            if fresh.bet:
                d = fresh

    def _live_conditions(self) -> list[str]:
        """conditionId рынков, по которым у нас есть незакрытая заявка."""
        with storage.session() as s:
            return [m.condition_id for m in storage.open_positions(s)
                    if m.condition_id]

    # ------------------------------------------------------------------
    # Заполнение
    # ------------------------------------------------------------------

    def _sync_fill(self, hour_start) -> None:
        """Спросить биржу о заполнении ордеров часа и записать результат.

        Единственное место, где заполнение попадает в БД: и опрос по расписанию,
        и событие из WebSocket, и добор приходят сюда. Кусков два вида — лимитки
        (`order_id`) и добора (`cross_order_id`); итог записывается по обоим сразу.
        Уведомление шлётся только на прирост, поэтому двойной вызов безопасен.
        """
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            if m is None or m.status not in (storage.ORDERED, storage.FILLED):
                return
            side, price = m.side, m.price
            order_id, cross_id = m.order_id, m.cross_order_id
            slug, was = m.slug, m.filled_shares or 0.0

        try:
            info = markets.find(pd.Timestamp(hour_start), with_book=False)
            limit_legs = self.trader.legs(info, order_id)
            cross_legs = self.trader.legs(info, cross_id)
        except Exception as e:                                # noqa: BLE001
            log.warning("fill poll for {} failed: {}", slug, e)
            return

        legs = limit_legs + cross_legs
        filled = sum(leg.shares for leg in legs)
        if filled <= was + 1e-9:
            return
        # Роль позиции — по тому, какой кусок дал больше шейров.
        by_role = {
            storage.MAKER: sum(g.shares for g in limit_legs if g.role == "MAKER"),
            storage.IMMEDIATE_TAKER: sum(g.shares for g in limit_legs if g.role != "MAKER"),
            storage.CROSS: sum(g.shares for g in cross_legs),
        }
        role = max(by_role, key=by_role.get)
        got = sum(g.price * g.shares for g in legs) / filled
        # По формуле — консервативная оценка, из неё считается P&L; фактическое
        # списание — отдельно, и оно неизвестно, если хоть один кусок без данных.
        fee = sum(g.shares * pricing.fee_per_share(g.price)
                  for g in legs if g.role == "TAKER")
        charged = (None if any(g.fee_charged is None for g in legs)
                   else sum(g.fee_charged for g in legs))
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            m.status = storage.FILLED
            m.filled_shares = filled
            m.fill_price = got
            m.fill_role = role
            m.fee_usd = fee
            m.fee_charged_usd = charged
            m.filled_at = _now().to_pydatetime()
        log.success("filled {} {} × {:g} at {:.4f} [{}], fee {:.4f}$ by formula, "
                    "charged {}", hour_start, side, filled, got, role, fee,
                    "?" if charged is None else f"{charged:.4f}$")
        notify.filled(hour_start, side, got, filled,
                      role=role, fee=fee, account=self._account())

    def poll_fills(self) -> None:
        """Проверить, не исполнились ли выставленные ордера.

        Опрашиваются и частично исполненные: лимитку могут доналить, а добор
        дописывает свои куски. Полностью набранная позиция больше не опрашивается.
        """
        now = _now()
        with storage.session() as s:
            rows = [(storage.as_utc(m.hour_start), m.order_id, m.status,
                     m.filled_shares or 0.0, m.shares or 0.0)
                    for m in storage.open_positions(s)]

        for hour_start, order_id, status, filled, shares in rows:
            end = pd.Timestamp(hour_start) + pd.Timedelta(hours=1)
            if now >= end:
                if status == storage.ORDERED:
                    self._close_unfilled(hour_start, order_id)
                continue
            if status == storage.ORDERED or filled < shares - 1e-9:
                self._sync_fill(hour_start)

    def _close_unfilled(self, hour_start, order_id: str | None) -> None:
        """Страховка конца часа: в норме до сюда доходит только то, что пропустил
        добор (например, бот лежал весь час)."""
        if order_id:
            self.trader.cancel(order_id)
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            if m and m.status == storage.ORDERED:
                m.status = storage.UNFILLED
                m.reason = "не исполнился до конца часа"
        log.warning("{}: order not filled", hour_start)

    # ------------------------------------------------------------------
    # Добор тейкером
    # ------------------------------------------------------------------

    def cross_remaining(self) -> None:
        """Добор тейкером на первом тике после hour_start + CROSS_AFTER_SECONDS.

        Касается часов, где ставку решили сделать, но набрано меньше размера:
        лимитка лежит (ORDERED), налита частично (FILLED) или так и не легла
        (PENDING). По каждому — ровно одна попытка, дальше час живёт до резолва
        с тем, что набралось. Подробности — `_cross_one`.
        """
        now = _now()
        with storage.session() as s:
            rows = [storage.as_utc(m.hour_start)
                    for m in s.query(storage.Market)
                    .filter(storage.Market.status.in_([storage.ORDERED, storage.FILLED,
                                                       storage.PENDING]))
                    .filter(storage.Market.cross_ts.is_(None)).all()]

        for hour_start in rows:
            if now < cross_at(hour_start):
                continue
            try:
                self._cross_one(hour_start, now)
            except Exception as e:                            # noqa: BLE001
                # Строка без cross_ts — следующий тик попробует снова, пока не
                # выйдет CROSS_LATEST_SECONDS.
                log.exception("cross for {} failed: {}", hour_start, e)

    def _cross_one(self, hour_start, now: pd.Timestamp) -> None:
        """Снять лимитку, дождаться подтверждения и докупить недостающее.

        Порядок важен: сначала последнее заполнение, потом отмена с
        подтверждением, потом заполнение ещё раз (куски могли прийти, пока
        снимали), и только после этого остаток. Без подтверждения отмены не
        добираем вовсе — иначе лимитку могли бы доналить поверх добора и купить
        больше размера ставки. Размер — из строки рынка, а не из /shares.

        Отказываемся добирать (и пишем почему в cross_note): набрано полностью;
        позже CROSS_LATEST_SECONDS; пауза или TRADING_ENABLED=False; остаток
        меньше минимального ордера биржи; стакан пуст; лучший аск выше TAKER_MAX.
        Матожидание при доборе не перепроверяется — по замеру важен только потолок.
        """
        hs = pd.Timestamp(hour_start)
        late = now >= hs + pd.Timedelta(seconds=config.CROSS_LATEST_SECONDS)

        self._sync_fill(hour_start)
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            side, shares, order_id = m.side, m.shares or 0.0, m.order_id
            status, filled = m.status, m.filled_shares or 0.0

        matched = None
        if order_id and filled < shares - 1e-9:
            ok, matched = self.trader.cancel_confirmed(order_id)
            if not ok:
                if late:
                    self._finish_cross(hour_start, now, "отмена лимитки не "
                                       "подтвердилась до конца окна добора — не добираем")
                else:
                    log.warning("{}: cancel not confirmed — cross on the next tick", hs)
                return
            self._sync_fill(hour_start)
            with storage.session() as s:
                filled = storage.get_market(s, hour_start).filled_shares or 0.0

        done = max(filled, matched or 0.0)
        rest = round(shares - done, 6)
        if rest <= 1e-9:
            self._finish_cross(hour_start, now, f"добор не нужен: набрано {done:g} "
                                                f"из {shares:g}")
            return
        if late:
            self._finish_cross(hour_start, now, "опоздали: позже "
                               f"{config.CROSS_LATEST_SECONDS // 60}-й минуты не добираем")
            return
        if self.paused or not config.TRADING_ENABLED:
            why = "пауза" if self.paused else "торговля выключена в конфиге"
            self._finish_cross(hour_start, now, f"{why} — лимитка снята, не добираем")
            return

        info = markets.find(hs, with_book=False)
        if rest < info.min_size - 1e-9:
            self._finish_cross(hour_start, now, f"остаток {rest:g} меньше минимума "
                                                f"биржи {info.min_size:g} — не добираем")
            return
        book = markets.snapshot(info)
        with storage.session() as s:
            storage.add_quote(s, hs.to_pydatetime(), "cross", book, at=now.to_pydatetime())
        ask = book["best_ask_up" if side == "Up" else "best_ask_down"]
        if ask is None:
            self._finish_cross(hour_start, now, "стакан пуст — не добираем")
            return
        if ask > config.TAKER_MAX + 1e-9:
            self._finish_cross(hour_start, now, f"аск {ask:.2f} выше потолка "
                                                f"{config.TAKER_MAX:.2f} — не добираем")
            return

        limit = min(round(ask + config.TAKER_SLIPPAGE_TICKS * info.tick_size, 4),
                    config.TAKER_MAX)
        note = f"добор {rest:g} по ≤ {limit:.2f} (аск {ask:.2f})"
        # Отметка ДО отправки: таймаут не означает, что заявка не дошла, а второй
        # добор по тому же часу купил бы сверх размера.
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            m.cross_ts, m.cross_note = now.to_pydatetime(), note
            if m.status == storage.PENDING:
                m.status = storage.ORDERED        # теперь по часу есть заявка на бирже
        res = self.trader.place(info, side, limit, rest, post_only=False,
                                order_type="FAK")
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            if res.ok:
                m.cross_order_id = res.order_id
            else:
                m.cross_note = f"{note} — не исполнился: {res.error}"[:2000]
        log.info("{}: {}{}", hs, note, "" if res.ok else f" — failed: {res.error}")
        if res.ok:
            if self.user_ws and status == storage.PENDING:
                self.user_ws.update_markets(self._live_conditions())
            self._sync_fill(hour_start)
        self._settle_after_cross(hour_start)

    def _finish_cross(self, hour_start, now: pd.Timestamp, note: str) -> None:
        """Закрыть добор без заявки: записать причину и итог часа."""
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            m.cross_ts, m.cross_note = now.to_pydatetime(), note
        log.info("{}: {}", pd.Timestamp(hour_start), note)
        self._settle_after_cross(hour_start)

    def _settle_after_cross(self, hour_start) -> None:
        """После добора час либо в игре с тем, что набралось, либо без ставки."""
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            if not m.filled_shares and m.status in (storage.ORDERED, storage.PENDING):
                m.status = storage.UNFILLED
                m.reason = f"не исполнился: {m.cross_note}"[:2000]
                log.warning("{}: nothing filled — {}", pd.Timestamp(hour_start),
                            m.cross_note)

    # ------------------------------------------------------------------
    # Резолв
    # ------------------------------------------------------------------

    def resolve_pending(self) -> None:
        """Дописать исходы для завершившихся рынков."""
        now = _now().to_pydatetime()
        cutoff = now - dt.timedelta(hours=1, seconds=60)
        with storage.session() as s:
            rows = [(storage.as_utc(m.hour_start), m.side,
                     m.fill_price if m.fill_price else m.price,
                     m.filled_shares, m.fee_usd or 0.0)
                    for m in storage.pending_resolution(s, cutoff)]

        for hour_start, side, price, filled, fee in rows:
            hs = pd.Timestamp(hour_start)
            outcome = markets.outcome_from_binance(hs)
            source = "binance"
            if outcome is None:
                outcome = markets.outcome_of(hs)
                source = "polymarket"
            if outcome is None:
                continue

            # Сверка: расхождение источников означает, что либо данные кривые,
            # либо правило рынка изменилось. И то, и другое надо заметить сразу.
            pm = markets.outcome_of(hs)
            if pm and pm != outcome:
                msg = (f"{hs:%d.%m %H:%M}Z: Binance даёт {outcome}, "
                       f"Polymarket {pm} — проверьте правило резолва")
                log.error("{:%d.%m %H:%M}Z: Binance says {}, Polymarket says {} "
                         "— check the resolution rule", hs, outcome, pm)
                notify.error("Расхождение резолва", msg)
                storage.log_event("ERROR", "resolve_mismatch", msg)

            won = (side == outcome) if side else None
            pnl = 0.0
            if filled:
                # По реальной цене исполнения и за вычетом уплаченной комиссии —
                # иначе результат систематически завышен.
                pnl = (filled * (1 - price) if won else -filled * price) - fee

            with storage.session() as s:
                m = storage.get_market(s, hour_start)
                m.outcome, m.won, m.pnl_usd = outcome, won, pnl
                m.status = storage.RESOLVED
                m.resolved_at = _now().to_pydatetime()
            # Win is a good outcome, loss is a normal priced-in one — not an error.
            # `won is None` means we had no bet on this hour — neither, just a fact.
            log_fn = log.info if won is None else (log.success if won else log.warning)
            log_fn("{} resolved {} (via {}), bet {} → {:+.2f}$",
                  hs, outcome, source, side or "-", pnl)
            if filled:
                # Итог считаем после записи, чтобы этот час уже был в нём учтён.
                with storage.session() as s:
                    total = storage.stats(s)
                notify.resolved(hour_start, side, outcome, bool(won), pnl,
                                price=price, shares=filled,
                                total=total, account=self._account())

    # ------------------------------------------------------------------

    def _mark_skipped(self, hour_start: pd.Timestamp, reason: str) -> None:
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            m.status, m.reason = storage.SKIPPED, reason
            m.decided_at = _now().to_pydatetime()

    def _mark_error(self, hour_start: pd.Timestamp, reason: str) -> None:
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            m.status, m.reason = storage.ERROR, reason[:2000]

    def tick(self) -> None:
        """Один проход: заполнения, добор, резолвы и, если пора, решение по часу.

        Окно решения — от EARLY_SECONDS до MIN_LEAD_SECONDS перед началом часа.
        Оно шире периода тика, поэтому попадём в него при любом сдвиге фаз;
        повторные попадания гасит идемпотентность `handle_market`. Добор
        проверяется на каждом тике и срабатывает на первом после
        CROSS_AFTER_SECONDS — с опозданием не больше одного тика.
        """
        self._revalidate()
        self.poll_fills()
        self.retry_pending()
        self.cross_remaining()
        self.resolve_pending()

        now = _now()
        h = next_hour(now)
        if (h - now) <= pd.Timedelta(seconds=config.EARLY_SECONDS):
            self.handle_market(h)
        # После решения, а не до: иначе час, только что оставленный на повтор,
        # закрылся бы этим же тиком.
        self._close_stale(now)
