"""Оркестрация: цикл одного часа и планировщик.

Расписание для рынка, начинающегося в H (всё в UTC):

    H − 120с … H − 25с  окно решения: первый тик внутри него ставит ордер
    ... каждые 30 сек   опрос заполнения, пока час не кончился
    H + 1ч + 60 сек     резолв, P&L, запись, уведомление

Устойчивость к падениям держится на трёх вещах: ордер выставляется с истечением
в конце часа (умрёт сам), на каждый час в БД ровно одна строка (повторный запуск
не удвоит ставку), а при старте незакрытые рынки дорезолвливаются.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Callable

import pandas as pd

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


def next_hour(now: dt.datetime | None = None) -> pd.Timestamp:
    """Ближайший час, по которому мы ещё успеваем принять решение.

    Порог здесь — MIN_LEAD_SECONDS, а не EARLY_SECONDS: «поздно» значит «не успеем
    отправить ордер», а не «не попали в начало окна». Если сравнивать с EARLY_SECONDS,
    условие входа в `tick` становится недостижимым — окно схлопывается в точку.
    """
    now = pd.Timestamp(now or dt.datetime.now(dt.timezone.utc))
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
                    .filter(storage.Market.order_id.in_(ids))
                    .filter(storage.Market.status.in_([storage.ORDERED,
                                                       storage.FILLED])).all()]
        for hour_start in ours:
            log.info("ws: сделка по нашему ордеру на %s — сверяю по REST", hour_start)
            self._sync_fill(hour_start)

    def _revalidate(self) -> None:
        """Периодическая проверка правила резолва — раз в сутки, не перед ордером."""
        now = dt.datetime.now(dt.timezone.utc)
        if self._last_validation and now - self._last_validation < VALIDATE_EVERY:
            return
        try:
            markets.ensure_valid(next_hour(now), force=True)
            self._last_validation = now
        except Exception as e:                                # noqa: BLE001
            msg = f"правило резолва рынка изменилось или рынок не найден: {e}"
            log.error(msg)
            storage.log_event("ERROR", "validation", msg)
            notify.error("Проверка рынка", msg)

    def close(self) -> None:
        if self.user_ws:
            self.user_ws.stop()

    # ------------------------------------------------------------------
    # Решение и ордер
    # ------------------------------------------------------------------

    def handle_market(self, hour_start: pd.Timestamp) -> None:
        """Принять решение по одному рынку и выставить ордер. Идемпотентно."""
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            if m.status != storage.NEW:
                # В окно попадает несколько тиков подряд — это норма, не шумим в лог.
                log.debug("%s уже обработан (%s) — пропуск", hour_start, m.status)
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
        left = (pd.Timestamp(hour_start) - pd.Timestamp.now(tz="UTC")).total_seconds()
        if left > MIN_LEAD_SECONDS and hour_start.to_pydatetime() not in self._dispatched:
            log.warning("%s: %s — в окне ещё %.0fс, повторю на следующем тике",
                        hour_start, exc, left)
            with storage.session() as s:
                m = storage.get_or_create(s, hour_start.to_pydatetime())
                m.reason = str(exc)[:2000]
            return

        log.exception("ошибка на рынке %s", hour_start)
        self._mark_error(hour_start, str(exc))
        notify.error("Ошибка при выставлении ордера",
                     f"{hour_start:%d.%m %H:%M}Z: {exc}")

    def _close_stale(self, now: pd.Timestamp) -> None:
        """Зафиксировать часы, по которым решение так и не состоялось.

        Сбой внутри окна оставляет строку в NEW ради повтора. Когда окно
        закрылось, повторять уже негде — час становится ошибкой, чтобы не висел
        в журнале «новым» и чтобы о потере пришло уведомление.
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
            log.error("%s: час потерян — %s", hs, reason)
            notify.error("Час пропущен из-за сбоя", f"{hs:%d.%m %H:%M}Z: {reason}")

    def _decide_and_order(self, hour_start: pd.Timestamp) -> None:
        if self.paused:
            log.info("на паузе — ставка на %s не делается", hour_start)
            self._mark_skipped(hour_start, "пауза по команде из Telegram")
            return

        verdict = risk.check()
        if not verdict.allowed:
            log.warning("ставка пропущена: %s", verdict.reason)
            self._mark_skipped(hour_start, verdict.reason)
            return

        info = markets.find(hour_start)
        if not info.accepting_orders:
            self._mark_skipped(hour_start, "рынок не принимает ордера")
            return

        minutes = features.for_market(hour_start)
        proba = model.predict(minutes, hour_start)["p_up"]
        # Стакан читаем по той стороне, на которую собираемся ставить: решение
        # «пассивно или тейкером» зависит от предложения именно по нашему токену.
        side = "Up" if proba >= 0.5 else "Down"
        side_book = markets.book(info.token(side))
        d = pricing.decide(proba, tick=info.tick_size, best_ask=side_book.get("ask"))
        log.info("%s | %s | %s", hour_start, info.slug, pricing.describe(d))

        book = markets.snapshot(info)
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            m.slug, m.condition_id = info.slug, info.condition_id
            m.token_up, m.token_down = info.token_up, info.token_down
            m.decided_at = dt.datetime.now(dt.timezone.utc)
            m.proba_up, m.ev_usd, m.reason = d.proba_up, d.ev_usd, d.reason
            storage.add_quote(s, hour_start.to_pydatetime(), "decision", book)

            if not d.bet:
                m.status = storage.SKIPPED
                notify.skipped(hour_start, d.reason)
                return
            m.side, m.price, m.shares = d.side, d.price, d.shares

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
                # Ждём и пробуем снова на каждом тике, пока час не кончится.
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
            m.ordered_at = dt.datetime.now(dt.timezone.utc)

        # Канал своих сделок слушает конкретные рынки: пока список пуст, событий нет.
        if self.user_ws:
            self.user_ws.update_markets(self._live_conditions())
        notify.order_placed(hour_start, d.side, d.price, d.shares, d.proba_up,
                            ev=d.ev_usd, cross=d.cross, bid=bid, ask=ask, tries=tries)
        # Тейкерный ордер исполняется в момент постановки — не ждём следующего тика.
        self._sync_fill(pd.Timestamp(hour_start).to_pydatetime())

    def retry_pending(self) -> None:
        """Продолжать ставить ордера, которые ещё не легли в книгу.

        Потолка по числу попыток нет: пробуем на каждом тике, пока рынок жив.
        Единственная граница — конец часа, и это не лимит попыток, а исчезновение
        самого рынка: после резолва ставить уже некуда.

        Почему не крутить это в одном длинном цикле внутри часа: он заблокировал бы
        основной поток, и бот на этот час перестал бы проверять исполнение, резолвить
        прошлый рынок и решать по следующему.
        """
        now = pd.Timestamp.now(tz="UTC")
        with storage.session() as s:
            rows = [(storage.as_utc(m.hour_start), m.side, m.price, m.shares,
                     m.proba_up, m.place_attempts or 0)
                    for m in s.query(storage.Market)
                    .filter(storage.Market.status == storage.PENDING).all()]

        for hour_start, side, price, shares, proba, done in rows:
            hs = pd.Timestamp(hour_start)
            if now >= hs + pd.Timedelta(hours=1):
                with storage.session() as s:
                    m = storage.get_market(s, hour_start)
                    m.status = storage.SKIPPED
                    m.reason = (f"за {done} попыток пассивно встать не удалось: "
                                f"встречка по {price:.2f} держалась весь час")
                    reason = m.reason
                log.info("%s: %s", hs, reason)
                notify.skipped(hs, reason)
                continue

            # Короткая серия внутри тика: плотнее ловим момент, когда встречка
            # уходит, но не превращаем тик в бесконечное ожидание.
            until = time.monotonic() + config.RETRY_BURST_SECONDS
            while True:
                try:
                    info = markets.find(hs, with_book=False)
                    book = markets.book(info.token(side))
                    d = pricing.decide(proba, tick=info.tick_size,
                                       best_ask=book.get("ask"))
                    if not d.bet:
                        break
                    res = self.trader.place(info, d.side, d.price, d.shares,
                                            post_only=config.POST_ONLY and not d.cross)
                except Exception as e:                        # noqa: BLE001
                    log.warning("повторная постановка %s не удалась: %s", hs, e)
                    break

                done += 1
                if not res.post_only_rejected:
                    if res.ok:
                        log.info("%s: ордер лёг с %d-й попытки", hs, done)
                    self._record_placement(hs, info, d, res, done,
                                           bid=book.get("bid"), ask=book.get("ask"))
                    break
                if time.monotonic() >= until or self._stopping():
                    with storage.session() as s:
                        storage.get_market(s, hour_start).place_attempts = done
                    break
                time.sleep(config.RETRY_SECONDS)

    def _place_until_resting(self, info, d, hour_start
                             ) -> tuple[trader.OrderResult, "pricing.Decision", int]:
        """Ставить пассивный ордер, пока он не ляжет в книгу или не начнётся час.

        Отказ post-only означает «по нашей цене есть встречка»: исполниться сейчас
        можно только тейкером, а это дороже (комиссия 1.75¢/шейр против ребейта
        0.35¢). Поэтому ждём и пробуем снова — стакан меняется. Перед каждой
        попыткой стакан перечитывается: если предложение упало до цены, которая
        окупает комиссию, `decide` сам переключит нас на тейкера.

        Здесь только быстрая серия до начала часа — самый ценный момент входа, ради
        него стоит опрашивать стакан каждые несколько секунд. Если и она не помогла,
        рынок не бросается: строка уходит в статус PENDING, и `retry_pending`
        продолжает попытки на каждом тике до конца часа. Держать длинное ожидание
        здесь нельзя — оно заблокировало бы весь остальной цикл.
        """
        deadline = pd.Timestamp(hour_start)
        tries = 0
        while True:
            tries += 1
            post_only = config.POST_ONLY and not d.cross
            # Отмечаем до отправки, а не после: если вызов оборвётся таймаутом,
            # ордер всё равно мог дойти, и повторять этот час уже нельзя.
            self._dispatched.add(pd.Timestamp(hour_start).to_pydatetime())
            res = self.trader.place(info, d.side, d.price, d.shares,
                                    post_only=post_only)
            if res.ok or not res.post_only_rejected:
                if tries > 1:
                    log.info("%s: встали пассивно с %d-й попытки", hour_start, tries)
                return res, d, tries

            left = (deadline - pd.Timestamp.now(tz="UTC")).total_seconds()
            if left <= config.RETRY_SECONDS or self._stopping():
                log.info("%s: за %d быстрых попыток не встали, "
                         "продолжу на тиках до конца часа", hour_start, tries)
                return res, d, tries
            time.sleep(config.RETRY_SECONDS)

            # Стакан мог измениться — решение пересчитываем, а не повторяем старое.
            try:
                ask = markets.book(info.token(d.side)).get("ask")
            except Exception as e:                            # noqa: BLE001
                log.warning("стакан не прочитался на попытке %d: %s", tries, e)
                continue
            fresh = pricing.decide(d.proba_up, tick=info.tick_size, best_ask=ask)
            if fresh.bet:
                if fresh.cross and not d.cross:
                    log.info("%s: ask %.2f окупает комиссию — идём тейкером",
                             hour_start, ask)
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
        """Спросить биржу о заполнении одного ордера и записать результат.

        Единственное место, где заполнение попадает в БД: и опрос по расписанию,
        и событие из WebSocket приходят сюда. Уведомление шлётся только на прирост,
        поэтому двойной вызов из обоих источников безопасен.
        """
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            if m is None or m.status not in (storage.ORDERED, storage.FILLED):
                return
            side, price, shares = m.side, m.price, m.shares
            order_id, slug, was = m.order_id, m.slug, m.filled_shares or 0.0

        try:
            info = markets.find(pd.Timestamp(hour_start), with_book=False)
            res = (self.trader.poll_fill(info, side, price, shares, order_id)
                   if self.trader.live
                   else self.trader.poll_fill(info, side, price, shares))
        except Exception as e:                                # noqa: BLE001
            log.warning("опрос заполнения %s не удался: %s", slug, e)
            return

        if not res.filled_shares or res.filled_shares <= was + 1e-9:
            return
        got = res.fill_price or price
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            m.status = storage.FILLED
            m.filled_shares = res.filled_shares
            m.fill_price = got
            m.fill_role = res.role
            m.fee_usd = res.fee_usd
            m.filled_at = dt.datetime.now(dt.timezone.utc)
        log.info("исполнение %s %s × %g по %.4f [%s], комиссия %.4f$",
                 hour_start, side, res.filled_shares, got, res.role or "?",
                 res.fee_usd)
        notify.filled(hour_start, side, got, res.filled_shares,
                      role=res.role, fee=res.fee_usd)

    def poll_fills(self) -> None:
        """Проверить, не исполнились ли выставленные ордера."""
        now = dt.datetime.now(dt.timezone.utc)
        with storage.session() as s:
            rows = [(storage.as_utc(m.hour_start), m.order_id)
                    for m in storage.open_positions(s)
                    if m.status == storage.ORDERED]

        for hour_start, order_id in rows:
            end = pd.Timestamp(hour_start) + pd.Timedelta(hours=1)
            if now >= end:
                self._close_unfilled(hour_start, order_id)
                continue
            self._sync_fill(hour_start)

    def _close_unfilled(self, hour_start, order_id: str | None) -> None:
        if order_id:
            self.trader.cancel(order_id)
        with storage.session() as s:
            m = storage.get_market(s, hour_start)
            if m and m.status == storage.ORDERED:
                m.status = storage.UNFILLED
                m.reason = "не исполнился до конца часа"
        log.info("%s: ордер не исполнился", hour_start)

    # ------------------------------------------------------------------
    # Резолв
    # ------------------------------------------------------------------

    def resolve_pending(self) -> None:
        """Дописать исходы для завершившихся рынков."""
        now = dt.datetime.now(dt.timezone.utc)
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
                log.error(msg)
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
                m.resolved_at = dt.datetime.now(dt.timezone.utc)
            log.info("%s резолв %s (%s), ставили %s → %+.2f$",
                     hs, outcome, source, side or "-", pnl)
            if filled:
                # Итоги считаем после записи, чтобы этот час уже был в них учтён.
                midnight = dt.datetime.combine(now.date(), dt.time.min,
                                               tzinfo=dt.timezone.utc)
                with storage.session() as s:
                    day = storage.pnl_since(s, midnight)
                    total = storage.stats(s)
                notify.resolved(hour_start, side, outcome, bool(won), pnl,
                                price=price, shares=filled,
                                day_pnl=day, total=total)

    # ------------------------------------------------------------------

    def _mark_skipped(self, hour_start: pd.Timestamp, reason: str) -> None:
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            m.status, m.reason = storage.SKIPPED, reason
            m.decided_at = dt.datetime.now(dt.timezone.utc)

    def _mark_error(self, hour_start: pd.Timestamp, reason: str) -> None:
        with storage.session() as s:
            m = storage.get_or_create(s, hour_start.to_pydatetime())
            m.status, m.reason = storage.ERROR, reason[:2000]

    def tick(self) -> None:
        """Один проход: заполнения, резолвы и, если пора, решение по следующему часу.

        Окно решения — от EARLY_SECONDS до MIN_LEAD_SECONDS перед началом часа.
        Оно шире периода тика, поэтому попадём в него при любом сдвиге фаз;
        повторные попадания гасит идемпотентность `handle_market`.
        """
        self._revalidate()
        self.poll_fills()
        self.retry_pending()
        self.resolve_pending()

        now = pd.Timestamp.now(tz="UTC")
        h = next_hour(now)
        if (h - now) <= pd.Timedelta(seconds=config.EARLY_SECONDS):
            self.handle_market(h)
        # После решения, а не до: иначе час, только что оставленный на повтор,
        # закрылся бы этим же тиком.
        self._close_stale(now)
