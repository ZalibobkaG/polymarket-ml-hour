"""Уведомления в Telegram.

Правило простое: уведомление никогда не должно ронять торговлю. Любая ошибка
отправки логируется и проглатывается — потерять сообщение не страшно, потерять
позицию из-за недоступного Telegram страшно.

Что показываем. Цену — всегда в центах и двумя видами: цена заявки и **эффективная**
цена с комиссией. Вторая совпадает с тем, что Polymarket пишет в строке «Покупка»,
так что сообщение можно сверить с приложением. Плюс порог безубытка: он численно
равен эффективной цене, и рядом с прогнозом модели сразу видно запас.

Объём. Часовых рынков 24 в сутки; шлём заполнения и ошибки со звуком, остальное —
тихо, чтобы поток не превращался в шум.
"""

from __future__ import annotations

import datetime as dt

import httpx

import config
from app import storage
from app.log import get

log = get("notify")

_API = "https://api.telegram.org/bot{token}/sendMessage"
_API_DOC = "https://api.telegram.org/bot{token}/sendDocument"
_API_PHOTO = "https://api.telegram.org/bot{token}/sendPhoto"
_TIMEOUT = 10.0
#: Отправка файла — это загрузка, а не короткий POST: даём ей больше времени.
_DOC_TIMEOUT = 60.0
#: Потолок Telegram на загрузку файла ботом — 50 МБ. Держим запас: смысла
#: упираться в границу нет, а внятный отказ лучше, чем ошибка от API.
_MAX_DOC_BYTES = 45 * 1024 * 1024
#: Ограничение Telegram на подпись к файлу.
_MAX_CAPTION = 1024


def enabled() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def _mask(text: str) -> str:
    """Убирает токен из текста ошибки.

    httpx кладёт полный URL в сообщение исключения, а в URL входит токен бота —
    без маскировки он попадает в лог-файл и живёт там месяц.
    """
    tok = config.TELEGRAM_BOT_TOKEN
    return text.replace(tok, "***") if tok else text


def send(text: str, silent: bool = False) -> bool:
    if not enabled():
        log.debug("telegram not configured, message not sent")
        return False
    if config.DRY_RUN:
        # В бумажном режиме это надо видеть, иначе сообщения не отличить от боевых.
        text = "🧪 <i>бумага</i>\n" + text
    try:
        r = httpx.post(
            _API.format(token=config.TELEGRAM_BOT_TOKEN),
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "HTML", "disable_notification": silent,
                  "disable_web_page_preview": True},
            timeout=_TIMEOUT)
        if r.status_code == 400:
            desc = ""
            try:
                desc = r.json().get("description", "")
            except Exception:                                # noqa: BLE001
                pass
            if "chat not found" in desc.lower():
                # Телеграм не разрешает боту писать первым — диалог начинает человек.
                log.error("telegram: chat {} not found. Open the bot in Telegram "
                          "and press Start, otherwise notifications won't work",
                          config.TELEGRAM_CHAT_ID)
            else:
                log.warning("telegram declined: {}", _mask(desc))
            return False
        r.raise_for_status()
        return True
    except Exception as e:                                   # noqa: BLE001
        log.warning("failed to send to telegram: {}", _mask(str(e)))
        return False


def send_document(data: bytes, filename: str, caption: str = "",
                  silent: bool = True) -> bool:
    """Отправить файл в чат. Возвращает True, если Telegram его принял.

    Как и `send`, ошибку не выбрасывает: не доехавший лог — неприятность,
    а не повод ронять поток команд.
    """
    if not enabled():
        log.debug("telegram not configured, file not sent")
        return False
    if not data:
        log.warning("nothing to send: file is empty")
        return False
    if len(data) > _MAX_DOC_BYTES:
        log.warning("file {} not sent: {:.1f} MB over the {:.0f} MB cap",
                    filename, len(data) / 1024 / 1024, _MAX_DOC_BYTES / 1024 / 1024)
        return False
    try:
        r = httpx.post(
            _API_DOC.format(token=config.TELEGRAM_BOT_TOKEN),
            data={"chat_id": config.TELEGRAM_CHAT_ID,
                  "disable_notification": str(silent).lower(),
                  **({"caption": caption[:_MAX_CAPTION],
                      "parse_mode": "HTML"} if caption else {})},
            files={"document": (filename, data, "text/plain; charset=utf-8")},
            timeout=_DOC_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:                                   # noqa: BLE001
        log.warning("failed to send file to telegram: {}", _mask(str(e)))
        return False


def send_photo(data: bytes, caption: str = "",
               buttons: list[list[tuple[str, str]]] | None = None,
               silent: bool = True) -> bool:
    """Отправить картинку (график) в чат, опционально с кнопками под ней.

    Кнопки в том же формате, что у `commands.Reply.buttons` — `(подпись, команда)`,
    команда пишется так же, как её набрали бы руками. Это позволяет, например,
    после графика оставить кнопки навигации по остальным разделам /stats.
    """
    if not enabled():
        log.debug("telegram not configured, photo not sent")
        return False
    if not data:
        log.warning("nothing to send: image is empty")
        return False
    payload: dict = {"chat_id": config.TELEGRAM_CHAT_ID,
                     "disable_notification": str(silent).lower()}
    if caption:
        payload["caption"] = caption[:_MAX_CAPTION]
        payload["parse_mode"] = "HTML"
    if buttons:
        import json
        payload["reply_markup"] = json.dumps({"inline_keyboard": [
            [{"text": label, "callback_data": cmd} for label, cmd in row]
            for row in buttons]})
    try:
        r = httpx.post(
            _API_PHOTO.format(token=config.TELEGRAM_BOT_TOKEN),
            data=payload,
            files={"photo": ("chart.png", data, "image/png")},
            timeout=_DOC_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:                                   # noqa: BLE001
        log.warning("failed to send photo to telegram: {}", _mask(str(e)))
        return False


# ----------------------------------------------------------------------
# Оформление
# ----------------------------------------------------------------------

def _hour(hour_start: dt.datetime) -> str:
    """Час рынка в UTC и в ET — Polymarket называет рынки по нью-йоркскому времени."""
    end = hour_start + dt.timedelta(hours=1)
    et = hour_start.astimezone(dt.timezone(dt.timedelta(hours=-4)))
    h = et.hour % 12 or 12
    return (f"{hour_start:%d.%m} {hour_start:%H:%M}–{end:%H:%M} UTC "
            f"({h}{'AM' if et.hour < 12 else 'PM'} ET)")


def _cents(price: float) -> str:
    return f"{price * 100:.1f}¢"


def _effective(price: float, shares: float, fee: float) -> float:
    """Цена шейра с комиссией — то же число, что Polymarket пишет в «Покупке»."""
    return (price * shares + fee) / shares if shares else price


def _quote(*lines: str) -> str:
    """Обернуть блок строк в <blockquote> — визуально отделяет группу от заголовка
    и от соседних групп. Пустые строки/группы отбрасываются, чтобы не плодить
    пустые рамки."""
    body = "\n".join(l for l in lines if l)
    return f"<blockquote>{body}</blockquote>" if body else ""


def _account_line(account: tuple[float, float] | None) -> str:
    """Portfolio/Cash одной строкой — как в приложении Polymarket. `account` —
    `(cash, portfolio)` от `ClobTrader.account_value()`; None — не показываем
    (бумажный режим или не удалось получить)."""
    if account is None:
        return ""
    cash, portfolio = account
    return f"💰 Портфель {portfolio:.2f}$ / Кэш {cash:.2f}$"


# ----------------------------------------------------------------------
# События
# ----------------------------------------------------------------------

def order_placed(hour_start: dt.datetime, side: str, price: float, shares: float,
                 proba_up: float, ev: float = 0.0, cross: bool = False,
                 bid: float | None = None, ask: float | None = None,
                 tries: int = 1, account: tuple[float, float] | None = None) -> None:
    q = proba_up if side == "Up" else 1 - proba_up
    how = ("аск уже не выше цены — исполнится сразу" if cross
           else f"лимиткой в книгу до {config.CROSS_AFTER_SECONDS // 60}-й минуты")
    meta = f"{_hour(hour_start)} · {how}"
    if tries > 1:
        meta += f" · встали с {tries}-й попытки"
    book_line = (f"Стакан по {side}: {_cents(bid) if bid else '—'} / "
                f"{_cents(ask) if ask else '—'}") if bid is not None or ask is not None else ""
    header = f"📌 <b>Ордер: {side} × {shares:g} по {_cents(price)}</b>\n{meta}"
    details = _quote(f"Модель даёт {q * 100:.1f}% на {side}, безубыток "
                     f"{price * 100:.1f}% → запас {(q - price) * 100:+.1f} п.п.",
                     f"EV {ev:+.3f}$, в риске {shares * price:.2f}$", book_line)
    sections = [header, details, _account_line(account)]
    send("\n\n".join(s for s in sections if s), silent=True)


_ROLE_NAMES = {"TAKER": "Тейкером", "MIXED": "Частями",
               "immediate_taker": "Сразу тейкером", "cross": "Добор тейкером"}


def filled(hour_start: dt.datetime, side: str, price: float, shares: float,
           role: str | None = None, fee: float = 0.0,
           account: tuple[float, float] | None = None) -> None:
    eff = _effective(price, shares, fee)
    if role in ("MAKER", "maker"):
        role_line = "Мейкером — комиссии нет"
    elif role:
        role_line = f"{_ROLE_NAMES.get(role, role)} — комиссия {fee:.3f}$ по формуле"
    else:
        role_line = ""
    meta = f"{_hour(hour_start)}" + (f" · {role_line}" if role_line else "")
    header = f"✅ <b>Исполнен: {side} × {shares:g} по {_cents(price)}</b>\n{meta}"
    details = _quote(f"Всего {price * shares + fee:.2f}$ = {_cents(eff)} за шейр "
                     f"(так и покажет Polymarket)",
                     f"Отбивается при {eff * 100:.1f}% угаданных")
    sections = [header, details, _account_line(account)]
    send("\n\n".join(s for s in sections if s))


def resolved(hour_start: dt.datetime, side: str, outcome: str, won: bool,
             pnl: float, price: float | None = None, shares: float = 0.0,
             total: dict | None = None,
             account: tuple[float, float] | None = None) -> None:
    head = "🟢 <b>Победа</b>" if won else "🔴 <b>Проигрыш</b>"
    meta = _hour(hour_start)
    if price is not None:
        meta += f" · ставили {side} × {shares:g} по {_cents(price)}"
    total_line = (f"Всего {total['P&L, $']:+.2f}$ "
                 f"({total['угадано']}/{total['ставок']}, "
                 f"{total['точность, %']:.1f}%)") if total else ""
    header = f"{head} · итог {outcome}\n{meta}"
    details = _quote(f"Результат <b>{pnl:+.2f}$</b>", total_line)
    sections = [header, details, _account_line(account)]
    send("\n\n".join(s for s in sections if s), silent=True)


# --------------------------------------------------------------------------
# Отладка схемы исполнения (config.DEBUG_NOTIFY) — временно, на прогон
# --------------------------------------------------------------------------

def _debug_send(text: str) -> None:
    if config.DEBUG_NOTIFY:
        send(text, silent=True)


def _px(p: float | None) -> str:
    return _cents(p) if p is not None else "—"


def decision_inputs(hour_start: dt.datetime, proba_up: float, side: str,
                    price: float, shares: float, book: dict,
                    last_minute: dt.datetime | None, last_close: float | None,
                    marketable: bool, open_positions: int,
                    pnl_day: float, pnl_month: float) -> None:
    """Что видел бот в момент решения — до постановки заявки."""
    q = proba_up if side == "Up" else 1 - proba_up
    data_line = (f"Последняя минутка {last_minute:%H:%M}Z, close {last_close:,.2f}"
                 if last_minute is not None and last_close is not None else "")
    ask_side = book.get("best_ask_up" if side == "Up" else "best_ask_down")
    how = (f"аск {_px(ask_side)} ≤ {_cents(price)} — исполнится сразу тейкером"
           if marketable else
           f"аск {_px(ask_side)} > {_cents(price)} — ляжет в книгу до "
           f"{config.CROSS_AFTER_SECONDS // 60}-й минуты")
    body = [
        f"Модель: p(Up) = {proba_up:.4f} → сторона <b>{side}</b> ({q * 100:.1f}%)",
        data_line,
        f"Стакан Up: {_px(book.get('best_bid_up'))} / {_px(book.get('best_ask_up'))}"
        f" · Down: {_px(book.get('best_bid_down'))} / {_px(book.get('best_ask_down'))}",
        f"Заявка: {side} × {shares:g} по {_cents(price)}, {how}",
        f"Риск: открытых позиций {open_positions}/{config.MAX_OPEN_POSITIONS}, "
        f"день {pnl_day:+.2f}$ (лимит −{config.DAILY_LOSS_LIMIT_USD:.0f}$), "
        f"месяц {pnl_month:+.2f}$ (лимит −{config.MONTHLY_LOSS_LIMIT_USD:.0f}$)",
    ]
    _debug_send(f"🧠 <b>Вводные решения</b>\n{_hour(hour_start)}\n\n" + _quote(*body))


_ROLE_SHORT = {"maker": "мейкером", "immediate_taker": "сразу тейкером",
               "cross": "добором", "none": "ничего"}


def cross_result(hour_start: dt.datetime, side: str, shares: float,
                 filled: float, fill_price: float | None, role: str | None,
                 note: str, status: str) -> None:
    """Что произошло на 12-й минуте: сколько налила лимитка, чем кончился добор."""
    if filled:
        got = (f"Набрано {filled:g} из {shares:g} по {_cents(fill_price or 0)} "
               f"в среднем, в основном {_ROLE_SHORT.get(role or '', role or '?')}")
    else:
        got = f"Набрано 0 из {shares:g}"
    outcome = ("час без ставки" if status == "unfilled"
               else f"в игре с {filled:g} шейрами {side}")
    _debug_send(f"🔁 <b>Добор, {config.CROSS_AFTER_SECONDS // 60}-я минута</b>\n"
                f"{_hour(hour_start)}\n\n" + _quote(f"Решение: {note}", got,
                                                     f"Итог: {outcome}"))


def execution_breakdown(hour_start: dt.datetime, side: str, shares: float,
                        filled: float, price: float | None, fill_price: float | None,
                        role: str | None, fee: float, fee_charged: float | None,
                        ordered_at: dt.datetime | None, filled_at: dt.datetime | None,
                        cross_ts: dt.datetime | None, cross_note: str | None) -> None:
    """Разбор исполнения часа — отдельным сообщением после исхода."""
    def _secs(a, b) -> str:
        return f"{(b - a).total_seconds():.0f} с" if a and b else "—"

    body = [
        f"Заявка {side} × {shares:g} по {_px(price)} → набрано {filled:g} "
        f"по {_px(fill_price)} в среднем",
        f"Роль: {_ROLE_SHORT.get(role or '', role or '—')}",
        f"Комиссия по формуле {fee:.4f}$, биржа списала "
        + ("?" if fee_charged is None else f"{fee_charged:.4f}$"),
        f"От постановки до исполнения: {_secs(ordered_at, filled_at)}"
        + (f" · добор на {_secs(ordered_at, cross_ts)} после постановки"
           if cross_ts else " · до добора не дошло"),
        f"Добор: {cross_note}" if cross_note else "",
    ]
    _debug_send(f"🧾 <b>Разбор исполнения</b>\n{_hour(hour_start)}\n\n" + _quote(*body))


def skipped(hour_start: dt.datetime, reason: str) -> None:
    send(f"⏭ <b>Пропуск часа</b>\n" + _quote(_hour(hour_start), reason), silent=True)


def error(kind: str, message: str) -> None:
    send(f"⚠️ <b>{kind}</b>\n{message}")


def daily_summary(stats: dict, day: dt.date) -> None:
    if not stats.get("ставок"):
        send(f"📊 <b>Итог за {day:%d.%m}</b>\n" + _quote("ставок не было"),
             silent=True)
        return
    fees = stats.get("комиссии, $") or 0.0
    maker, taker = stats.get("пассивных", 0), stats.get("тейкером", 0)
    body = [f"Ставок {stats['ставок']}, угадано {stats['угадано']} "
           f"({stats['точность, %']:.1f}%)",
           f"P&amp;L <b>{stats['P&L, $']:+.2f}$</b>"]
    if fees:
        body.append(f"Комиссий уплачено {fees:.3f}$")
    if maker or taker:
        body.append(f"Пассивно {maker}, тейкером {taker}")
    if stats.get("пропущено"):
        body.append(f"Пропущено часов: {stats['пропущено']}")
    body.append(f"На Up {stats['ставок на Up, %']:.0f}% ставок")
    send(f"📊 <b>Итог за {day:%d.%m}</b>\n" + _quote(*body), silent=True)


def startup(problems: list[str]) -> None:
    if problems:
        send("🚫 <b>Бот не стартовал</b>\n" + "\n".join(f"• {p}" for p in problems))
        return
    with storage.session() as s:
        shares = storage.current_shares(s)
    body = [f"Ставка {shares:g} шейров по {_cents(config.LIMIT_PRICE)} "
           f"— {shares * config.LIMIT_PRICE:.2f}$ в час",
           f"Решение за {config.EARLY_SECONDS // 60} мин до часа"]
    cross_min = config.CROSS_AFTER_SECONDS // 60
    if config.POST_ONLY:
        body.append(f"Ставим пассивно, при отказе пробуем снова до {cross_min}-й минуты")
    body.append(f"Лимитка живёт до {cross_min}-й минуты, затем остаток добирается "
                f"тейкером не дороже {_cents(config.TAKER_MAX)}")
    if config.DRY_RUN:
        body.append("Режим бумажный: ордера на биржу не уходят")
    send("🤖 <b>Бот запущен</b>\n" + _quote(*body), silent=True)
