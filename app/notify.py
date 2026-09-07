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
from app.log import get

log = get("notify")

_API = "https://api.telegram.org/bot{token}/sendMessage"
_API_DOC = "https://api.telegram.org/bot{token}/sendDocument"
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


# ----------------------------------------------------------------------
# События
# ----------------------------------------------------------------------

def order_placed(hour_start: dt.datetime, side: str, price: float, shares: float,
                 proba_up: float, ev: float = 0.0, cross: bool = False,
                 bid: float | None = None, ask: float | None = None,
                 tries: int = 1) -> None:
    q = proba_up if side == "Up" else 1 - proba_up
    how = "тейкером — цена окупает комиссию" if cross else "пассивно (post-only)"
    lines = [f"📌 <b>Ордер: {side} × {shares:g} по {_cents(price)}</b>",
             _hour(hour_start),
             f"Способ: {how}",
             f"Модель даёт {q * 100:.1f}% на {side}, безубыток {price * 100:.1f}% "
             f"→ запас {(q - price) * 100:+.1f} п.п.",
             f"EV {ev:+.3f}$, в риске {shares * price:.2f}$"]
    if bid is not None or ask is not None:
        lines.append(f"Стакан по {side}: "
                     f"{_cents(bid) if bid else '—'} / {_cents(ask) if ask else '—'}")
    if tries > 1:
        lines.append(f"Встали с {tries}-й попытки")
    send("\n".join(lines), silent=True)


def filled(hour_start: dt.datetime, side: str, price: float, shares: float,
           role: str | None = None, fee: float = 0.0) -> None:
    eff = _effective(price, shares, fee)
    lines = [f"✅ <b>Исполнен: {side} × {shares:g} по {_cents(price)}</b>",
             _hour(hour_start)]
    if role == "MAKER":
        lines.append("Пассивно — комиссии нет")
    elif role:
        name = {"TAKER": "Тейкером", "MIXED": "Частями"}.get(role, role)
        lines.append(f"{name} — комиссия {fee:.3f}$")
    lines.append(f"Всего {price * shares + fee:.2f}$ = {_cents(eff)} за шейр "
                 f"(так и покажет Polymarket)")
    lines.append(f"Отбивается при {eff * 100:.1f}% угаданных")
    send("\n".join(lines))


def resolved(hour_start: dt.datetime, side: str, outcome: str, won: bool,
             pnl: float, price: float | None = None, shares: float = 0.0,
             day_pnl: float | None = None, total: dict | None = None) -> None:
    head = "🟢 <b>Победа</b>" if won else "🔴 <b>Проигрыш</b>"
    lines = [f"{head} · итог {outcome}", _hour(hour_start)]
    if price is not None:
        lines.append(f"Ставили {side} × {shares:g} по {_cents(price)}")
    lines.append(f"Результат <b>{pnl:+.2f}$</b>")
    tail = []
    if day_pnl is not None:
        tail.append(f"за сутки {day_pnl:+.2f}$")
    if total:
        tail.append(f"всего {total['P&L, $']:+.2f}$ "
                    f"({total['угадано']}/{total['ставок']}, "
                    f"{total['точность, %']:.1f}%)")
    if tail:
        lines.append(" · ".join(tail))
    send("\n".join(lines), silent=True)


def skipped(hour_start: dt.datetime, reason: str) -> None:
    send(f"⏭ <b>Пропуск часа</b>\n{_hour(hour_start)}\n{reason}", silent=True)


def error(kind: str, message: str) -> None:
    send(f"⚠️ <b>{kind}</b>\n{message}")


def daily_summary(stats: dict, day: dt.date) -> None:
    if not stats.get("ставок"):
        send(f"📊 <b>Итог за {day:%d.%m}</b>\nставок не было", silent=True)
        return
    lines = [f"📊 <b>Итог за {day:%d.%m}</b>",
             f"Ставок {stats['ставок']}, угадано {stats['угадано']} "
             f"({stats['точность, %']:.1f}%)",
             f"P&amp;L <b>{stats['P&L, $']:+.2f}$</b>"]
    fees = stats.get("комиссии, $") or 0.0
    if fees:
        lines.append(f"Комиссий уплачено {fees:.3f}$")
    maker, taker = stats.get("пассивных", 0), stats.get("тейкером", 0)
    if maker or taker:
        lines.append(f"Пассивно {maker}, тейкером {taker}")
    if stats.get("пропущено"):
        lines.append(f"Пропущено часов: {stats['пропущено']}")
    lines.append(f"На Up {stats['ставок на Up, %']:.0f}% ставок")
    send("\n".join(lines), silent=True)


def startup(problems: list[str]) -> None:
    if problems:
        send("🚫 <b>Бот не стартовал</b>\n" + "\n".join(f"• {p}" for p in problems))
        return
    lines = ["🤖 <b>Бот запущен</b>",
             f"Ставка {config.SHARES:g} шейров по {_cents(config.LIMIT_PRICE)} "
             f"— {config.SHARES * config.LIMIT_PRICE:.2f}$ в час",
             f"Решение за {config.EARLY_SECONDS // 60} мин до часа"]
    if config.POST_ONLY:
        lines.append("Ставим пассивно, при отказе пробуем снова до начала часа; "
                     "тейкером — только если цена окупает комиссию")
    if config.DRY_RUN:
        lines.append("Режим бумажный: ордера на биржу не уходят")
    send("\n".join(lines), silent=True)
