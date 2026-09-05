"""Точка входа.

    python main.py check       предполётные проверки, ничего не торгует
    python main.py check --live-order
                               то же плюс настоящий ордер по цене одного тика
                               (не исполнится, снимается сразу) — проверка, что
                               биржа наши ордера принимает
    python main.py once        один проход цикла (решение/заполнения/резолвы)
    python main.py run         рабочий режим: бесконечный цикл.
                               Управление из Telegram: /status /stats /pause
                               /resume /stop — только с id из TG_ALLOWED_USER_IDS
    python main.py stats       сводка по накопленным сделкам
    python main.py retrain     переобучить модель на новых данных (см. app/retrain.py);
                               живой бот подхватит её сам, без перезапуска

Перед первым запуском:
    cp .env.example .env       и заполнить (для боевого режима)

Модель — комплект `poly_hourly_kit/` от напарника, уже обучена и лежит в репозитории
готовым файлом. Раз в интервал обучения её стоит обновлять командой `retrain` —
удобнее всего через cron раз в месяц, см. docstring `app/retrain.py`.
"""

from __future__ import annotations

import datetime as dt
import sys
import threading

import pandas as pd

import config
from app import commands, features, notify, risk, storage
from app import log as applog
from app.log import get

log = get("main")

#: Как часто крутится основной цикл. Решение привязано к часу, а не к тику,
#: поэтому частый опрос нужен только для отслеживания заполнения.
TICK_SECONDS = 30

#: Сколько строк лога шлёт `/log` без аргумента. Сутки — это около тысячи строк,
#: так что по умолчанию уезжает примерно последний день, а не весь файл.
LOG_TAIL_LINES = 2000


def cmd_check(live_order: bool = False) -> int:
    print(f"режим: {'БУМАГА (ордера не отправляются)' if config.DRY_RUN else 'БОЙ'}")
    problems = risk.preflight()

    print("\nсамопроверка модели (живые минутки → poly_hourly_kit)…")
    try:
        ans = features.verify(n=3)
        print(f"  ок: p_up={ans['p_up']:.4f} на час {ans['feature_hour_utc']}, "
              f"модель {ans['model_id']}")
    except Exception as e:                                    # noqa: BLE001
        problems.append(f"самопроверка модели не прошла: {e}")

    print("\nсвязь с Polymarket…")
    info = None
    try:
        from app import markets, runner
        h = runner.next_hour()
        info = markets.find(h)
        print(f"  рынок {info.slug}")
        print(f"  стакан Up: bid={info.best_bid_up} ask={info.best_ask_up}, "
              f"мин. размер {info.min_size:g}, шаг {info.tick_size}")
        if config.SHARES < info.min_size:
            problems.append(f"SHARES={config.SHARES:g} меньше минимума рынка "
                            f"{info.min_size:g}")
    except Exception as e:                                    # noqa: BLE001
        problems.append(f"рынок не найден: {e}")

    # Сборка и подпись настоящего ордера. На биржу ничего не уходит, но проходится
    # ровно тот код, что и в бою: тип подписи, шаг цены, размер, срок жизни.
    # Именно здесь ловятся отказы вида «Invalid order inputs» — до начала часа,
    # а не за две минуты до него.
    if info is not None and config.PRIVATE_KEY:
        print("\nсборка и подпись ордера (на биржу не отправляется)…")
        try:
            from app.trader import SIGNATURE_TYPES, ClobTrader
            t = ClobTrader()
            signed = t.build_order(info, "Up", config.LIMIT_PRICE, config.SHARES)
            print(f"  подписан: тип {config.SIGNATURE_TYPE} "
                  f"({SIGNATURE_TYPES[config.SIGNATURE_TYPE]}), "
                  f"{config.SHARES:g} шейров по {config.LIMIT_PRICE:.2f}")
            if live_order:
                problems += _live_order_test(t, info)
        except Exception as e:                                # noqa: BLE001
            problems.append(f"ордер не собирается: {type(e).__name__}: {e}")

    print("\ntelegram:", "настроен" if notify.enabled() else "не настроен")

    if problems:
        print("\nПРОБЛЕМЫ:")
        for p in problems:
            print("  •", p)
        return 1
    print("\nвсё в порядке, можно запускать: python main.py run")
    return 0


def _live_order_test(t, info) -> list[str]:
    """Единственная проверка, которую нельзя сделать вхолостую: примет ли биржа ордер.

    Ставим покупку по минимальной цене шага — такая заявка гарантированно не
    исполнится (рынок стоит около 50 центов) — и сразу снимаем её. Под риском
    остаётся размер минимального ордера по цене одного тика, у часовых рынков это
    центы. Всё остальное — подпись, размер, срок, права — проверяется без денег.
    """
    price = info.tick_size
    risk_usd = price * config.SHARES
    print(f"\nбоевая проверка приёма ордера: покупка {config.SHARES:g} × {price:.2f} "
          f"= {risk_usd:.2f}$ под риском, снимаем сразу")
    res = t.place(info, "Up", price, config.SHARES)
    if not res.ok:
        return [f"биржа не приняла ордер: {res.error}"]
    print(f"  принят, id {res.order_id}")
    if res.filled_shares:
        print(f"  ВНИМАНИЕ: неожиданно исполнилось {res.filled_shares:g} шейров")
    if t.cancel(res.order_id):
        print("  снят")
        return []
    return [f"ордер {res.order_id} принят, но НЕ снялся — снимите вручную"]


def cmd_once() -> int:
    problems = risk.preflight()
    if problems:
        for p in problems:
            log.error("предполётная проверка: %s", p)
        return 1
    from app.runner import Bot
    bot = Bot()
    try:
        bot.tick()
    finally:
        bot.close()
    return 0


def _tg_handlers(bot, stop: threading.Event) -> dict:
    """Команды из Telegram. Каждая возвращает текст ответа автору."""
    from app import runner

    def status(_: str = "") -> str:
        now = dt.datetime.now(dt.timezone.utc)
        midnight = dt.datetime.combine(now.date(), dt.time.min, tzinfo=dt.timezone.utc)
        with storage.session() as s:
            day = storage.pnl_since(s, midnight)
            total = storage.stats(s)
            live = s.query(storage.Market).filter(
                storage.Market.status.in_([storage.ORDERED, storage.FILLED,
                                           storage.PENDING])).all()
            rows = [(storage.as_utc(m.hour_start), m.status, m.side,
                     m.filled_shares, m.fill_price or m.price,
                     m.place_attempts or 0) for m in live]
        h = runner.next_hour(now)
        left = int((h - pd.Timestamp(now)).total_seconds()
                   - config.EARLY_SECONDS)
        lines = [f"🤖 <b>{'НА ПАУЗЕ' if bot.paused else 'Работает'}</b>",
                 f"Следующее решение через {max(left, 0) // 60} мин "
                 f"{max(left, 0) % 60} сек (рынок {h:%H:%M} UTC)",
                 f"За сутки {day:+.2f}$ · всего {total['P&L, $']:+.2f}$"]
        if total["ставок"]:
            lines.append(f"Ставок {total['ставок']}, угадано {total['угадано']} "
                         f"({total['точность, %']:.1f}%), "
                         f"комиссий {total.get('комиссии, $', 0):.3f}$")
        if rows:
            lines.append("В работе:")
            for hs, st, side, sh, fp, att in rows:
                if st == storage.PENDING:
                    lines.append(f"• {hs:%d.%m %H:%M} {side} по "
                                 f"{(fp or 0) * 100:.1f}¢ — ставлю пассивно, "
                                 f"попыток {att}")
                else:
                    lines.append(f"• {hs:%d.%m %H:%M} {side} × {sh or 0:g} "
                                 f"по {(fp or 0) * 100:.1f}¢ — {st}")
        else:
            lines.append("Открытых позиций нет")
        return "\n".join(lines)

    def stop_cmd(_: str = "") -> str:
        stop.set()
        return ("🛑 Останавливаюсь. Открытые позиции не закрываю: рынок "
                "резолвится сам, выигрыш гасится автоматически, а исход "
                "допишется в базу при следующем запуске.")

    def pause(_: str = "") -> str:
        bot.paused = True
        return ("⏸ Пауза. Новых ставок не делаю, начатое довожу до резолва. "
                "Снять — /resume")

    def resume(_: str = "") -> str:
        bot.paused = False
        return "▶️ Продолжаю. Ставки со следующего часа."

    def log_cmd(arg: str = "") -> str:
        """Прислать текущий лог файлом. `/log` — хвост, `/log all` — целиком.

        По умолчанию хвост, а не весь файл: за сутки набегает под тысячу строк,
        и читать с телефона интересен обычно конец. Число строк можно задать
        явно: `/log 500`.
        """
        arg = (arg or "").strip().lower()
        if arg in ("", "tail"):
            lines: int | None = LOG_TAIL_LINES
        elif arg in ("all", "весь", "всё", "full"):
            lines = None
        elif arg.isdigit() and int(arg) > 0:
            lines = int(arg)
        else:
            return ("Не понял аргумент. Так можно:\n"
                    f"/log — последние {LOG_TAIL_LINES} строк\n"
                    "/log 500 — последние 500 строк\n"
                    "/log all — файл целиком")

        try:
            data = applog.tail(lines)
        except FileNotFoundError as e:
            return str(e)
        if not data:
            return "Лог пуст — писать пока нечего."

        now = dt.datetime.now(dt.timezone.utc)
        name = f"bot-{now:%Y%m%d-%H%M}Z.txt"
        n_lines = data.count(b"\n")
        what = "целиком" if lines is None else f"последние {n_lines} строк"
        caption = (f"📄 <b>Лог</b> ({what}, {len(data) / 1024:.0f} КБ)\n"
                   f"{applog.path().name} на {now:%d.%m %H:%M} UTC")
        if notify.send_document(data, name, caption=caption):
            return ""                       # файл уже ушёл, дублировать нечем
        return "Не удалось отправить файл — подробности в логе."

    def stats_cmd(_: str = "") -> str:
        with storage.session() as s:
            month = storage.stats(s, dt.datetime.now(dt.timezone.utc)
                                  - dt.timedelta(days=30))
        if not month["ставок"]:
            return "За 30 дней ставок не было"
        return ("📊 <b>За 30 дней</b>\n"
                f"Ставок {month['ставок']}, угадано {month['угадано']} "
                f"({month['точность, %']:.1f}%)\n"
                f"P&amp;L {month['P&L, $']:+.2f}$, комиссий "
                f"{month.get('комиссии, $', 0):.3f}$\n"
                f"Пассивно {month.get('пассивных', 0)}, "
                f"тейкером {month.get('тейкером', 0)}, "
                f"пропущено {month.get('пропущено', 0)}")

    handlers = {"stop": stop_cmd, "status": status, "pause": pause,
                "resume": resume, "stats": stats_cmd, "log": log_cmd}
    handlers["help"] = lambda _="": ("Команды:\n"
                                     "/status — что происходит сейчас\n"
                                     "/stats — итоги за 30 дней\n"
                                     "/log — прислать лог файлом "
                                     "(/log 500, /log all)\n"
                                     "/pause — не делать новых ставок\n"
                                     "/resume — снять паузу\n"
                                     "/stop — остановить бота")
    return handlers


def cmd_run() -> int:
    problems = risk.preflight()
    if problems:
        for p in problems:
            log.error("предполётная проверка: %s", p)
        notify.startup(problems)
        return 1

    from app.runner import Bot
    stop = threading.Event()
    bot = Bot(stopping=stop.is_set)
    listener = commands.Listener(_tg_handlers(bot, stop))
    listener.start()
    notify.startup([])
    log.info("цикл запущен, режим %s, тик %s сек",
             "БУМАГА" if config.DRY_RUN else "БОЙ", TICK_SECONDS)

    last_summary_day = dt.datetime.now(dt.timezone.utc).date()
    code = 0
    while not stop.is_set():
        try:
            bot.tick()
            today = dt.datetime.now(dt.timezone.utc).date()
            if today != last_summary_day:
                with storage.session() as s:
                    since = dt.datetime.combine(last_summary_day, dt.time.min,
                                                tzinfo=dt.timezone.utc)
                    notify.daily_summary(storage.stats(s, since), last_summary_day)
                last_summary_day = today
        except KeyboardInterrupt:
            log.info("остановка по Ctrl+C")
            break
        except Exception as e:                                # noqa: BLE001
            # Цикл не должен умирать от единичной ошибки: следующий час важнее.
            log.exception("ошибка в цикле")
            storage.log_event("ERROR", "tick", str(e))
            notify.error("Ошибка в цикле", str(e)[:500])
        # Ожидание прерываемое: команда /stop не должна ждать конца тика.
        stop.wait(TICK_SECONDS)

    if stop.is_set():
        log.info("остановка по команде из Telegram")
    listener.stop()
    bot.close()
    return code


def cmd_stats() -> int:
    storage.init()
    with storage.session() as s:
        overall = storage.stats(s)
        month = storage.stats(s, dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30))
        open_n = len(storage.open_positions(s))
    print(f"режим: {'БУМАГА' if config.DRY_RUN else 'БОЙ'}")
    print("\n=== за всё время ===")
    for k, v in overall.items():
        print(f"  {k:18s} {v if not isinstance(v, float) else round(v, 2)}")
    print("\n=== за 30 дней ===")
    for k, v in month.items():
        print(f"  {k:18s} {v if not isinstance(v, float) else round(v, 2)}")
    print(f"\nоткрытых позиций: {open_n}")
    return 0


def cmd_retrain() -> int:
    from app import retrain
    return retrain.run()


COMMANDS = {"check": cmd_check, "once": cmd_once, "run": cmd_run, "stats": cmd_stats,
           "retrain": cmd_retrain}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd not in COMMANDS:
        print(__doc__)
        return 2
    if cmd == "check":
        return cmd_check(live_order="--live-order" in sys.argv)
    return COMMANDS[cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
