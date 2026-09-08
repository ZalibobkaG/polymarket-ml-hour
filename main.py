"""Точка входа.

    python main.py check       предполётные проверки, ничего не торгует
    python main.py check --live-order
                               то же плюс настоящий ордер по цене одного тика
                               (не исполнится, снимается сразу) — проверка, что
                               биржа наши ордера принимает
    python main.py once        один проход цикла (решение/заполнения/резолвы)
    python main.py run         рабочий режим: бесконечный цикл.
                               Управление из Telegram: /menu (кнопки), /status
                               /stats /log /pause /resume /stop — только с id
                               из TG_ALLOWED_USER_IDS
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
from app import charts, commands, features, notify, risk, storage
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
            log.error("preflight check: {}", p)
        return 1
    from app.runner import Bot
    bot = Bot()
    try:
        bot.tick()
    finally:
        bot.close()
    return 0


def _tg_handlers(bot, stop: threading.Event) -> dict:
    """Команды из Telegram. Каждая возвращает текст ответа автору.

    Возврат — `str` либо `commands.Reply`, если под ответом нужны кнопки.
    Кнопка несёт ту же строку, что набирается руками, поэтому меню и команды
    не могут разъехаться: см. docstring `app/commands.py`.
    """
    from app import runner

    def menu_buttons() -> list[list[tuple[str, str]]]:
        """Кнопки меню под текущее состояние бота.

        Собираются заново на каждый показ: пауза и работа — это одна кнопка,
        и подпись у неё должна отвечать тому, что произойдёт по нажатию.
        """
        toggle = (("▶️ Продолжить", "/resume") if bot.paused
                  else ("⏸ Пауза", "/pause"))
        return [[("📊 Статус", "/status"), ("📈 Статистика", "/stats")],
                [toggle, ("📄 Лог", "/log")],
                [("💰 Ставка", "/shares")],
                [("🛑 Остановить", "/stop ask")]]

    def menu(_: str = "") -> commands.Reply:
        return commands.Reply(
            "🤖 <b>Управление</b>\n"
            f"Сейчас: {'на паузе' if bot.paused else 'работает'}",
            buttons=menu_buttons())

    def status(_: str = "") -> commands.Reply:
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
        return commands.Reply("\n".join(lines), buttons=menu_buttons())

    def stop_cmd(arg: str = "") -> "str | commands.Reply":
        """Остановка бота. С кнопки — через переспрос, руками — сразу.

        Набранный `/stop` — осознанное действие, а кнопку легко задеть пальцем,
        поэтому в меню она ведёт на подтверждение (`/stop ask`), а не на выход.
        """
        if arg.strip().lower() == "ask":
            return commands.Reply(
                "🛑 <b>Остановить бота?</b>\n"
                "Открытые позиции останутся: рынок резолвится сам, а исход "
                "допишется в базу при следующем запуске.",
                buttons=[[("✅ Да, остановить", "/stop yes"),
                          ("↩️ Отмена", "/menu")]])
        stop.set()
        return ("🛑 Останавливаюсь. Открытые позиции не закрываю: рынок "
                "резолвится сам, выигрыш гасится автоматически, а исход "
                "допишется в базу при следующем запуске.")

    def pause(_: str = "") -> commands.Reply:
        bot.paused = True
        return commands.Reply(
            "⏸ Пауза. Новых ставок не делаю, начатое довожу до резолва.",
            buttons=menu_buttons())

    def resume(_: str = "") -> commands.Reply:
        bot.paused = False
        return commands.Reply("▶️ Продолжаю. Ставки со следующего часа.",
                              buttons=menu_buttons())

    def shares_cmd(arg: str = "") -> commands.Reply:
        """Изменить размер ставки на лету. `/shares` — показать текущий.

        Действует со следующего решения — открытые позиции не трогает (у них
        размер уже зафиксирован в своей строке `Market`). Пишется в БД
        (`storage.ShareChange`), поэтому переживает рестарт бота и остаётся
        вертикальной линией на графике `/stats equity`.
        """
        arg = (arg or "").strip()
        with storage.session() as s:
            current = storage.current_shares(s)
        if not arg:
            return commands.Reply(
                f"💰 Сейчас ставка {current:g} шейров.\nИзменить: /shares 15",
                buttons=menu_buttons())
        try:
            new = float(arg)
        except ValueError:
            return commands.Reply("Не понял число. Пример: /shares 15",
                                  buttons=menu_buttons())
        if new < 5:                     # orderMinSize у часовых рынков — меньше нельзя
            return commands.Reply(
                "Меньше 5 нельзя — минимум ордера на бирже (orderMinSize).",
                buttons=menu_buttons())
        with storage.session() as s:
            change = storage.set_shares(s, new)
        if change is None:
            return commands.Reply(f"Уже стоит {current:g} — менять нечего.",
                                  buttons=menu_buttons())
        return commands.Reply(
            f"✅ Ставка изменена: {change.old_shares:g} → {change.new_shares:g} "
            f"шейров.\nПрименится со следующего решения, открытые позиции "
            f"не трогает.", buttons=menu_buttons())

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

    def stats_menu_buttons() -> list[list[tuple[str, str]]]:
        return [[("📅 30 дней", "/stats 30d"), ("📈 Equity", "/stats equity")],
                [("🎯 Уверенность модели", "/stats confidence")],
                [("🪙 Угадывания", "/stats walk")],
                [("ℹ️ Как это считается", "/stats about")],
                [("⬅️ В меню", "/menu")]]

    _CHARTS = {
        "equity": (charts.equity_curve, "📈 <b>Equity curve</b>",
                  "Пока нечего показать — ни одна ставка ещё не резолвнулась."),
        "confidence": (charts.confidence_curve, "🎯 <b>Уверенность модели</b>",
                       "Пока нечего показать — решений ещё не было."),
        "walk": (charts.guess_walk, "🪙 <b>Угадывания</b>",
                "Пока нечего показать — резолвнутых рынков ещё нет."),
    }

    def stats_30d() -> str:
        with storage.session() as s:
            month = storage.stats(s, dt.datetime.now(dt.timezone.utc)
                                  - dt.timedelta(days=30))
        if not month["ставок"]:
            return "📅 <b>За 30 дней</b>\nставок не было"
        return ("📅 <b>За 30 дней</b>\n"
                f"Ставок {month['ставок']}, угадано {month['угадано']} "
                f"({month['точность, %']:.1f}%)\n"
                f"P&amp;L {month['P&L, $']:+.2f}$, комиссий "
                f"{month.get('комиссии, $', 0):.3f}$\n"
                f"Пассивно {month.get('пассивных', 0)}, "
                f"тейкером {month.get('тейкером', 0)}, "
                f"пропущено {month.get('пропущено', 0)}")

    _STATS_ABOUT = (
        "ℹ️ <b>Как читать графики</b>\n"
        "<blockquote><b>📈 Equity curve</b>\n"
        "Кумулятивный P&amp;L по резолвнутым ставкам (только заполненные), по "
        "порядку часов рынка. Каждая точка — сумма выигрышей и проигрышей "
        "минус уплаченные комиссии на этот момент.</blockquote>\n"
        "<blockquote><b>🎯 Уверенность модели</b>\n"
        "P(Up), которую модель даёт каждому часу, где принималось решение — "
        "включая пропущенные и незаполненные, не только реальные ставки. "
        "Пунктир на 50% — модель не видит перевеса ни в одну сторону; чем "
        "дальше линия от него в любую сторону, тем увереннее прогноз.</blockquote>\n"
        "<blockquote><b>🪙 Угадывания</b>\n"
        "Кумулятивная сумма ±1 (угадали/нет), у каждой линии свой шаг вправо — "
        "считаются только её собственные события. Серая линия «по всем» — "
        "каждый рынок с решённой стороной, даже если ордер не встал. Синяя "
        "«по заполненным» — только там, где реально была ставка. Расхождение "
        "линий показывает, сколько результата съедает именно исполнение, а не "
        "качество прогноза.</blockquote>")

    def stats_cmd(arg: str = "") -> "str | commands.Reply | None":
        arg = (arg or "").strip().lower()

        if arg in ("", "menu"):
            return commands.Reply("📊 <b>Статистика</b>\nЧто показать?",
                                  buttons=stats_menu_buttons())
        if arg in ("30d", "30", "days", "месяц"):
            return commands.Reply(stats_30d(), buttons=stats_menu_buttons())
        if arg == "about":
            return commands.Reply(_STATS_ABOUT, buttons=stats_menu_buttons())
        if arg in _CHARTS:
            fn, caption, empty_msg = _CHARTS[arg]
            png = fn()
            if png is None:
                return commands.Reply(empty_msg, buttons=stats_menu_buttons())
            if notify.send_photo(png, caption=caption, buttons=stats_menu_buttons()):
                return None                 # картинка уже ушла, дублировать нечем
            return commands.Reply("Не удалось отправить график — подробности в логе.",
                                  buttons=stats_menu_buttons())
        return commands.Reply("Не понял, что показать.", buttons=stats_menu_buttons())

    def help_cmd(_: str = "") -> commands.Reply:
        return commands.Reply(
            "<b>Команды</b>\n"
            "/menu — меню с кнопками\n"
            "/status — что происходит сейчас\n"
            "/stats — статистика: 30 дней, графики (equity, уверенность модели, "
            "угадывания) и разбор, как они считаются\n"
            "/log — прислать лог файлом (/log 500, /log all)\n"
            "/shares — изменить размер ставки (/shares 15)\n"
            "/pause — не делать новых ставок\n"
            "/resume — снять паузу\n"
            "/stop — остановить бота\n\n"
            "Те же команды лежат в меню ☰ рядом с полем ввода.",
            buttons=menu_buttons())

    # Порядок здесь и есть порядок в меню ☰ Telegram: сверху то, чем пользуешься
    # каждый день, снизу — остановка.
    return {
        "menu": commands.Command(menu, "меню с кнопками"),
        "status": commands.Command(status, "что происходит сейчас"),
        "stats": commands.Command(stats_cmd, "статистика: 30д, графики"),
        "log": commands.Command(log_cmd, "прислать лог файлом"),
        "shares": commands.Command(shares_cmd, "изменить размер ставки"),
        "pause": commands.Command(pause, "не делать новых ставок"),
        "resume": commands.Command(resume, "снять паузу"),
        "stop": commands.Command(stop_cmd, "остановить бота"),
        "help": commands.Command(help_cmd, "список команд"),
    }


def cmd_run() -> int:
    problems = risk.preflight()
    if problems:
        for p in problems:
            log.error("preflight check: {}", p)
        notify.startup(problems)
        return 1

    from app.runner import Bot
    stop = threading.Event()
    bot = Bot(stopping=stop.is_set)
    listener = commands.Listener(_tg_handlers(bot, stop))
    listener.start()
    notify.startup([])
    log.success("loop started, mode {}, tick {}s",
               "PAPER" if config.DRY_RUN else "LIVE", TICK_SECONDS)

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
            log.info("stopped by Ctrl+C")
            break
        except Exception as e:                                # noqa: BLE001
            # Цикл не должен умирать от единичной ошибки: следующий час важнее.
            log.exception("error in the loop")
            storage.log_event("ERROR", "tick", str(e))
            notify.error("Ошибка в цикле", str(e)[:500])
        # Ожидание прерываемое: команда /stop не должна ждать конца тика.
        stop.wait(TICK_SECONDS)

    if stop.is_set():
        log.info("stopped via Telegram command")
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
