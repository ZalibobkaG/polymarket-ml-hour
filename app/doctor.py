"""`python main.py doctor` — проверка готовности бота. Ничего не запускает и не торгует.

Печатает список из девяти проверок с отметками «есть» / «нет» / «не проверено»
и код возврата 1, если хоть одна не прошла (не считая тех, что упёрлись в
отсутствие сети — это «не проверено», а не провал).

Каждая проверка независима и защищена от падения остальных: сломанный
config.py не должен уронить проверку переменных окружения, а отсутствие
сети — проверку контрольных сумм модели. Поэтому `config` импортируется один
раз в начале в try/except, а всё, что от него зависит, при неудаче печатает
«не проверено» вместо трассировки стека.

Секреты (приватный ключ, API-креды, токен телеграма) никогда не печатаются —
ни значением, ни маской: только факт «заполнено» / «пусто», и ни один запрос
к Telegram API не логирует URL с токеном.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Ровно восемь переменных из .env.example — больше и меньше не бывает.
ENV_VARS = ["POLY_PRIVATE_KEY", "POLY_FUNDER", "POLY_SIGNATURE_TYPE",
           "POLY_API_KEY", "POLY_API_SECRET", "POLY_API_PASSPHRASE",
           "TG_TOKEN", "TG_ALLOWED_USER_IDS"]

_CLOCK_SKEW_LIMIT_SECONDS = 5.0
_HTTP_TIMEOUT = 10.0


def _status(has_problem: bool) -> str:
    return "нет" if has_problem else "есть"


# --------------------------------------------------------------------------
# config.py: единственная точка, где ловится сломанный config.py
# --------------------------------------------------------------------------

def _load_config():
    """(модуль config, None) при удаче, (None, текст ошибки) при неудаче.

    Ловит любое исключение: неизвестный COIN падает в config.py обычным
    `ValueError`, опечатка в числе — `SyntaxError` — доктору всё равно нужно
    не упасть, а сказать человеку, что не так.
    """
    try:
        import config
        return config, None
    except Exception as e:                                    # noqa: BLE001
        return None, str(e)


# --------------------------------------------------------------------------
# 1. переменные окружения
# --------------------------------------------------------------------------

def check_env(problems: list[str], config_mod) -> str:
    print("1. Переменные окружения (.env, по списку .env.example)")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    filled = {}
    for name in ENV_VARS:
        value = os.environ.get(name, "")
        filled[name] = bool(value.strip())
        print(f"   {name:<20} {'заполнено' if filled[name] else 'пусто'}")
    print(f"   заполнено {sum(filled.values())} из {len(ENV_VARS)}")

    local_problems: list[str] = []
    if not (ROOT / ".env").exists():
        local_problems.append(
            ".env не найден — скопируйте .env.example в .env и заполните "
            "нужные значения."
        )
    elif (config_mod is not None and not config_mod.DRY_RUN
          and not filled["POLY_PRIVATE_KEY"]):
        local_problems.append(
            "боевой режим (DRY_RUN=False), а POLY_PRIVATE_KEY пуст — бот не "
            "сможет подписывать ордера. Заполните POLY_PRIVATE_KEY в .env "
            "или верните DRY_RUN=True."
        )

    problems.extend(local_problems)
    status = _status(bool(local_problems))
    print(f"   статус: {status}")
    return status


# --------------------------------------------------------------------------
# 2. доступность CLOB_HOST / GAMMA_HOST
# --------------------------------------------------------------------------

@dataclass
class Probe:
    label: str
    url: str
    status_code: int | None = None
    elapsed_ms: float | None = None
    date: dt.datetime | None = None
    error: str | None = None


def _probe(label: str, url: str, params: dict | None = None) -> Probe:
    """Один GET-запрос на чтение. Не бросает исключений — сбой сети такой же
    законный исход, как и ответ: разница только в том, что печатать дальше.
    """
    import httpx

    t0 = time.monotonic()
    try:
        r = httpx.get(url, params=params, timeout=_HTTP_TIMEOUT)
    except Exception as e:                                    # noqa: BLE001
        return Probe(label=label, url=url, error=str(e))
    elapsed_ms = (time.monotonic() - t0) * 1000

    server_date = None
    date_hdr = r.headers.get("date")
    if date_hdr:
        import email.utils as eut
        try:
            parsed = eut.parsedate_to_datetime(date_hdr)
            server_date = (parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None
                          else parsed.astimezone(dt.timezone.utc))
        except Exception:                                      # noqa: BLE001
            server_date = None

    return Probe(label=label, url=url, status_code=r.status_code,
                elapsed_ms=elapsed_ms, date=server_date)


def check_hosts(problems: list[str], config_mod) -> dict[str, Probe]:
    print("2. Доступность CLOB_HOST и GAMMA_HOST")
    if config_mod is None:
        print("   не проверено — не загрузился config.py (см. пункт 8)")
        return {}

    probes = {
        "CLOB_HOST": _probe("CLOB_HOST", config_mod.CLOB_HOST.rstrip("/") + "/"),
        "GAMMA_HOST": _probe("GAMMA_HOST",
                            config_mod.GAMMA_HOST.rstrip("/") + "/markets",
                            params={"limit": 1}),
    }

    any_reached = False
    local_problems: list[str] = []
    for label, p in probes.items():
        if p.error:
            print(f"   {label:<10} {p.url:<48} нет соединения: {p.error}")
            continue
        any_reached = True
        print(f"   {label:<10} {p.url:<48} код {p.status_code}, "
             f"{p.elapsed_ms:.0f} мс")
        if p.status_code is not None and p.status_code >= 500:
            local_problems.append(
                f"{label} отвечает ошибкой сервера {p.status_code} — сервис "
                "Polymarket сейчас недоступен, попробуйте позже."
            )

    if not any_reached:
        print("   статус: не проверено (нет сети)")
        return probes

    problems.extend(local_problems)
    print(f"   статус: {_status(bool(local_problems))}")
    return probes


# --------------------------------------------------------------------------
# 3. ближайший часовой рынок
# --------------------------------------------------------------------------

def _next_hour(now: pd.Timestamp, min_lead_seconds: int = 25) -> pd.Timestamp:
    """Ближайший час рынка. Копия правила `app.runner.next_hour()` без
    импорта самого runner — тот тянет за собой загрузку модели, а рынок
    нужно уметь проверить и тогда, когда модель не грузится (см. пункт 4).
    """
    now = pd.Timestamp(now)
    h = now.ceil("h")
    if h - now < pd.Timedelta(seconds=min_lead_seconds):
        h += pd.Timedelta(hours=1)
    return h


def check_market(problems: list[str], config_mod, probes: dict[str, Probe]) -> str:
    print("3. Ближайший часовой рынок")
    if config_mod is None:
        print("   не проверено — не загрузился config.py (см. пункт 8)")
        return "не проверено"

    gamma = probes.get("GAMMA_HOST")
    if gamma is None or gamma.error:
        print("   не проверено — нет сети (см. пункт 2)")
        return "не проверено"

    from app import markets

    h = _next_hour(pd.Timestamp.now(tz="UTC"))
    try:
        info = markets.find(h)
    except markets.MarketError as e:
        print(f"   рынок не найден: {e}")
        problems.append(
            f"ближайший часовой рынок {config_mod.COIN_TITLE} не найден: {e}"
        )
        print("   статус: нет")
        return "нет"
    except Exception as e:                                    # noqa: BLE001
        print(f"   не проверено — {e}")
        return "не проверено"

    bid = f"{info.best_bid_up:.3f}" if info.best_bid_up is not None else "—"
    ask = f"{info.best_ask_up:.3f}" if info.best_ask_up is not None else "—"
    print(f"   монета:       {config_mod.COIN_TITLE} ({config_mod.SYMBOL})")
    print(f"   слаг:         {info.slug}")
    print(f"   начало часа:  {info.hour_start:%Y-%m-%d %H:%M} UTC")
    print(f"   бид/аск (Up): {bid} / {ask}")
    print(f"   шаг цены:     {info.tick_size}")

    if not info.accepting_orders:
        problems.append(
            f"рынок {info.slug} найден, но сейчас не принимает заявки "
            "(accepting_orders=false) — заявку разместить не получится."
        )
        print("   статус: нет")
        return "нет"
    print("   статус: есть")
    return "есть"


# --------------------------------------------------------------------------
# 4. комплект модели
# --------------------------------------------------------------------------

def _fixture_prediction(kit_dir: Path, model_mod) -> float:
    """Тестовый прогноз без сети: по минуткам из фикстуры комплекта, на
    последнем часе, который в неё вошёл (`passport.json` -> fixtures ->
    last_hour_utc). Тот же файл, которым пользуется `kit.selfcheck`.
    """
    fixtures_path = kit_dir / "fixtures" / "minutes.csv.gz"
    passport_path = kit_dir / "passport.json"
    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    last_hour = (passport.get("fixtures") or {}).get("last_hour_utc")
    if not fixtures_path.exists() or not last_hour:
        raise RuntimeError(
            f"нет фикстуры для тестового прогноза без сети ({fixtures_path})"
        )

    minutes = pd.read_csv(fixtures_path, float_precision="round_trip")
    minutes["open_time"] = pd.to_datetime(minutes["open_time"], utc=True,
                                          format="ISO8601")
    minutes = minutes.set_index("open_time").sort_index()

    ans = model_mod.predict(minutes, pd.Timestamp(last_hour))
    return float(ans["p_up"])


def check_model(problems: list[str], config_mod) -> str:
    kit_name = config_mod.KIT_DIR.name if config_mod is not None else "?"
    print(f"4. Комплект модели ({kit_name})")
    if config_mod is None:
        print("   не проверено — не загрузился config.py (см. пункт 8)")
        return "не проверено"

    kit_dir = config_mod.KIT_DIR
    if not kit_dir.exists():
        print(f"   каталог комплекта не найден: {kit_dir}")
        problems.append(
            f"нет каталога комплекта {kit_dir} — положите комплект модели "
            "рядом с ботом или поправьте COIN в config.py."
        )
        print("   статус: нет")
        return "нет"

    required = {
        "passport.json": kit_dir / "passport.json",
        "kit/features.py": kit_dir / "kit" / "features.py",
        f"model/{config_mod.MODEL_STEM}.txt": kit_dir / "model" / f"{config_mod.MODEL_STEM}.txt",
        f"model/{config_mod.MODEL_STEM}.json": kit_dir / "model" / f"{config_mod.MODEL_STEM}.json",
    }
    missing = [rel for rel, p in required.items() if not p.exists()]
    for rel in required:
        print(f"   {rel:<28} {'НЕТ ФАЙЛА' if rel in missing else 'есть'}")
    if missing:
        problems.append(
            "в комплекте модели не хватает файлов: " + ", ".join(missing) +
            " — восстановите комплект целиком из релиза или бэкапа."
        )
        print("   статус: нет")
        return "нет"

    try:
        from app import model as model_mod
        card = model_mod.card()
    except Exception as e:                                    # noqa: BLE001
        problems.append(f"комплект модели не загрузился: {e}")
        print(f"   контрольные суммы / загрузка: ОШИБКА — {e}")
        print("   статус: нет")
        return "нет"
    print(f"   контрольные суммы:          совпали (модель {card.get('model_id')})")

    try:
        p_up = _fixture_prediction(kit_dir, model_mod)
    except Exception as e:                                    # noqa: BLE001
        problems.append(f"тестовый прогноз модели не посчитался: {e}")
        print(f"   тестовый прогноз: ОШИБКА — {e}")
        print("   статус: нет")
        return "нет"

    in_range = 0.0 <= p_up <= 1.0
    print(f"   тестовый прогноз (фикстура): p_up={p_up:.4f} "
         f"({'в диапазоне 0..1' if in_range else 'ВНЕ ДИАПАЗОНА 0..1'})")
    if not in_range:
        problems.append(f"тестовый прогноз модели вне диапазона 0..1: {p_up}")
        print("   статус: нет")
        return "нет"

    print("   статус: есть")
    return "есть"


# --------------------------------------------------------------------------
# 5. расхождение часов
# --------------------------------------------------------------------------

def check_clock(problems: list[str], probes: dict[str, Probe]) -> str:
    print("5. Расхождение часов с биржей (по заголовку HTTP Date)")
    probe = next((probes[k] for k in ("CLOB_HOST", "GAMMA_HOST")
                 if probes.get(k) and not probes[k].error and probes[k].date), None)
    if probe is None:
        print("   не проверено — нет ответа с сервера (см. пункт 2)")
        return "не проверено"

    local_now = dt.datetime.now(dt.timezone.utc)
    skew = (local_now - probe.date).total_seconds()
    print(f"   локальное время:              {local_now:%Y-%m-%d %H:%M:%S} UTC")
    print(f"   время сервера ({probe.label}): {probe.date:%Y-%m-%d %H:%M:%S} UTC")
    print(f"   расхождение:                  {skew:+.1f} с")

    if abs(skew) > _CLOCK_SKEW_LIMIT_SECONDS:
        problems.append(
            f"часы машины расходятся со временем биржи на {skew:+.1f} с "
            f"(порог {_CLOCK_SKEW_LIMIT_SECONDS:.0f} с). Решение о ставке "
            "принимается за 2 минуты до часа — при неточных часах бот "
            "промахнётся мимо момента входа. Синхронизируйте системное время "
            "(Windows: Параметры -> Время и язык -> Синхронизировать сейчас)."
        )
        print("   статус: нет")
        return "нет"
    print("   статус: есть")
    return "есть"


# --------------------------------------------------------------------------
# 6. база data/bot.db
# --------------------------------------------------------------------------

def check_db(problems: list[str], config_mod) -> str:
    print("6. База data/bot.db")
    if config_mod is None:
        print("   не проверено — не загрузился config.py (см. пункт 8)")
        return "не проверено"

    try:
        from app import storage
        storage.init()
        print(f"   файл:      {config_mod.DB_PATH}")
        print("   создаётся: да")
        with storage.session() as s:
            ev = storage.Event(level="INFO", kind="doctor",
                               message="doctor: проверка готовности")
            s.add(ev)
            s.flush()
            s.delete(ev)
        print("   пишется:   да")
        print("   миграции:  пройдены")
    except Exception as e:                                    # noqa: BLE001
        problems.append(f"база data/bot.db не работает: {e}")
        print(f"   ошибка: {e}")
        print("   статус: нет")
        return "нет"

    print("   статус: есть")
    return "есть"


# --------------------------------------------------------------------------
# 7. telegram
# --------------------------------------------------------------------------

def _looks_like_tg_id(x: str) -> bool:
    x = x.strip()
    return x not in ("", "-") and x.lstrip("-").isdigit()


def check_telegram(problems: list[str], config_mod) -> str:
    print("7. Telegram")
    if config_mod is None:
        print("   не проверено — не загрузился config.py (см. пункт 8)")
        return "не проверено"

    token = config_mod.TELEGRAM_BOT_TOKEN
    ids = config_mod.TG_ALLOWED_USER_IDS
    if not token or not ids:
        print("   не настроен (пуст TG_TOKEN или TG_ALLOWED_USER_IDS)")
        problems.append(
            "Telegram не настроен — бот не сможет присылать уведомления, и "
            "им нельзя будет управлять из чата. Заполните TG_TOKEN и "
            "TG_ALLOWED_USER_IDS в .env, если это не осознанный выбор."
        )
        print("   статус: нет")
        return "нет"

    import httpx

    try:
        r = httpx.get(f"https://api.telegram.org/bot{token}/getMe",
                      timeout=_HTTP_TIMEOUT)
        data = r.json()
    except Exception as e:                                    # noqa: BLE001
        print(f"   не проверено — нет сети: {e}")
        return "не проверено"

    if not data.get("ok"):
        desc = data.get("description", f"HTTP {r.status_code}")
        print(f"   токен: НЕ РАБОТАЕТ ({desc})")
        problems.append(f"токен телеграма не работает: {desc}. Проверьте "
                        "TG_TOKEN в .env.")
        print("   статус: нет")
        return "нет"
    print(f"   токен рабочий: да (бот @{data['result'].get('username', '?')})")

    bad_ids = [x for x in ids if not _looks_like_tg_id(x)]
    if bad_ids:
        print(f"   id получателя: НЕ ПОХОЖ на число: {', '.join(bad_ids)}")
        problems.append(
            "TG_ALLOWED_USER_IDS содержит значение, не похожее на числовой "
            f"user_id: {', '.join(bad_ids)}. Поправьте .env."
        )
        print("   статус: нет")
        return "нет"
    print(f"   id получателя: допустим ({config_mod.TELEGRAM_CHAT_ID})")

    try:
        rc = httpx.get(f"https://api.telegram.org/bot{token}/getChat",
                       params={"chat_id": config_mod.TELEGRAM_CHAT_ID},
                       timeout=_HTTP_TIMEOUT)
        cdata = rc.json()
    except Exception as e:                                    # noqa: BLE001
        print(f"   чат: не проверено ({e})")
        print("   статус: есть")
        return "есть"

    if not cdata.get("ok"):
        desc = cdata.get("description", "")
        print(f"   чат: недоступен ({desc})")
        problems.append(
            "Telegram-бот не может написать получателю "
            f"{config_mod.TELEGRAM_CHAT_ID}: {desc}. Обычно значит, что "
            "получатель ещё не открыл бота и не нажал Start."
        )
        print("   статус: нет")
        return "нет"

    print("   чат: доступен, бот может писать")
    print("   статус: есть")
    return "есть"


# --------------------------------------------------------------------------
# 8. config.py
# --------------------------------------------------------------------------

#: Параметры, которые печатаются в пункте 8, — те, что напарнику запрещено
#: менять (docs/НАПАРНИК_ПРОГОН.md, раздел 7) плюс размер ставки и лимиты.
_CONFIG_KEYS = ("COIN", "LIMIT_PRICE", "POST_ONLY", "CROSS_AFTER_SECONDS",
                "TAKER_MAX", "MAX_ENTRY_PRICE", "CUTOFF_MINUTES", "SHARES",
                "DAILY_LOSS_LIMIT_USD", "MONTHLY_LOSS_LIMIT_USD", "TICK_SECONDS")


def check_config(problems: list[str], config_mod, config_error: str | None) -> str:
    print("8. config.py")
    if config_error is not None:
        print("   config.py не загрузился:")
        print(f"     {config_error}")
        problems.append(f"config.py не загрузился: {config_error}")
        print("   статус: нет")
        return "нет"
    for key in _CONFIG_KEYS:
        print(f"   {key:<24}{getattr(config_mod, key, '?')!r}")
    print("   статус: есть")
    return "есть"


# --------------------------------------------------------------------------
# 9. режим работы
# --------------------------------------------------------------------------

def check_mode(config_mod) -> None:
    print("9. РЕЖИМ РАБОТЫ")
    if config_mod is None:
        print("   не проверено — не загрузился config.py (см. пункт 8)")
        return

    bar = "=" * 56
    print("   " + bar)
    dry = config_mod.DRY_RUN
    print(f"   DRY_RUN = {dry}" + (
        "   -> БУМАГА, ордера на биржу НЕ отправляются" if dry
        else "   -> !!! БОЙ, ордера отправляются по-настоящему !!!"))
    enabled = config_mod.TRADING_ENABLED
    print(f"   TRADING_ENABLED = {enabled}" + (
        "   -> новые ставки разрешены" if enabled
        else "   -> новые ставки ЗАПРЕЩЕНЫ (рубильник выключен)"))
    print("   " + bar)


# --------------------------------------------------------------------------
# оркестрация
# --------------------------------------------------------------------------

def run() -> int:
    print("=" * 60)
    print("  python main.py doctor -- проверка готовности")
    print("=" * 60)
    print()

    config_mod, config_error = _load_config()
    problems: list[str] = []

    check_env(problems, config_mod)
    print()
    probes = check_hosts(problems, config_mod)
    print()
    check_market(problems, config_mod, probes)
    print()
    check_model(problems, config_mod)
    print()
    check_clock(problems, probes)
    print()
    check_db(problems, config_mod)
    print()
    check_telegram(problems, config_mod)
    print()
    check_config(problems, config_mod, config_error)
    print()
    check_mode(config_mod)
    print()

    print("-" * 60)
    if problems:
        print(f"ПРОБЛЕМЫ ({len(problems)}):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        print()
        print("Бот НЕ готов — см. проблемы выше.")
        return 1

    print("Проблем не найдено.")
    return 0
