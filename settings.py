"""settings.toml — необязательные переопределения config.py.

Читает и проверяет `config.py`, в самом конце файла, после того как отработали
все значения по умолчанию. Файла нет или он пустой — ничего не меняется,
поведение такое же, как без settings.toml. Пример со всеми ключами и
пояснением к каждому — `settings.example.toml`, он в git; сам `settings.toml`
— нет (в `.gitignore`), поэтому `git pull` его не тронет и не перезапишет.

Формат — TOML: `tomllib` из стандартной библиотеки на Python 3.11 и новее.
На более старом Python (нет `tomllib`) — запасной разбор через `configparser`,
совместимое подмножество без секций: просто `ИМЯ = значение` построчно, строки
в кавычках, числа как есть, `true`/`false`. Новых зависимостей это не требует.

Секреты (приватный ключ, API-ключи и креды, токен телеграма) сюда переносить
нельзя — они остаются только в `.env`. Попытка записать такой ключ в
settings.toml — ошибка при старте, а не тихий игнор.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class SettingsError(ValueError):
    """settings.toml не прошёл проверку — бот не должен стартовать."""


#: Единственные ключи, которые разрешено переопределять. Имена совпадают с
#: переменными в config.py. Любой ключ вне этого списка (и не секрет) — ошибка
#: со списком допустимых, см. parse_file().
ALLOWED_KEYS: frozenset[str] = frozenset({
    "COIN", "DRY_RUN", "TRADING_ENABLED", "ENTRY_MODE", "EARLY_SECONDS",
    "CUTOFF_MINUTES", "HISTORY_DAYS", "FEE_RATE", "MAKER_REBATE_SHARE",
    "LIMIT_PRICE", "POST_ONLY", "CROSS_AFTER_SECONDS", "CROSS_LATEST_SECONDS",
    "TAKER_MAX", "TAKER_SLIPPAGE_TICKS", "RETRY_SECONDS", "RETRY_BURST_SECONDS",
    "SHARES", "MAX_ENTRY_PRICE", "MAX_OPEN_POSITIONS",
    "DAILY_LOSS_LIMIT_USD", "MONTHLY_LOSS_LIMIT_USD",
})

#: Ожидаемый тип каждого значения — для проверки при чтении и для запасного
#: разбора через configparser (там всё приходит строкой, её нужно во что-то
#: превратить).
_TYPES: dict[str, type] = {
    "COIN": str, "DRY_RUN": bool, "TRADING_ENABLED": bool, "ENTRY_MODE": str,
    "EARLY_SECONDS": int, "CUTOFF_MINUTES": int, "HISTORY_DAYS": int,
    "FEE_RATE": float, "MAKER_REBATE_SHARE": float, "LIMIT_PRICE": float,
    "POST_ONLY": bool, "CROSS_AFTER_SECONDS": int, "CROSS_LATEST_SECONDS": int,
    "TAKER_MAX": float, "TAKER_SLIPPAGE_TICKS": int, "RETRY_SECONDS": int,
    "RETRY_BURST_SECONDS": int, "SHARES": int, "MAX_ENTRY_PRICE": float,
    "MAX_OPEN_POSITIONS": int, "DAILY_LOSS_LIMIT_USD": float,
    "MONTHLY_LOSS_LIMIT_USD": float,
}

#: Секреты счёта — им в settings.toml не место, только в .env (см. .env.example).
SECRET_KEYS: frozenset[str] = frozenset({
    "PRIVATE_KEY", "POLY_PRIVATE_KEY", "POLYMARKET_PRIVATE_KEY",
    "API_KEY", "POLY_API_KEY", "POLYMARKET_API_KEY",
    "API_SECRET", "POLY_API_SECRET", "POLYMARKET_API_SECRET",
    "API_PASSPHRASE", "POLY_API_PASSPHRASE", "POLYMARKET_API_PASSPHRASE",
    "TELEGRAM_BOT_TOKEN", "TG_TOKEN",
})

#: Значения ENTRY_MODE/COIN, которые код уже понимает (см. комментарии в
#: config.py рядом с ENTRY_MODE и COINS).
ENTRY_MODE_CHOICES: frozenset[str] = frozenset({"early", "fresh"})
COIN_CHOICES: frozenset[str] = frozenset({"btc", "eth"})


def parse_file(path: Path) -> dict[str, Any]:
    """Читает settings.toml и проверяет ключи и типы. Нет файла/пустой -> {}."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    try:
        import tomllib
    except ImportError:
        raw = _parse_ini_like(text)
    else:
        try:
            raw = tomllib.loads(text)
        except Exception as e:                                  # noqa: BLE001
            raise SettingsError(
                f"{path.name}: файл повреждён или не в формате TOML — {e}"
            ) from e

    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key in SECRET_KEYS:
            raise SettingsError(
                f"{path.name}: параметр {key} — это секрет, ему тут не место. "
                "Приватный ключ, API-ключи и токен телеграма остаются только "
                "в .env."
            )
        if key not in ALLOWED_KEYS:
            raise SettingsError(
                f"{path.name}: неизвестный параметр {key}. Допустимые "
                f"параметры: {', '.join(sorted(ALLOWED_KEYS))}."
            )
        result[key] = _check_type(path.name, key, value)
    return result


def _check_type(filename: str, key: str, value: Any) -> Any:
    expected = _TYPES[key]
    if expected is bool:
        if not isinstance(value, bool):
            raise SettingsError(
                f"{filename}: {key} — записано {value!r}, допустимо true "
                "или false."
            )
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(
                f"{filename}: {key} — записано {value!r}, допустимо число."
            )
        return float(value)
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(
                f"{filename}: {key} — записано {value!r}, допустимо целое "
                "число."
            )
        return value
    # str
    if not isinstance(value, str):
        raise SettingsError(
            f"{filename}: {key} — записано {value!r}, допустима строка."
        )
    return value


def _parse_ini_like(text: str) -> dict[str, Any]:
    """Запасной разбор для Python <3.11 (нет tomllib): подмножество TOML без
    секций и таблиц — просто `ИМЯ = значение` построчно.
    """
    import configparser
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cp.read_string("[settings]\n" + text)
    raw: dict[str, Any] = {}
    for key, value in cp["settings"].items():
        raw[key.upper()] = _coerce_scalar(value.strip())
    return raw


def _coerce_scalar(s: str) -> Any:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s  # неэкранированная строка — тип (и вменяемость) проверит _check_type


def _cents_ok(x: float) -> bool:
    return abs(round(x * 100) - x * 100) < 1e-6


def validate_effective(values: dict[str, Any]) -> None:
    """Проверяет итоговый набор значений (умолчания + settings.toml).

    Падает на первой же ошибке с текстом по-русски: имя параметра, что
    записано, что допустимо. `values` должен содержать все ключи из
    ALLOWED_KEYS.
    """
    def fail(key: str, allowed: str) -> None:
        raise SettingsError(f"{key}: записано {values[key]!r}, допустимо {allowed}.")

    lp = values["LIMIT_PRICE"]
    if not (0.30 <= lp <= 0.70) or not _cents_ok(lp):
        fail("LIMIT_PRICE", "0.30–0.70, кратно 0.01")

    mep = values["MAX_ENTRY_PRICE"]
    if not (0.30 <= mep <= 0.99) or mep < lp:
        fail("MAX_ENTRY_PRICE", f"0.30–0.99 и не ниже LIMIT_PRICE ({lp})")

    tm = values["TAKER_MAX"]
    if not (0.30 <= tm <= 0.99) or tm < lp:
        fail("TAKER_MAX", f"0.30–0.99 и не ниже LIMIT_PRICE ({lp})")

    caf = values["CROSS_AFTER_SECONDS"]
    clat = values["CROSS_LATEST_SECONDS"]
    if not (60 <= caf <= 3000) or caf >= clat:
        fail("CROSS_AFTER_SECONDS",
             f"60–3000 и строго меньше CROSS_LATEST_SECONDS ({clat})")
    if clat <= caf or clat > 3300:
        fail("CROSS_LATEST_SECONDS",
             f"больше CROSS_AFTER_SECONDS ({caf}) и не больше 3300")

    shares = values["SHARES"]
    if shares < 5:
        fail("SHARES", "целое, не меньше 5")

    mop = values["MAX_OPEN_POSITIONS"]
    if not (1 <= mop <= 10):
        fail("MAX_OPEN_POSITIONS", "1–10")

    dl = values["DAILY_LOSS_LIMIT_USD"]
    if not dl > 0:
        fail("DAILY_LOSS_LIMIT_USD", "больше 0")

    ml = values["MONTHLY_LOSS_LIMIT_USD"]
    if ml < dl:
        fail("MONTHLY_LOSS_LIMIT_USD", f"не меньше DAILY_LOSS_LIMIT_USD ({dl})")

    fee = values["FEE_RATE"]
    if not (0 <= fee <= 0.20):
        fail("FEE_RATE", "0–0.20")

    mrs = values["MAKER_REBATE_SHARE"]
    if not (0 <= mrs <= 1):
        fail("MAKER_REBATE_SHARE", "0–1")

    cm = values["CUTOFF_MINUTES"]
    if not (1 <= cm <= 10):
        fail("CUTOFF_MINUTES", "1–10")

    es = values["EARLY_SECONDS"]
    if not (30 <= es <= 3600):
        fail("EARLY_SECONDS", "30–3600")

    coin = values["COIN"]
    if coin not in COIN_CHOICES:
        fail("COIN", "только btc или eth")

    em = values["ENTRY_MODE"]
    if em not in ENTRY_MODE_CHOICES:
        fail("ENTRY_MODE", f"только {', '.join(sorted(ENTRY_MODE_CHOICES))}")


def format_summary(overrides: dict[str, Any]) -> str:
    """Одна строка для лога и телеграма: что взято из settings.toml."""
    if not overrides:
        return "settings.toml: не найден или пуст, все параметры по умолчанию"
    items = ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items()))
    return f"settings.toml: переопределены {items}; остальные параметры по умолчанию"
