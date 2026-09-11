"""Проверки settings.toml: разбор, ошибки, применение к config.py.

    python -m pytest tests/test_settings.py

Юнит-тесты (сценарии 1-7) бьют по settings.py напрямую, без импорта config —
он импортируется один раз на весь прогон pytest (см. test_execution.py), и
перечитывать его с другим settings.toml здесь означало бы менять глобальное
состояние для всех остальных тестов пакета. Последний сценарий (8, корректное
перекрытие) прогоняет config.py в отдельном процессе — так проверяется
реальный путь "config.py читает settings.toml в конце файла", не трогая
процесс, в котором уже импортирован config для других тестов.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import settings  # noqa: E402


# 1. файла нет -----------------------------------------------------------

def test_no_file(tmp_path):
    assert settings.parse_file(tmp_path / "settings.toml") == {}


# 2. пустой файл -----------------------------------------------------------

def test_empty_file(tmp_path):
    p = tmp_path / "settings.toml"
    p.write_text("", encoding="utf-8")
    assert settings.parse_file(p) == {}

    p.write_text("\n\n   \n", encoding="utf-8")
    assert settings.parse_file(p) == {}


# 3. неизвестный ключ -----------------------------------------------------------

def test_unknown_key(tmp_path):
    p = tmp_path / "settings.toml"
    p.write_text('FOO_BAR = 1\n', encoding="utf-8")
    with pytest.raises(settings.SettingsError, match="неизвестный параметр FOO_BAR"):
        settings.parse_file(p)


# 4. значение вне диапазона -----------------------------------------------------------

def test_out_of_range():
    values = _defaults()
    values["LIMIT_PRICE"] = 0.90  # выше 0.70
    with pytest.raises(settings.SettingsError, match="LIMIT_PRICE"):
        settings.validate_effective(values)


def test_out_of_range_via_file(tmp_path):
    p = tmp_path / "settings.toml"
    p.write_text("SHARES = 2\n", encoding="utf-8")  # < 5
    overrides = settings.parse_file(p)
    values = _defaults()
    values.update(overrides)
    with pytest.raises(settings.SettingsError, match="SHARES"):
        settings.validate_effective(values)


# 5. TAKER_MAX ниже LIMIT_PRICE -----------------------------------------------------------

def test_taker_max_below_limit_price():
    values = _defaults()
    values["LIMIT_PRICE"] = 0.55
    values["TAKER_MAX"] = 0.50
    with pytest.raises(settings.SettingsError, match="TAKER_MAX"):
        settings.validate_effective(values)


# 6. CROSS_AFTER_SECONDS больше CROSS_LATEST_SECONDS -----------------------------------------------------------

def test_cross_after_past_cross_latest():
    values = _defaults()
    values["CROSS_AFTER_SECONDS"] = 1300
    values["CROSS_LATEST_SECONDS"] = 1200
    with pytest.raises(settings.SettingsError, match="CROSS_AFTER_SECONDS"):
        settings.validate_effective(values)


# 7. секрет в файле -----------------------------------------------------------

@pytest.mark.parametrize("key", ["PRIVATE_KEY", "POLY_PRIVATE_KEY", "API_KEY",
                                  "TG_TOKEN", "TELEGRAM_BOT_TOKEN"])
def test_secret_rejected(tmp_path, key):
    p = tmp_path / "settings.toml"
    p.write_text(f'{key} = "x"\n', encoding="utf-8")
    with pytest.raises(settings.SettingsError, match="секрет"):
        settings.parse_file(p)


# 8. корректное перекрытие двух значений (сквозной прогон config.py) --------

def test_two_values_applied_via_config(tmp_path):
    settings_file = ROOT / "settings.toml"
    assert not settings_file.exists(), (
        "в рабочем каталоге уже есть settings.toml — тест его не трогает, "
        "но не должен запускаться поверх чужого файла"
    )
    settings_file.write_text(
        'SHARES = 15\nMAX_OPEN_POSITIONS = 5\n', encoding="utf-8"
    )
    try:
        code = (
            "import config\n"
            "assert config.SHARES == 15, config.SHARES\n"
            "assert config.MAX_OPEN_POSITIONS == 5, config.MAX_OPEN_POSITIONS\n"
            "assert config.LIMIT_PRICE == 0.51, config.LIMIT_PRICE\n"
            "print('OK', config.SETTINGS_OVERRIDES)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout
    finally:
        settings_file.unlink(missing_ok=True)


def test_settings_error_stops_config_import(tmp_path):
    settings_file = ROOT / "settings.toml"
    assert not settings_file.exists()
    settings_file.write_text('TAKER_MAX = 0.10\n', encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import config"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "SettingsError" in result.stderr
        assert "TAKER_MAX" in result.stderr
    finally:
        settings_file.unlink(missing_ok=True)


# --- `python main.py settings` --------------------------------------------

def test_cmd_settings_valid(tmp_path):
    settings_file = ROOT / "settings.toml"
    assert not settings_file.exists()
    settings_file.write_text('SHARES = 15\nMAX_OPEN_POSITIONS = 5\n',
                              encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "settings"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SHARES" in result.stdout and "15" in result.stdout
        assert "MAX_OPEN_POSITIONS" in result.stdout and "5" in result.stdout
        assert "settings.toml" in result.stdout  # источник хотя бы для одной строки
        assert "по умолчанию" in result.stdout  # и для остальных
        assert "Traceback" not in result.stdout + result.stderr
    finally:
        settings_file.unlink(missing_ok=True)


def test_cmd_settings_invalid(tmp_path):
    settings_file = ROOT / "settings.toml"
    assert not settings_file.exists()
    settings_file.write_text('TAKER_MAX = 0.10\n', encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "settings"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "TAKER_MAX" in result.stdout
        assert "не прошёл проверку" in result.stdout
        assert "Traceback" not in result.stdout + result.stderr
    finally:
        settings_file.unlink(missing_ok=True)


# --------------------------------------------------------------------------

def _defaults() -> dict:
    """Копия нынешних значений по умолчанию из config.py — не импортирует
    config (см. docstring модуля), чтобы не трогать общее состояние.
    """
    return {
        "COIN": "btc", "DRY_RUN": True, "TRADING_ENABLED": True,
        "ENTRY_MODE": "early", "EARLY_SECONDS": 120, "CUTOFF_MINUTES": 2,
        "HISTORY_DAYS": 60, "FEE_RATE": 0.07, "MAKER_REBATE_SHARE": 0.20,
        "LIMIT_PRICE": 0.51, "POST_ONLY": False, "CROSS_AFTER_SECONDS": 720,
        "CROSS_LATEST_SECONDS": 1200, "TAKER_MAX": 0.55,
        "TAKER_SLIPPAGE_TICKS": 1, "RETRY_SECONDS": 5,
        "RETRY_BURST_SECONDS": 20, "SHARES": 10, "MAX_ENTRY_PRICE": 0.55,
        "MAX_OPEN_POSITIONS": 3, "DAILY_LOSS_LIMIT_USD": 40.0,
        "MONTHLY_LOSS_LIMIT_USD": 300.0,
    }


def test_defaults_are_valid():
    """Подстраховка: если кто-то поменяет умолчания в config.py мимо этого
    файла, тест напомнит поправить и здесь.
    """
    settings.validate_effective(_defaults())
