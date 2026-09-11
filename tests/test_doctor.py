"""Проверки `python main.py doctor`.

    python -m pytest tests/test_doctor.py

Каждая функция-проверка (`check_env`, `check_hosts`, ...) тестируется отдельно,
с поддельными `config`/`app.markets`/`app.model`/httpx — без реальной сети и
без настоящего комплекта модели. Настоящие модули (`config`, `app.storage`)
используются только там, где это и есть предмет проверки (settings.toml,
data/bot.db) — так же, как в `test_execution.py` и `test_settings.py`.

Секретность (пункт 1: значения переменных окружения нигде не печатаются)
проверяется отдельно и жёстко: тест обязан упасть, если хоть один символ
секрета просочится в stdout.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import httpx
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
import config  # noqa: E402
from app import doctor  # noqa: E402


def _cfg(**over):
    """Лёгкая подделка config.py: только то, что нужно проверкам doctor."""
    base = dict(
        DRY_RUN=True, TRADING_ENABLED=True,
        CLOB_HOST="https://clob.example", GAMMA_HOST="https://gamma.example",
        COIN_TITLE="Bitcoin", SYMBOL="BTCUSDT",
        KIT_DIR=Path("/does/not/exist"), MODEL_STEM="poly_hourly",
        DB_PATH=Path("/does/not/exist/bot.db"),
        TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID="", TG_ALLOWED_USER_IDS=[],
        SETTINGS_SUMMARY="settings.toml: не найден или пуст, все параметры по умолчанию",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json


# ==========================================================================
# 1. переменные окружения
# ==========================================================================

def test_check_env_missing_dotenv_is_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    for name in doctor.ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    problems = []
    status = doctor.check_env(problems, _cfg())

    assert status == "нет"
    assert any(".env не найден" in p for p in problems)


def test_check_env_filled_paper_mode_no_problem(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text(
        "POLY_PRIVATE_KEY=deadbeefsecret\nTG_TOKEN=123:abcSECRET\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    for name in doctor.ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    problems = []
    status = doctor.check_env(problems, _cfg(DRY_RUN=True))
    out = capsys.readouterr().out

    assert status == "есть"
    assert problems == []
    assert "заполнено" in out and "пусто" in out
    # секреты нигде не печатаются, даже частично
    assert "deadbeefsecret" not in out
    assert "abcSECRET" not in out
    assert "123:abcSECRET" not in out


def test_check_env_live_mode_without_private_key_is_problem(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TG_TOKEN=x\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    for name in doctor.ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    problems = []
    status = doctor.check_env(problems, _cfg(DRY_RUN=False))

    assert status == "нет"
    assert any("боевой режим" in p for p in problems)


def test_check_env_live_mode_with_private_key_ok(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("POLY_PRIVATE_KEY=x\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    for name in doctor.ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    problems = []
    status = doctor.check_env(problems, _cfg(DRY_RUN=False))

    assert status == "есть"
    assert problems == []


# ==========================================================================
# 2. доступность хостов
# ==========================================================================

def test_probe_network_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no route")
    monkeypatch.setattr(httpx, "get", boom)

    p = doctor._probe("X", "https://x.example/")
    assert p.error is not None
    assert p.status_code is None


def test_probe_parses_date_header(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse(200, headers={"date": "Tue, 15 Nov 1994 08:12:31 GMT"})
    monkeypatch.setattr(httpx, "get", fake_get)

    p = doctor._probe("X", "https://x.example/")
    assert p.status_code == 200
    assert p.date is not None
    assert p.date.year == 1994 and p.date.month == 11 and p.date.day == 15


def test_check_hosts_no_config_is_skipped():
    probes = doctor.check_hosts([], None)
    assert probes == {}


def test_check_hosts_no_network_is_not_proverено(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", boom)

    problems = []
    probes = doctor.check_hosts(problems, _cfg())
    assert problems == []
    assert probes["CLOB_HOST"].error is not None
    assert probes["GAMMA_HOST"].error is not None


def test_check_hosts_server_error_is_problem(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(503))
    problems = []
    doctor.check_hosts(problems, _cfg())
    assert any("503" in p for p in problems)


def test_check_hosts_ok(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(200))
    problems = []
    probes = doctor.check_hosts(problems, _cfg())
    assert problems == []
    assert probes["CLOB_HOST"].status_code == 200


# ==========================================================================
# 3. ближайший рынок
# ==========================================================================

def test_check_market_no_config():
    assert doctor.check_market([], None, {}) == "не проверено"


def test_check_market_no_network():
    probes = {"GAMMA_HOST": doctor.Probe(label="GAMMA_HOST", url="x", error="offline")}
    assert doctor.check_market([], _cfg(), probes) == "не проверено"


def test_check_market_found_and_accepting(monkeypatch):
    fake = types.ModuleType("app.markets")

    class MarketError(RuntimeError):
        pass

    info = types.SimpleNamespace(
        slug="bitcoin-up-or-down-test", hour_start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        best_bid_up=0.51, best_ask_up=0.53, tick_size=0.001, accepting_orders=True,
    )
    fake.MarketError = MarketError
    fake.find = lambda h, with_book=True: info
    monkeypatch.setitem(sys.modules, "app.markets", fake)
    monkeypatch.setattr(app, "markets", fake, raising=False)

    probes = {"GAMMA_HOST": doctor.Probe(label="GAMMA_HOST", url="x", status_code=200)}
    problems = []
    status = doctor.check_market(problems, _cfg(), probes)

    assert status == "есть"
    assert problems == []


def test_check_market_not_accepting_orders_is_problem(monkeypatch):
    fake = types.ModuleType("app.markets")

    class MarketError(RuntimeError):
        pass

    info = types.SimpleNamespace(
        slug="bitcoin-up-or-down-test", hour_start=pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        best_bid_up=None, best_ask_up=None, tick_size=0.001, accepting_orders=False,
    )
    fake.MarketError = MarketError
    fake.find = lambda h, with_book=True: info
    monkeypatch.setitem(sys.modules, "app.markets", fake)
    monkeypatch.setattr(app, "markets", fake, raising=False)

    probes = {"GAMMA_HOST": doctor.Probe(label="GAMMA_HOST", url="x", status_code=200)}
    problems = []
    status = doctor.check_market(problems, _cfg(), probes)

    assert status == "нет"
    assert any("не принимает заявки" in p for p in problems)


def test_check_market_not_found_is_problem(monkeypatch):
    fake = types.ModuleType("app.markets")

    class MarketError(RuntimeError):
        pass

    def find(h, with_book=True):
        raise MarketError("рынок bitcoin-up-or-down-test не найден")

    fake.MarketError = MarketError
    fake.find = find
    monkeypatch.setitem(sys.modules, "app.markets", fake)
    monkeypatch.setattr(app, "markets", fake, raising=False)

    probes = {"GAMMA_HOST": doctor.Probe(label="GAMMA_HOST", url="x", status_code=200)}
    problems = []
    status = doctor.check_market(problems, _cfg(), probes)

    assert status == "нет"
    assert any("не найден" in p for p in problems)


# ==========================================================================
# 4. комплект модели
# ==========================================================================

def test_check_model_no_config():
    assert doctor.check_model([], None) == "не проверено"


def test_check_model_kit_dir_missing(tmp_path):
    problems = []
    status = doctor.check_model(problems, _cfg(KIT_DIR=tmp_path / "nope"))
    assert status == "нет"
    assert any("нет каталога комплекта" in p for p in problems)


def test_check_model_missing_files(tmp_path):
    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()
    problems = []
    status = doctor.check_model(problems, _cfg(KIT_DIR=kit_dir))
    assert status == "нет"
    assert any("не хватает файлов" in p for p in problems)


def _make_kit_files(kit_dir: Path, stem: str) -> None:
    kit_dir.mkdir(parents=True, exist_ok=True)
    (kit_dir / "kit").mkdir(exist_ok=True)
    (kit_dir / "model").mkdir(exist_ok=True)
    (kit_dir / "passport.json").write_text("{}", encoding="utf-8")
    (kit_dir / "kit" / "features.py").write_text("# stub", encoding="utf-8")
    (kit_dir / "model" / f"{stem}.txt").write_text("stub", encoding="utf-8")
    (kit_dir / "model" / f"{stem}.json").write_text("{}", encoding="utf-8")


def test_check_model_load_failure(tmp_path, monkeypatch):
    kit_dir = tmp_path / "kit"
    _make_kit_files(kit_dir, "poly_hourly")

    fake_model = types.ModuleType("app.model")

    def card():
        raise RuntimeError("файл модели не совпадает с паспортом")
    fake_model.card = card
    monkeypatch.setitem(sys.modules, "app.model", fake_model)
    monkeypatch.setattr(app, "model", fake_model, raising=False)

    problems = []
    status = doctor.check_model(problems, _cfg(KIT_DIR=kit_dir))

    assert status == "нет"
    assert any("не загрузился" in p for p in problems)


def test_check_model_prediction_out_of_range(tmp_path, monkeypatch):
    kit_dir = tmp_path / "kit"
    _make_kit_files(kit_dir, "poly_hourly")

    fake_model = types.ModuleType("app.model")
    fake_model.card = lambda: {"model_id": "abc123"}
    monkeypatch.setitem(sys.modules, "app.model", fake_model)
    monkeypatch.setattr(app, "model", fake_model, raising=False)
    monkeypatch.setattr(doctor, "_fixture_prediction", lambda kit_dir, model_mod: 1.5)

    problems = []
    status = doctor.check_model(problems, _cfg(KIT_DIR=kit_dir))

    assert status == "нет"
    assert any("вне диапазона" in p for p in problems)


def test_check_model_success(tmp_path, monkeypatch):
    kit_dir = tmp_path / "kit"
    _make_kit_files(kit_dir, "poly_hourly")

    fake_model = types.ModuleType("app.model")
    fake_model.card = lambda: {"model_id": "abc123"}
    monkeypatch.setitem(sys.modules, "app.model", fake_model)
    monkeypatch.setattr(app, "model", fake_model, raising=False)
    monkeypatch.setattr(doctor, "_fixture_prediction", lambda kit_dir, model_mod: 0.42)

    problems = []
    status = doctor.check_model(problems, _cfg(KIT_DIR=kit_dir))

    assert status == "есть"
    assert problems == []


# ==========================================================================
# 5. расхождение часов
# ==========================================================================

def test_check_clock_no_data():
    assert doctor.check_clock([], {}) == "не проверено"


def test_check_clock_within_threshold():
    import datetime as dt
    probes = {"CLOB_HOST": doctor.Probe(label="CLOB_HOST", url="x", status_code=200,
                                        date=dt.datetime.now(dt.timezone.utc))}
    problems = []
    status = doctor.check_clock(problems, probes)
    assert status == "есть"
    assert problems == []


def test_check_clock_exceeds_threshold():
    import datetime as dt
    stale = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30)
    probes = {"CLOB_HOST": doctor.Probe(label="CLOB_HOST", url="x", status_code=200, date=stale)}
    problems = []
    status = doctor.check_clock(problems, probes)
    assert status == "нет"
    assert any("часы машины" in p for p in problems)


# ==========================================================================
# 6. база данных (настоящий app.storage, временный файл)
# ==========================================================================

def test_check_db_creates_writes_migrates(tmp_path, monkeypatch):
    from app import storage
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "bot.db")
    monkeypatch.setattr(storage, "_Session", None)
    monkeypatch.setattr(storage, "_engine", None)

    problems = []
    status = doctor.check_db(problems, config)

    assert status == "есть"
    assert problems == []
    assert (tmp_path / "bot.db").exists()


def test_check_db_failure_is_reported(tmp_path, monkeypatch):
    from app import storage
    blocker = tmp_path / "blocker"
    blocker.write_text("не каталог", encoding="utf-8")
    monkeypatch.setattr(config, "DB_PATH", blocker / "bot.db")
    monkeypatch.setattr(storage, "_Session", None)
    monkeypatch.setattr(storage, "_engine", None)

    problems = []
    status = doctor.check_db(problems, config)

    assert status == "нет"
    assert any("bot.db не работает" in p for p in problems)


def test_check_db_no_config():
    assert doctor.check_db([], None) == "не проверено"


# ==========================================================================
# 7. telegram
# ==========================================================================

def test_check_telegram_not_configured_is_problem():
    problems = []
    status = doctor.check_telegram(problems, _cfg(TELEGRAM_BOT_TOKEN="", TG_ALLOWED_USER_IDS=[]))
    assert status == "нет"
    assert any("не настроен" in p for p in problems)


def test_check_telegram_bad_token(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(
        401, {"ok": False, "description": "Unauthorized"}))
    problems = []
    status = doctor.check_telegram(
        problems, _cfg(TELEGRAM_BOT_TOKEN="secrettoken123", TG_ALLOWED_USER_IDS=["1"]))
    assert status == "нет"
    assert any("токен телеграма не работает" in p for p in problems)


def test_check_telegram_bad_id_format(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse(
        200, {"ok": True, "result": {"username": "test_bot"}}))
    problems = []
    status = doctor.check_telegram(
        problems, _cfg(TELEGRAM_BOT_TOKEN="tok", TG_ALLOWED_USER_IDS=["not-a-number"]))
    assert status == "нет"
    assert any("не похожее на числовой" in p for p in problems)


def test_check_telegram_chat_not_found(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if "getMe" in url:
            return FakeResponse(200, {"ok": True, "result": {"username": "test_bot"}})
        return FakeResponse(400, {"ok": False, "description": "Bad Request: chat not found"})
    monkeypatch.setattr(httpx, "get", fake_get)

    problems = []
    status = doctor.check_telegram(
        problems, _cfg(TELEGRAM_BOT_TOKEN="tok", TG_ALLOWED_USER_IDS=["123"],
                       TELEGRAM_CHAT_ID="123"))
    assert status == "нет"
    assert any("не нажал Start" in p for p in problems)


def test_check_telegram_success(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if "getMe" in url:
            return FakeResponse(200, {"ok": True, "result": {"username": "test_bot"}})
        return FakeResponse(200, {"ok": True, "result": {"id": 123}})
    monkeypatch.setattr(httpx, "get", fake_get)

    problems = []
    status = doctor.check_telegram(
        problems, _cfg(TELEGRAM_BOT_TOKEN="tok", TG_ALLOWED_USER_IDS=["123"],
                       TELEGRAM_CHAT_ID="123"))
    assert status == "есть"
    assert problems == []


def test_check_telegram_network_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", boom)

    problems = []
    status = doctor.check_telegram(
        problems, _cfg(TELEGRAM_BOT_TOKEN="tok", TG_ALLOWED_USER_IDS=["123"]))
    assert status == "не проверено"
    assert problems == []


def test_check_telegram_never_prints_token(monkeypatch, capsys):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse(200, {"ok": True, "result": {"username": "test_bot", "id": 1}})
    monkeypatch.setattr(httpx, "get", fake_get)

    doctor.check_telegram(
        [], _cfg(TELEGRAM_BOT_TOKEN="TOPSECRETTOKEN", TG_ALLOWED_USER_IDS=["123"],
                 TELEGRAM_CHAT_ID="123"))
    out = capsys.readouterr().out
    assert "TOPSECRETTOKEN" not in out


# ==========================================================================
# 8. settings.toml
# ==========================================================================

def test_check_settings_ok():
    problems = []
    status = doctor.check_settings(problems, _cfg(), None)
    assert status == "есть"
    assert problems == []


def test_check_settings_error():
    problems = []
    status = doctor.check_settings(problems, None, "TAKER_MAX: записано 0.1, допустимо 0.30-0.99")
    assert status == "нет"
    assert any("TAKER_MAX" in p for p in problems)


# ==========================================================================
# 9. режим работы (только печать, без problems)
# ==========================================================================

def test_check_mode_prints_both_flags_prominently(capsys):
    doctor.check_mode(_cfg(DRY_RUN=True, TRADING_ENABLED=True))
    out = capsys.readouterr().out
    assert "DRY_RUN = True" in out
    assert "TRADING_ENABLED = True" in out


def test_check_mode_warns_on_live(capsys):
    doctor.check_mode(_cfg(DRY_RUN=False, TRADING_ENABLED=True))
    out = capsys.readouterr().out
    assert "БОЙ" in out


def test_check_mode_no_config_does_not_crash(capsys):
    doctor.check_mode(None)
    out = capsys.readouterr().out
    assert "не проверено" in out


# ==========================================================================
# оркестрация run()
# ==========================================================================

def test_run_returns_0_when_all_pass(monkeypatch):
    monkeypatch.setattr(doctor, "_load_config", lambda: (_cfg(), None))
    monkeypatch.setattr(doctor, "check_env", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_hosts", lambda problems, c: {})
    monkeypatch.setattr(doctor, "check_market", lambda problems, c, p: "есть")
    monkeypatch.setattr(doctor, "check_model", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_clock", lambda problems, p: "есть")
    monkeypatch.setattr(doctor, "check_db", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_telegram", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_settings", lambda problems, c, e: "есть")
    monkeypatch.setattr(doctor, "check_mode", lambda c: None)

    assert doctor.run() == 0


def test_run_returns_1_when_any_check_adds_a_problem(monkeypatch):
    def bad_model(problems, c):
        problems.append("комплект модели не загрузился: boom")
        return "нет"

    monkeypatch.setattr(doctor, "_load_config", lambda: (_cfg(), None))
    monkeypatch.setattr(doctor, "check_env", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_hosts", lambda problems, c: {})
    monkeypatch.setattr(doctor, "check_market", lambda problems, c, p: "есть")
    monkeypatch.setattr(doctor, "check_model", bad_model)
    monkeypatch.setattr(doctor, "check_clock", lambda problems, p: "есть")
    monkeypatch.setattr(doctor, "check_db", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_telegram", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_settings", lambda problems, c, e: "есть")
    monkeypatch.setattr(doctor, "check_mode", lambda c: None)

    assert doctor.run() == 1


def test_run_not_проверено_alone_does_not_fail(monkeypatch, capsys):
    """«Не проверено» (нет сети) — не провал: без единой добавленной проблемы
    код возврата обязан остаться 0, даже если половина пунктов пропущена."""
    monkeypatch.setattr(doctor, "_load_config", lambda: (None, "settings.toml сломан"))
    monkeypatch.setattr(doctor, "check_env", lambda problems, c: "есть")
    monkeypatch.setattr(doctor, "check_hosts", lambda problems, c: {})
    monkeypatch.setattr(doctor, "check_market", lambda problems, c, p: "не проверено")
    monkeypatch.setattr(doctor, "check_model", lambda problems, c: "не проверено")
    monkeypatch.setattr(doctor, "check_clock", lambda problems, p: "не проверено")
    monkeypatch.setattr(doctor, "check_db", lambda problems, c: "не проверено")
    monkeypatch.setattr(doctor, "check_telegram", lambda problems, c: "не проверено")

    def bad_settings(problems, c, e):
        problems.append(f"settings.toml не прошёл проверку: {e}")
        return "нет"
    monkeypatch.setattr(doctor, "check_settings", bad_settings)
    monkeypatch.setattr(doctor, "check_mode", lambda c: None)

    # здесь settings.toml сломан -> это одна настоящая проблема, код возврата 1
    assert doctor.run() == 1
    out = capsys.readouterr().out
    assert "ПРОБЛЕМЫ" in out


# ==========================================================================
# `python main.py doctor` целиком (настоящий подпроцесс)
# ==========================================================================

def test_cmd_doctor_runs_without_traceback():
    import subprocess
    result = subprocess.run(
        [sys.executable, "main.py", "doctor"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    out = result.stdout + result.stderr
    assert "Traceback" not in out
    assert result.returncode in (0, 1)
    for heading in ("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "):
        assert heading in result.stdout


def test_cmd_doctor_broken_settings_toml_reported_gracefully():
    import subprocess
    settings_file = ROOT / "settings.toml"
    assert not settings_file.exists(), (
        "в рабочем каталоге уже есть settings.toml — тест его не трогает, но "
        "не должен запускаться поверх чужого файла"
    )
    settings_file.write_text("TAKER_MAX = 0.10\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "doctor"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        out = result.stdout + result.stderr
        assert "Traceback" not in out
        assert result.returncode == 1
        assert "TAKER_MAX" in result.stdout
        assert "не проверено" in result.stdout  # пункты, зависящие от config
    finally:
        settings_file.unlink(missing_ok=True)
