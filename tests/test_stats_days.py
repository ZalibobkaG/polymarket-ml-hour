"""Проверки `python main.py stats --days N`.

    python -m pytest tests/test_stats_days.py

Три слоя: разбор `--days` из аргументов (`main._stats_days_arg`), группировка
причин отказа (`storage._reason_bucket`) и сама сводка по окну времени
(`storage.window_report`, а через неё и `main._cmd_stats_days`) — на настоящей
временной базе, как в `test_execution.py`.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import main  # noqa: E402
from app import storage  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "bot.db")
    monkeypatch.setattr(storage, "_Session", None)
    monkeypatch.setattr(storage, "_engine", None)
    storage.init()


# ==========================================================================
# --days: разбор аргументов
# ==========================================================================

def test_days_arg_absent_is_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "stats"])
    assert main._stats_days_arg() is None


def test_days_arg_parsed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "stats", "--days", "3"])
    assert main._stats_days_arg() == 3


def test_days_arg_missing_value(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "stats", "--days"])
    with pytest.raises(ValueError, match="требует число"):
        main._stats_days_arg()


def test_days_arg_not_an_integer(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "stats", "--days", "abc"])
    with pytest.raises(ValueError, match="не целое число"):
        main._stats_days_arg()


@pytest.mark.parametrize("value", ["0", "-5"])
def test_days_arg_not_positive(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["main.py", "stats", "--days", value])
    with pytest.raises(ValueError, match="положительным"):
        main._stats_days_arg()


# ==========================================================================
# storage._reason_bucket
# ==========================================================================

@pytest.mark.parametrize("text, label", [
    ("стакан пуст — не добираем", "пустой стакан"),
    ("аск 0.60 выше потолка 0.55 — не добираем", "аск выше потолка добора"),
    ("остаток 3 меньше минимума биржи 5 — не добираем", "остаток меньше минимума биржи"),
    ("опоздали: позже 20-й минуты не добираем", "опоздали к добору"),
    ("отмена лимитки не подтвердилась до конца окна добора — не добираем",
     "отмена лимитки не подтвердилась"),
    ("пауза — лимитка снята, не добираем", "бот был на паузе"),
    ("торговля выключена в конфиге — лимитка снята, не добираем",
     "торговля выключена (TRADING_ENABLED)"),
    ("не исполнился до конца часа", "не исполнился до конца часа"),
])
def test_reason_bucket_known_causes(text, label):
    assert storage._reason_bucket(text) == label


def test_reason_bucket_unknown_text_passthrough():
    assert storage._reason_bucket("что-то новое, чего ещё не было") == \
        "что-то новое, чего ещё не было"


def test_reason_bucket_empty_text():
    assert storage._reason_bucket("") == "причина не записана"


# ==========================================================================
# storage.window_report
# ==========================================================================

def _make_market(s, hour_start, **over):
    m = storage.get_or_create(s, hour_start)
    for k, v in over.items():
        setattr(m, k, v)
    return m


def test_window_report_no_rows_is_all_zeros(db):
    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(hours=2)
    with storage.session() as s:
        r = storage.window_report(s, since, until)
    assert r["часов в окне"] == 2
    assert r["решено ставить"] == 0
    assert r["исполнено (хотя бы частично)"] == 0
    assert r["мейкером"] == 0
    assert r["средняя цена входа"] is None
    assert r["отказы"] == {}


def test_window_report_counts_decisions_fills_and_rejections(db):
    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(hours=2)

    with storage.session() as s:
        # исполнилась мейкером — попадает и в "решено", и в "исполнено", и в "мейкером"
        _make_market(s, until - dt.timedelta(minutes=90),
                    status=storage.FILLED, side="Up", price=0.51, shares=10,
                    filled_shares=10, fill_price=0.51, fill_role=storage.MAKER,
                    dry_run=config.DRY_RUN)
        # решили ставить, но не налили — попадает в "решено" и в "отказы"
        _make_market(s, until - dt.timedelta(minutes=30),
                    status=storage.UNFILLED, side="Down", price=0.51, shares=10,
                    filled_shares=0,
                    cross_note="аск 0.60 выше потолка 0.55 — не добираем",
                    reason="не исполнился: аск 0.60 выше потолка 0.55 — не добираем",
                    dry_run=config.DRY_RUN)
        # решение ещё не принято (NEW) — не должна попасть ни в один счётчик
        storage.get_or_create(s, until + dt.timedelta(minutes=10))
        # чужой режим (боевой, пока бот в бумаге) — не должна попасть в счётчики
        _make_market(s, until - dt.timedelta(minutes=45),
                    status=storage.FILLED, side="Up", price=0.51, shares=10,
                    filled_shares=10, fill_price=0.51, fill_role=storage.MAKER,
                    dry_run=not config.DRY_RUN)

    with storage.session() as s:
        r = storage.window_report(s, since, until)

    assert r["часов в окне"] == 2
    assert r["решено ставить"] == 2
    assert r["исполнено (хотя бы частично)"] == 1
    assert r["мейкером"] == 1
    assert r["средняя цена входа"] == pytest.approx(0.51)
    assert r["отказы"] == {"аск выше потолка добора": 1}


def test_window_report_average_price_is_share_weighted(db):
    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(hours=2)

    with storage.session() as s:
        _make_market(s, until - dt.timedelta(minutes=90),
                    status=storage.FILLED, side="Up", price=0.51, shares=10,
                    filled_shares=4, fill_price=0.51, fill_role=storage.MAKER,
                    dry_run=config.DRY_RUN)
        _make_market(s, until - dt.timedelta(minutes=30),
                    status=storage.FILLED, side="Up", price=0.51, shares=10,
                    filled_shares=6, fill_price=0.54, fill_role=storage.CROSS,
                    dry_run=config.DRY_RUN)

    with storage.session() as s:
        r = storage.window_report(s, since, until)

    assert r["средняя цена входа"] == pytest.approx((4 * 0.51 + 6 * 0.54) / 10)
    assert r["мейкером"] == 1
    assert r["исполнено (хотя бы частично)"] == 2


# ==========================================================================
# main._cmd_stats_days — вывод целиком
# ==========================================================================

def test_cmd_stats_days_prints_percentages_and_rejections(db, capsys):
    until = dt.datetime.now(dt.timezone.utc)

    with storage.session() as s:
        _make_market(s, until - dt.timedelta(minutes=90),
                    status=storage.FILLED, side="Up", price=0.51, shares=10,
                    filled_shares=10, fill_price=0.51, fill_role=storage.MAKER,
                    dry_run=config.DRY_RUN)
        _make_market(s, until - dt.timedelta(minutes=30),
                    status=storage.UNFILLED, side="Down", price=0.51, shares=10,
                    filled_shares=0,
                    cross_note="стакан пуст — не добираем",
                    reason="не исполнился: стакан пуст — не добираем",
                    dry_run=config.DRY_RUN)

    code = main._cmd_stats_days(1)
    out = capsys.readouterr().out

    assert code == 0
    assert "решение принято:" in out
    assert "заявка исполнилась:" in out
    assert "доля мейкера:" in out
    assert "средняя цена входа:  0.5100" in out
    assert "пустой стакан" in out
    assert "1" in out


def test_cmd_stats_days_no_data_does_not_crash(db, capsys):
    code = main._cmd_stats_days(1)
    out = capsys.readouterr().out
    assert code == 0
    assert "нет данных" in out
    assert "нет исполненных ставок" in out
    assert "отказов нет" in out


def test_cmd_stats_routes_to_days_report(db, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "stats", "--days", "1"])
    code = main.cmd_stats(main._stats_days_arg())
    out = capsys.readouterr().out
    assert code == 0
    assert "окно: последние 1 суток" in out
