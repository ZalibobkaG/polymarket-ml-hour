"""Проверки схемы исполнения: лимитка 0.51 без post-only и добор на 12-й минуте.

    python -m pytest tests

Комплект модели здесь не нужен и не импортируется: `app.model` подменяется
заглушкой до импорта runner, биржа и стакан — управляемыми подделками, время —
симулированными часами. Каждый тест работает со своей временной БД.
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MODEL = types.ModuleType("app.model")
_MODEL.predict = lambda minutes, hour_start: {"p_up": _MODEL.p_up}
_MODEL.card = lambda: {"model_id": "test"}
_MODEL.p_up = 0.60
sys.modules["app.model"] = _MODEL

import config  # noqa: E402
import app  # noqa: E402

app.model = _MODEL
from app import (features, markets, notify, pricing, risk, runner,  # noqa: E402
                 storage, trader, ws)
from app.trader import Leg, OrderResult  # noqa: E402

H = pd.Timestamp("2026-08-01 12:00", tz="UTC")


# --------------------------------------------------------------------------
# pricing.decide
# --------------------------------------------------------------------------

def test_decide_places_limit_at_051_passive_when_ask_above():
    d = pricing.decide(0.60, tick=0.01, best_ask=0.55, shares=10)
    assert d.bet and d.side == "Up" and d.price == pytest.approx(0.51)
    assert d.cross is False


def test_decide_marketable_is_normal_outcome_not_a_reason_to_skip():
    d = pricing.decide(0.60, tick=0.01, best_ask=0.48, shares=10)
    assert d.bet and d.cross is True
    # Решение и его матожидание — по нашей цене, без выбора «тейкер выгоднее».
    assert d.price == pytest.approx(0.51)
    assert d.ev_usd == pytest.approx(pricing.expected_value(0.60, 0.51, "Up", 10))


def test_decide_bets_every_hour_regardless_of_probability():
    # Отказа по матожиданию нет: EV ≤ 0 только пишется в журнал.
    d = pricing.decide(0.505, tick=0.01, best_ask=0.55, shares=10)
    assert d.bet and d.side == "Up" and d.shares == 10
    assert d.ev_usd < 0 and "EV=" in d.reason
    assert pricing.decide(0.52, tick=0.01, best_ask=0.55, shares=10).side == "Up"
    d = pricing.decide(0.30, tick=0.01, best_ask=0.55, shares=10)
    assert d.bet and d.side == "Down"


def test_decide_side_flips_at_one_half_without_threshold():
    up = pricing.decide(0.5005, tick=0.01, best_ask=0.55, shares=10)
    down = pricing.decide(0.4995, tick=0.01, best_ask=0.55, shares=10)
    assert up.bet and up.side == "Up"
    assert down.bet and down.side == "Down"


def test_decide_keeps_max_entry_price_check(monkeypatch):
    monkeypatch.setattr(config, "LIMIT_PRICE", 0.56)
    assert not pricing.decide(0.80, tick=0.01, best_ask=0.60, shares=10).bet


def test_passive_order_expires_5s_before_cross():
    info = _info(H)
    assert trader.passive_expiration(info) == int(
        (H + pd.Timedelta(seconds=config.CROSS_AFTER_SECONDS - 5)).timestamp())


# --------------------------------------------------------------------------
# Подделки биржи
# --------------------------------------------------------------------------

def _info(hs):
    return markets.MarketInfo(hour_start=hs, slug="test-market", condition_id="cond",
                              token_up="UP", token_down="DOWN",
                              end_date=hs + pd.Timedelta(hours=1), accepting_orders=True,
                              min_size=5.0, tick_size=0.01, neg_risk=False,
                              best_bid_up=None, best_ask_up=None)


class FakeExchange:
    """Стакан и заявки, которыми тест управляет руками."""

    live = False

    def __init__(self):
        self.asks = {"UP": 0.55, "DOWN": 0.47}
        self.calls: list[tuple] = []
        self.fills: dict[str, list[Leg]] = {}
        self.cancel_ok: list[bool] = []          # очередь ответов на отмену
        self.reject_post_only = False

    # --- markets
    def book(self, token):
        ask = self.asks.get(token)
        return {"bid": None if ask is None else round(ask - 0.02, 2), "ask": ask}

    def snapshot(self, info):
        up, down = self.book("UP"), self.book("DOWN")
        return {"best_bid_up": up["bid"], "best_ask_up": up["ask"],
                "best_bid_down": down["bid"], "best_ask_down": down["ask"],
                "spread_up": None}

    # --- trader
    def place(self, info, side, price, shares, post_only=False, order_type="GTD"):
        oid = f"{order_type}-{len(self.calls)}"
        self.calls.append(("place", oid, side, round(price, 4), shares, post_only,
                           order_type))
        ask = self.asks.get(info.token(side))
        if post_only and ask is not None and ask <= price + 1e-9 and self.reject_post_only:
            return OrderResult(ok=False, post_only_rejected=True, error="would cross")
        self.fills[oid] = ([Leg(ask, shares, "TAKER", 0.0)]
                           if ask is not None and ask <= price + 1e-9 else [])
        return OrderResult(ok=True, order_id=oid)

    def legs(self, info, order_id):
        return list(self.fills.get(order_id or "", []))

    def cancel_confirmed(self, order_id):
        self.calls.append(("cancel", order_id))
        ok = self.cancel_ok.pop(0) if self.cancel_ok else True
        return (ok, sum(g.shares for g in self.fills.get(order_id, []))) if ok else (False, None)

    def cancel(self, order_id):
        return self.cancel_confirmed(order_id)[0]

    def maker_fill(self, order_id, shares, price=0.51):
        self.fills[order_id].append(Leg(price, shares, "MAKER", 0.0))

    def places(self, order_type=None):
        return [c for c in self.calls if c[0] == "place"
                and (order_type is None or c[6] == order_type)]


class Clock:
    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += pd.Timedelta(seconds=s)

    def monotonic(self):
        return self.t.timestamp()


@pytest.fixture
def env(tmp_path, monkeypatch):
    ex = FakeExchange()
    clock = Clock(H - pd.Timedelta(seconds=113))
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "bot.db")
    monkeypatch.setattr(storage, "_Session", None)
    monkeypatch.setattr(markets, "find", lambda hs, with_book=True: _info(pd.Timestamp(hs)))
    monkeypatch.setattr(markets, "book", ex.book)
    monkeypatch.setattr(markets, "snapshot", ex.snapshot)
    monkeypatch.setattr(markets, "ensure_valid", lambda *a, **k: None)
    monkeypatch.setattr(markets, "outcome_of", lambda hs: None)
    monkeypatch.setattr(markets, "outcome_from_binance", lambda hs: None)
    monkeypatch.setattr(features, "for_market", lambda hs, cutoff=None: pd.DataFrame())
    monkeypatch.setattr(notify, "send", lambda *a, **k: False)
    monkeypatch.setattr(ws, "user_channel", lambda *a, **k: None)
    monkeypatch.setattr(risk, "check", lambda now=None: risk.Verdict(True))
    monkeypatch.setattr(trader, "build", lambda: ex)
    monkeypatch.setattr(runner, "_now", clock.now)
    monkeypatch.setattr(runner, "time", clock)
    _MODEL.p_up = 0.60
    bot = runner.Bot()
    return types.SimpleNamespace(bot=bot, ex=ex, clock=clock)


def _row(hs=H):
    with storage.session() as s:
        return storage.get_market(s, hs.to_pydatetime())


def _decide(env):
    env.bot.handle_market(H)
    return env.ex.places("GTD")[0][1]


def _at_cross(env, extra_seconds=7):
    env.clock.t = H + pd.Timedelta(seconds=config.CROSS_AFTER_SECONDS + extra_seconds)
    env.bot.tick()


# --------------------------------------------------------------------------
# Постановка и добор
# --------------------------------------------------------------------------

def test_limit_goes_without_post_only_and_rests(env):
    oid = _decide(env)
    place = env.ex.places("GTD")[0]
    assert place[3] == pytest.approx(0.51) and place[5] is False    # не post-only
    m = _row()
    assert m.status == storage.ORDERED and m.order_id == oid and m.place_ts is not None


def test_nothing_happens_before_cross_time(env):
    _decide(env)
    env.clock.t = H + pd.Timedelta(seconds=config.CROSS_AFTER_SECONDS - 23)
    env.bot.tick()
    assert not env.ex.places("FAK") and ("cancel",) not in [c[:1] for c in env.ex.calls]
    assert _row().cross_ts is None


def test_cross_cancels_first_then_buys_only_the_remainder_once(env):
    oid = _decide(env)
    env.ex.maker_fill(oid, 4)                 # лимитку налили на 4 из 10
    env.ex.asks["UP"] = 0.53
    _at_cross(env)
    kinds = [c[0] if c[0] == "cancel" else c[6] for c in env.ex.calls]
    assert kinds == ["GTD", "cancel", "FAK"]  # отмена раньше добора
    fak = env.ex.places("FAK")[0]
    assert fak[4] == pytest.approx(6) and fak[3] == pytest.approx(0.54)
    m = _row()
    # Роль — по большинству шейров: 6 добором против 4 мейкером.
    assert m.filled_shares == pytest.approx(10) and m.fill_role == storage.CROSS
    assert m.fill_price == pytest.approx((4 * 0.51 + 6 * 0.53) / 10)
    assert m.cross_ts is not None and m.cross_order_id
    _at_cross(env, extra_seconds=37)          # следующий тик — второго добора нет
    assert len(env.ex.places("FAK")) == 1


def test_cross_role_when_cross_gives_most_shares(env):
    oid = _decide(env)
    env.ex.maker_fill(oid, 2)
    _at_cross(env)
    m = _row()
    assert m.fill_role == storage.CROSS and m.filled_shares == pytest.approx(10)
    assert m.fee_charged_usd == pytest.approx(0.0)
    assert m.fee_usd == pytest.approx(8 * pricing.fee_per_share(0.55))


def test_cross_limit_never_above_taker_max(env):
    _decide(env)
    env.ex.asks["UP"] = 0.55                  # аск ровно на потолке: +1 тик срезается
    _at_cross(env)
    assert env.ex.places("FAK")[0][3] == pytest.approx(config.TAKER_MAX)


def test_no_cross_when_ask_above_ceiling(env):
    _decide(env)
    env.ex.asks["UP"] = 0.60
    _at_cross(env)
    assert not env.ex.places("FAK")
    m = _row()
    assert "выше потолка" in m.cross_note and m.status == storage.UNFILLED
    with storage.session() as s:
        tags = [q.tag for q in s.query(storage.Quote).all()]
    assert "cross" in tags                    # снимок перед добором записан


def test_immediate_fill_needs_no_cross(env):
    env.ex.asks["UP"] = 0.50                  # маркетабельно при постановке
    _decide(env)
    m = _row()
    assert m.status == storage.FILLED and m.fill_role == storage.IMMEDIATE_TAKER
    assert m.fill_price == pytest.approx(0.50)
    _at_cross(env)
    assert not env.ex.places("FAK") and "не нужен" in _row().cross_note


def test_cross_waits_for_cancel_confirmation(env):
    _decide(env)
    env.ex.cancel_ok = [False, True]
    _at_cross(env)
    assert not env.ex.places("FAK") and _row().cross_ts is None
    _at_cross(env, extra_seconds=37)
    assert len(env.ex.places("FAK")) == 1


def test_cross_size_comes_from_the_market_row_not_from_shares_command(env):
    oid = _decide(env)
    env.ex.maker_fill(oid, 3)
    with storage.session() as s:
        storage.set_shares(s, 20)             # /shares посреди часа
    _at_cross(env)
    assert env.ex.places("FAK")[0][4] == pytest.approx(7)


def test_pause_cancels_but_does_not_cross(env):
    _decide(env)
    env.bot.paused = True
    _at_cross(env)
    assert ("cancel", env.ex.places("GTD")[0][1]) in env.ex.calls
    assert not env.ex.places("FAK") and "пауза" in _row().cross_note


def test_too_late_no_cross(env):
    _decide(env)
    env.clock.t = H + pd.Timedelta(seconds=config.CROSS_LATEST_SECONDS + 1)
    env.bot.tick()
    assert not env.ex.places("FAK") and "опоздали" in _row().cross_note


def test_remainder_below_exchange_minimum_is_not_crossed(env):
    oid = _decide(env)
    env.ex.maker_fill(oid, 7)                 # остаток 3 < orderMinSize 5
    _at_cross(env)
    assert not env.ex.places("FAK") and "меньше минимума" in _row().cross_note
    assert _row().status == storage.FILLED


def test_pending_row_is_crossed_in_full(env, monkeypatch):
    monkeypatch.setattr(config, "POST_ONLY", True)
    env.ex.reject_post_only = True
    env.ex.asks["UP"] = 0.50                  # встречка держится — лимитка не встаёт
    env.clock.t = H - pd.Timedelta(seconds=30)
    env.bot.handle_market(H)
    assert _row().status == storage.PENDING
    env.ex.asks["UP"] = 0.52
    _at_cross(env)
    fak = env.ex.places("FAK")
    assert len(fak) == 1 and fak[0][4] == pytest.approx(10)
    assert _row().status == storage.FILLED and _row().fill_role == storage.CROSS


# --------------------------------------------------------------------------
# Журнал
# --------------------------------------------------------------------------

def test_migration_keeps_old_rows_and_stats_work(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE markets (id INTEGER PRIMARY KEY, hour_start DATETIME "
                "NOT NULL UNIQUE, status VARCHAR(16) NOT NULL, side VARCHAR(4), "
                "price FLOAT, shares FLOAT, filled_shares FLOAT, fill_price FLOAT, "
                "fill_role VARCHAR(8), fee_usd FLOAT, won BOOLEAN, pnl_usd FLOAT, "
                "resolved_at DATETIME, dry_run BOOLEAN NOT NULL)")
    con.execute("INSERT INTO markets (hour_start, status, side, price, shares, "
                "filled_shares, fill_price, fill_role, fee_usd, won, pnl_usd, "
                "resolved_at, dry_run) VALUES ('2026-09-01 10:00:00', 'resolved', "
                "'Up', 0.49, 5, 5, 0.49, 'MAKER', 0, 1, 2.55, "
                "'2026-09-01 11:01:00', 1)")
    con.commit()
    con.close()
    monkeypatch.setattr(storage, "_Session", None)
    storage.init(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(markets)")}
    assert {"place_ts", "cross_ts", "fee_charged_usd", "cross_order_id",
            "cross_note"} <= cols
    with storage.session() as s:
        st = storage.stats(s)
    assert st["ставок"] == 1 and st["мейкер, %"] == pytest.approx(100)
    assert st["средняя цена входа"] == pytest.approx(0.49)
    assert st["комиссия неизвестна, ставок"] == 1      # до новой схемы не писалась
