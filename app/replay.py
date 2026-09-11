"""Реплей схемы исполнения по записанному стакану и ленте сделок.

    python main.py replay                       новая схема, исполнение по очереди
    python main.py replay --fill touch          то же, но живой PaperTrader («по касанию»)
    python main.py replay --scheme old          старая схема (0.49, post-only, без добора)

Зачем. Бумажный режим в реальном времени копит 200 часов больше недели, а стакан
и лента за 27.07–06.09.2026 уже записаны монитором (`data/_book_reduced/` корневого
проекта). Здесь по этим часам гоняются настоящие `Bot.tick`, `pricing.decide`,
постановка лимитки, добор, отмена и журнал в БД — с симулированными часами
(тик 30 с, как в `main.py run`).

Что подменяется — только внешний мир:

* `markets` — стакан и рынок из записи монитора (записан только Up; стакан Down —
  зеркало: бид Down по p = аск Up по 1−p и наоборот); исход — из resolved.csv;
* `model.predict` — прогнозы скользящей проверки (`--pred`), а не комплект: сам
  комплект в реплее не нужен, а на Windows его контрольная сумма не сходится из-за
  переводов строк при checkout;
* `notify` — глушится целиком, в Telegram не уходит ничего; `ws` — выключен;
* риск-лимиты — выключены: схема исполнения меряется без остановок по P&L;
* БД — отдельный файл (`--db`), боевая `data/bot.db` не трогается.

Модель исполнения по умолчанию (`--fill queue`) — как в замере. Маркетабельная
часть заявки сразу проходит по уровням аска не дороже цены. Остаток встаёт в
очередь за объёмом, который уже стоит на нашем уровне цены. Очередь съедают
встречные сделки из ленты: продажа нашего токена по нашей цене или покупка
парного токена по 1 − наша цена. Сделка за нашим уровнем (продажа дешевле,
покупка парного дороже) значит, что уровень пройден насквозь — исполнены целиком.
Исполненный из очереди кусок — мейкер по нашей цене. Добор (FAK) проходит по
уровням аска не дороже своего лимита, остаток снимается.

Нужен DRY_RUN=True: реплей отказывается работать с боевым конфигом.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

ROOT = Path(__file__).resolve().parents[1]
#: Бот лежит в <корень исследования>/poly/polymarket-ml-hour — данные монитора
#: и прогнозы берутся из корня исследования. На сервере их нет, и это нормально:
#: реплей — инструмент проверки, а не часть боевого пути.
_RESEARCH = ROOT.parents[1]
DEFAULT_DATA = _RESEARCH / "data" / "_book_reduced"
DEFAULT_PRED = _RESEARCH / "research" / "poly_hourly_xcoin" / "results" / "predictions_btc_base.csv"
DEFAULT_DB = ROOT / "data" / "replay.db"

TICK_SECONDS = 30
#: Сдвиг тиков относительно часа: решение на H−113 с, добор на H+727 с — внутри
#: окон, не на самой границе.
TICK_PHASE_SECONDS = 7
_EPS = 1e-6

_BOOK_COLS = ["ts", "slug", "t", "btc", "d", "sig", "bid", "bid_sz", "ask", "ask_sz",
              "bids", "asks"]
_TRADE_COLS = ["ts", "slug", "outcome", "price", "size", "side", "fee", "t", "T",
               "btc", "d", "z", "sig"]


class ReplayError(RuntimeError):
    """Понятная человеку причина, почему реплей не запустился."""


# --------------------------------------------------------------------------
# Часы
# --------------------------------------------------------------------------

class Clock:
    """Симулированное время. Подставляется вместо `runner._now` и модуля `time`
    в runner: `sleep` двигает часы, `monotonic` — те же часы в секундах."""

    def __init__(self, t: pd.Timestamp) -> None:
        self.t = t

    def now(self) -> pd.Timestamp:
        return self.t

    def ts(self) -> float:
        return self.t.timestamp()

    def ms(self) -> int:
        return int(round(self.t.timestamp() * 1000))

    def advance(self, seconds: float) -> None:
        self.t += pd.Timedelta(seconds=seconds)

    # интерфейс модуля time для runner
    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def monotonic(self) -> float:
        return self.ts()


# --------------------------------------------------------------------------
# Запись монитора
# --------------------------------------------------------------------------

def _levels(s) -> list[tuple[float, float]]:
    """'0.51:120|0.50:300' → [(0.51, 120.0), (0.50, 300.0)]; пусто — []."""
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split("|"):
        p, _, sz = part.partition(":")
        out.append((float(p), float(sz)))
    return out


@dataclass
class _Market:
    slug: str
    hour: pd.Timestamp
    winner: str
    book_ts: np.ndarray
    bids: list
    asks: list
    tr_ts: np.ndarray
    tr_outcome: np.ndarray
    tr_price: np.ndarray
    tr_size: np.ndarray
    tr_side: np.ndarray


class Tape:
    """Стакан, лента и исходы по часам, с поиском «последний снимок не позже t»."""

    def __init__(self, data_dir: Path) -> None:
        files = {n: data_dir / n for n in ("book5s.csv", "trades.csv", "resolved.csv")}
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise ReplayError("нет файлов записи монитора: " + ", ".join(missing)
                              + ". Укажите каталог через --data.")
        book = pd.read_csv(files["book5s.csv"], header=None, names=_BOOK_COLS,
                           usecols=["ts", "slug", "bids", "asks"])
        trades = pd.read_csv(files["trades.csv"], header=None, names=_TRADE_COLS,
                             usecols=["ts", "slug", "outcome", "price", "size", "side",
                                      "fee"])
        resolved = pd.read_csv(files["resolved.csv"])
        resolved = resolved[resolved.slug.str.startswith("bitcoin")]
        #: Поле комиссии в публичной ленте сделок Polymarket не заполняется —
        #: нулей здесь ожидаемо 100%. Это НЕ проверка того, берёт ли биржа
        #: комиссию по факту (она берёт, см. блок комиссий в config.py), только
        #: диагностика самой записи.
        self.trades_total = len(trades)
        self.trades_fee_nonzero = int((trades.fee != 0).sum())

        book = book.sort_values(["slug", "ts"])
        trades = trades.sort_values(["slug", "ts"])
        bgroups = dict(tuple(book.groupby("slug")))
        tgroups = dict(tuple(trades.groupby("slug")))
        self.by_hour: dict[pd.Timestamp, _Market] = {}
        for r in resolved.itertuples():
            b = bgroups.get(r.slug)
            if b is None or r.winner not in ("Up", "Down"):
                continue
            t = tgroups.get(r.slug)
            if t is None:
                t = trades.iloc[0:0]
            hour = pd.Timestamp(r.window_start).tz_convert("UTC")
            self.by_hour[hour] = _Market(
                slug=r.slug, hour=hour, winner=r.winner,
                book_ts=b.ts.to_numpy(), bids=b.bids.tolist(), asks=b.asks.tolist(),
                tr_ts=t.ts.to_numpy(), tr_outcome=t.outcome.to_numpy(),
                tr_price=t.price.to_numpy(dtype=float),
                tr_size=t["size"].to_numpy(dtype=float), tr_side=t.side.to_numpy())
        self.by_slug = {m.slug: m for m in self.by_hour.values()}

    def book_at(self, slug: str, ms: int, outcome: str
                ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """(биды, аски) токена `outcome` по последнему снимку не позже `ms`.

        Биды — от лучшего к худшему, аски — от лучшего (дешёвого) к худшему.
        Down получается зеркалом Up: у записи монитора есть только стакан Up.
        """
        m = self.by_slug.get(slug)
        if m is None:
            return [], []
        i = int(np.searchsorted(m.book_ts, ms, side="right")) - 1
        if i < 0:
            return [], []
        up_bids, up_asks = _levels(m.bids[i]), _levels(m.asks[i])
        if outcome == "Up":
            return up_bids, up_asks
        return ([(round(1 - p, 6), s) for p, s in up_asks],
                [(round(1 - p, 6), s) for p, s in up_bids])


# --------------------------------------------------------------------------
# Исполнитель с очередью
# --------------------------------------------------------------------------

@dataclass
class _Order:
    oid: str
    hour: pd.Timestamp
    slug: str
    outcome: str
    price: float
    shares: float
    kind: str                      # "limit" / "cross"
    expires_ms: int                # после этого момента из очереди не исполняется
    queue_ahead: float
    cursor: int                    # первая ещё не просмотренная сделка ленты
    immediate: float = 0.0         # сколько прошло сразу при постановке
    legs: list = field(default_factory=list)
    opposing: float = 0.0          # встречный объём, прошедший по нашему уровню
    dead: bool = False

    @property
    def filled(self) -> float:
        return sum(g.shares for g in self.legs)


class QueueTrader:
    """Исполнение по очереди — см. docstring модуля. Интерфейс как у PaperTrader."""

    live = False

    def __init__(self, tape: Tape, clock: Clock, sell_only: bool = False) -> None:
        self.tape, self.clock = tape, clock
        #: Строже: очередь съедают только продажи нашего токена, покупки парного
        #: не считаются. Проверка чувствительности, не основной режим.
        self.sell_only = sell_only
        self.orders: dict[str, _Order] = {}
        #: Журнал для проверок инвариантов: (час, вид, время, цена, шейры, роль).
        self.fills: list[tuple] = []
        self.events: list[tuple] = []          # (час, событие, время)

    def place(self, info, side: str, price: float, shares: float,
              post_only: bool = False, order_type: str = "GTD"):
        from app.trader import Leg, OrderResult, passive_expiration

        now = self.clock.ms()
        kind = "cross" if order_type == "FAK" else "limit"
        bids, asks = self.tape.book_at(info.slug, now, side)
        immediate, rest = [], shares
        for p, sz in asks:
            if p > price + _EPS or rest <= _EPS:
                break
            take = min(sz, rest)
            immediate.append(Leg(p, take, "TAKER", 0.0))
            rest -= take
        if post_only and immediate:
            return OrderResult(ok=False, post_only_rejected=True,
                               error="replay: would cross")

        if not bids or price > bids[0][0] + _EPS or immediate:
            queue = 0.0                  # встаём лучшим бидом — впереди никого
        elif price < bids[-1][0] - _EPS:
            queue = math.inf             # глубже видимых пяти уровней
        else:
            queue = sum(s for p, s in bids if abs(p - price) < _EPS)

        m = self.tape.by_slug[info.slug]
        oid = f"replay-{info.hour_start:%Y%m%d%H}-{side.lower()}-{kind}"
        o = _Order(oid=oid, hour=pd.Timestamp(info.hour_start), slug=info.slug,
                   outcome=side, price=price, shares=shares, kind=kind,
                   expires_ms=(passive_expiration(info) * 1000
                               if order_type == "GTD" else now),
                   queue_ahead=queue,
                   cursor=int(np.searchsorted(m.tr_ts, now, side="right")),
                   immediate=shares - rest, legs=list(immediate),
                   dead=order_type != "GTD")
        self.orders[oid] = o
        self.events.append((o.hour, f"place_{kind}", now))
        for g in immediate:
            self.fills.append((o.hour, kind, now, g.price, g.shares, g.role, price))
        return OrderResult(ok=True, order_id=oid)

    def _advance(self, o: _Order) -> None:
        """Прогнать ленту от прошлого просмотра до «сейчас» (но не дальше срока)."""
        from app.trader import Leg

        if o.dead:
            return
        m = self.tape.by_slug[o.slug]
        end = min(self.clock.ms(), o.expires_ms)
        passive_cap = o.shares - o.immediate
        while o.cursor < len(m.tr_ts) and m.tr_ts[o.cursor] <= end:
            i = o.cursor
            o.cursor += 1
            maker_done = o.filled - o.immediate
            if maker_done >= passive_cap - _EPS:
                break
            q, out, side = m.tr_price[i], m.tr_outcome[i], m.tr_side[i]
            if out == o.outcome and side == "SELL":
                lvl = q                  # продавец нашего токена бьёт в биды
            elif out != o.outcome and side == "BUY" and not self.sell_only:
                lvl = 1 - q              # покупатель парного — это наш бид по 1−q
            else:
                continue
            if lvl < o.price - _EPS:     # уровень пройден насквозь
                target = passive_cap
            elif abs(lvl - o.price) < _EPS:
                o.opposing += m.tr_size[i]
                target = min(passive_cap, max(0.0, o.opposing - o.queue_ahead))
            else:
                continue
            new = target - maker_done
            if new > _EPS:
                o.legs.append(Leg(o.price, new, "MAKER", 0.0))
                self.fills.append((o.hour, o.kind, int(m.tr_ts[i]), o.price, new,
                                   "MAKER", o.price))

    def legs(self, info, order_id):
        o = self.orders.get(order_id or "")
        if o is None:
            return []
        self._advance(o)
        return list(o.legs)

    def cancel_confirmed(self, order_id: str):
        o = self.orders.get(order_id)
        if o is None:
            return True, None
        self._advance(o)
        o.dead = True
        self.events.append((o.hour, "cancel", self.clock.ms()))
        return True, o.filled

    def cancel(self, order_id: str) -> bool:
        return self.cancel_confirmed(order_id)[0]

    def open_orders(self) -> list[dict]:
        return []


# --------------------------------------------------------------------------
# Подмена внешнего мира
# --------------------------------------------------------------------------

def _install_fake_model(pred: dict) -> None:
    """Подложить `app.model` до импорта runner: прогноз — из таблицы проверки."""
    fake = types.ModuleType("app.model")
    fake.__doc__ = "Подмена для реплея: прогнозы скользящей проверки."

    def predict(minutes, hour_start):
        return {"p_up": float(pred[pd.Timestamp(hour_start)])}

    fake.predict = predict
    fake.card = lambda: {"model_id": "replay-walkforward"}
    fake.reload_if_changed = lambda: False
    sys.modules["app.model"] = fake
    import app
    app.model = fake


def _patch_world(tape: Tape, clock: Clock, hours: set, fill: str,
                 sell_only: bool = False) -> "QueueTrader | None":
    from app import features, markets, notify, risk, runner, trader, ws

    def find(hour_start, with_book: bool = True):
        hs = pd.Timestamp(hour_start)
        m = tape.by_hour.get(hs)
        if m is None or hs not in hours:
            raise markets.MarketError(f"реплей: нет записи монитора на {hs}")
        info = markets.MarketInfo(
            hour_start=hs, slug=m.slug, condition_id=m.slug,
            token_up=f"{m.slug}|Up", token_down=f"{m.slug}|Down",
            end_date=hs + pd.Timedelta(hours=1), accepting_orders=True,
            min_size=5.0, tick_size=0.01, neg_risk=False,
            best_bid_up=None, best_ask_up=None)
        if with_book:
            b = book(info.token_up)
            info.best_bid_up, info.best_ask_up = b["bid"], b["ask"]
        return info

    def book(token_id: str) -> dict:
        slug, _, outcome = token_id.partition("|")
        bids, asks = tape.book_at(slug, clock.ms(), outcome)
        return {"bid": bids[0][0] if bids else None, "ask": asks[0][0] if asks else None}

    def snapshot(info) -> dict:
        up, down = book(info.token_up), book(info.token_down)
        spread = (up["ask"] - up["bid"]) if (up["ask"] and up["bid"]) else None
        return {"best_bid_up": up["bid"], "best_ask_up": up["ask"],
                "best_bid_down": down["bid"], "best_ask_down": down["ask"],
                "spread_up": spread}

    def outcome(hour_start):
        hs = pd.Timestamp(hour_start)
        m = tape.by_hour.get(hs)
        if m is None or clock.now() < hs + pd.Timedelta(hours=1):
            return None
        return m.winner

    markets.find, markets.book, markets.snapshot = find, book, snapshot
    markets.ensure_valid = lambda *a, **k: None
    markets.outcome_of = outcome
    markets.outcome_from_binance = outcome
    trader.book = book                       # PaperTrader читает стакан через свой импорт
    trader._now_ts = clock.ts
    features.for_market = lambda hour_start, cutoff=None: pd.DataFrame()
    notify.send = lambda *a, **k: False
    notify.send_photo = lambda *a, **k: False
    notify.send_document = lambda *a, **k: False
    ws.user_channel = lambda *a, **k: None
    risk.check = lambda now=None: risk.Verdict(True)
    runner._now = clock.now
    runner.time = clock

    if fill == "queue":
        qt = QueueTrader(tape, clock, sell_only=sell_only)
        trader.build = lambda: qt
        return qt
    trader.build = lambda: trader.PaperTrader()
    return None


def _apply_scheme(scheme: str) -> None:
    """--scheme old: прежние параметры, чтобы сверить реплей с замером (+0.0143).

    Добор выключается потолком TAKER_MAX = 0 (любой аск выше); лимитка живёт
    почти до конца часа, как раньше (GTD до end_date − 30 с).
    """
    if scheme == "old":
        config.LIMIT_PRICE = 0.49
        config.POST_ONLY = True
        config.CROSS_AFTER_SECONDS = 3565 + 5
        config.CROSS_LATEST_SECONDS = 3590
        config.TAKER_MAX = 0.0


# --------------------------------------------------------------------------
# Прогон и сводка
# --------------------------------------------------------------------------

def _load_pred(path: Path) -> dict:
    if not path.exists():
        raise ReplayError(f"нет файла прогнозов {path}. Укажите его через --pred.")
    p = pd.read_csv(path)
    if "target_hour" in p.columns:
        hours = pd.to_datetime(p.target_hour, utc=True)
    elif "open_time" in p.columns:
        # open_time — начало последней свечи признаков; прогноз — на следующий час.
        hours = pd.to_datetime(p.open_time, utc=True) + pd.Timedelta(hours=1)
    else:
        raise ReplayError(f"в {path.name} нет колонки target_hour или open_time")
    if "proba" not in p.columns:
        raise ReplayError(f"в {path.name} нет колонки proba (вероятность Up)")
    return dict(zip(hours, p.proba.astype(float)))


def _simulate(bot, clock: Clock, hours: list[pd.Timestamp]) -> int:
    """Тикать, как `main.py run`, перепрыгивая пустые промежутки между часами."""
    from app import storage

    end = hours[-1] + pd.Timedelta(hours=1, seconds=150)
    lead = pd.Timedelta(seconds=150 - TICK_PHASE_SECONDS)
    ticks, k = 0, 0
    while clock.now() <= end:
        while k < len(hours) and hours[k] <= clock.now() - pd.Timedelta(seconds=60):
            k += 1
        if k < len(hours) and clock.now() < hours[k] - lead:
            with storage.session() as s:
                busy = s.query(storage.Market).filter(storage.Market.status.in_(
                    [storage.ORDERED, storage.FILLED, storage.PENDING,
                     storage.UNFILLED, storage.NEW])).count()
            if not busy:
                clock.t = hours[k] - lead      # в промежутке нечего делать
        bot.tick()
        ticks += 1
        clock.advance(TICK_SECONDS)
    return ticks


def _summary(hours: list[pd.Timestamp], tape: Tape, qt: "QueueTrader | None",
             pred: dict) -> dict:
    from app import storage

    with storage.session() as s:
        rows = {storage.as_utc(m.hour_start): m for m in s.query(storage.Market).all()}
    ms = [rows.get(h.to_pydatetime()) for h in hours]
    n = len(hours)
    bets = [m for m in ms if m is not None and m.side]
    filled = [m for m in bets if m.filled_shares]
    roles = [storage.role_of(m) for m in bets]
    shares_total = sum(m.filled_shares for m in filled)
    skipped = [m for m in ms if m is not None and m.status == storage.SKIPPED]
    errors = [m for m in ms if m is None or m.status == storage.ERROR]

    # На шейр ставки в час — как в замере: сумма P&L / (размер ставки × все часы).
    # `pnl_usd` уже уменьшен на комиссию (`_sync_fill` списывает её с тейкерных
    # кусков) — вариант без комиссии получаем, прибавив её обратно.
    size = float(config.SHARES)
    pnl_fee = np.array([(m.pnl_usd or 0.0) if m is not None else 0.0 for m in ms])
    fee = np.array([(m.fee_usd or 0.0) if m is not None and m.filled_shares else 0.0
                    for m in ms])
    per_share_fee = pnl_fee / size               # с комиссией по формуле — основной вариант
    per_share_nofee = (pnl_fee + fee) / size      # без неё — справочно, комиссия реальна

    # Диагностика для разбора «почему прирост с комиссией меньше»: доля тейкера
    # в исполнении и проверка, что комиссия не залезает на мейкерные куски —
    # у роли «мейкер» она обязана быть ~0 (см. tests/test_execution.py
    # test_cross_role_when_cross_gives_most_shares — там же проверено на уровне
    # кода, что комиссия считается только по тейкерным кускам).
    taker_share_pct = (sum(r in (storage.IMMEDIATE_TAKER, storage.CROSS) for r in roles)
                       / len(bets) * 100) if bets else float("nan")
    maker_fee_usd = sum((m.fee_usd or 0.0) for m, r in zip(bets, roles)
                        if r == storage.MAKER)

    def mean_ci(x):
        se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
        return float(x.mean()), float(1.96 * se), float(x.mean() / se) if se else float("nan")

    winners = {h: tape.by_hour[h].winner for h in hours}
    model_hit = [(pred[h] >= 0.5) == (winners[h] == "Up") for h in hours]
    bet_hit = [m.side == winners[pd.Timestamp(storage.as_utc(m.hour_start))]
               for m in bets]
    fill_hit = [bool(m.won) for m in filled]

    notes: dict[str, int] = {}
    for m in bets:
        note = (m.cross_note or "—")
        key = ("добор отправлен" if note.startswith("добор ") and "не нужен" not in note
               else note.split(":")[0].split(" — ")[0].split(" по ")[0])
        if note.startswith("аск ") and "выше потолка" in note:
            key = "аск выше потолка"
        if note.startswith("остаток "):
            key = "остаток меньше минимума биржи"
        notes[key] = notes.get(key, 0) + 1

    out = {
        "часов": n,
        "ставку решили сделать": len(bets),
        "пропуск по матожиданию": sum(1 for m in skipped
                                      if (m.reason or "").startswith("матожидание")),
        "прочие пропуски": sum(1 for m in skipped
                               if not (m.reason or "").startswith("матожидание")),
        "ошибки": len(errors),
        "с исполнением": len(filled),
        "без ставки, % всех часов": (n - len(filled)) / n * 100,
        "без исполнения, % решённых": (sum(r == storage.NO_FILL for r in roles)
                                       / len(bets) * 100) if bets else float("nan"),
        "мейкер, %": sum(r == storage.MAKER for r in roles) / len(bets) * 100,
        "сразу тейкер, %": sum(r == storage.IMMEDIATE_TAKER for r in roles) / len(bets) * 100,
        "добор, %": sum(r == storage.CROSS for r in roles) / len(bets) * 100,
        "тейкер всего (сразу + добор), %": taker_share_pct,
        "без исполнения, %": sum(r == storage.NO_FILL for r in roles) / len(bets) * 100,
        "средняя цена входа": (sum((m.fill_price or m.price) * m.filled_shares
                                   for m in filled) / shares_total) if shares_total else None,
        "набрано шейров в среднем": shares_total / len(filled) if filled else None,
        "точность модели на всех часах, %": float(np.mean(model_hit) * 100),
        "точность на решённых ставках, %": float(np.mean(bet_hit) * 100) if bet_hit else None,
        "точность на исполненных, %": float(np.mean(fill_hit) * 100) if fill_hit else None,
        "на шейр в час с комиссией по формуле": mean_ci(per_share_fee),
        "на шейр в час без комиссии (справочно)": mean_ci(per_share_nofee),
        "комиссия на часах с ролью «мейкер», $": maker_fee_usd,
        "итог добора": notes,
        "сделок в ленте": tape.trades_total,
        "сделок с ненулевой комиссией": tape.trades_fee_nonzero,
    }

    # Средняя цена по ролям — по кускам, из журнала исполнителя.
    if qt is not None:
        by_kind: dict[str, list] = {}
        for hour, kind, ts, price, sh, role, limit in qt.fills:
            key = "добор" if kind == "cross" else ("мейкер" if role == "MAKER" else "сразу тейкер")
            by_kind.setdefault(key, []).append((price, sh))
        out["средняя цена по видам кусков"] = {
            k: round(sum(p * s for p, s in v) / sum(s for _, s in v), 4)
            for k, v in by_kind.items()}
        out["инварианты (нарушений)"] = _invariants(qt, ms, hours)
    return out


def _invariants(qt: QueueTrader, ms: list, hours: list[pd.Timestamp]) -> dict:
    """То, что обязано быть нулём при исправной отмене и потолке."""
    cross_s = config.CROSS_AFTER_SECONDS
    late_maker = sum(1 for hour, kind, ts, *_ in qt.fills
                     if kind == "limit"
                     and ts > (hour + pd.Timedelta(seconds=cross_s - 5)).timestamp() * 1000)
    over_cap = sum(1 for _, kind, _, price, _, _, limit in qt.fills
                   if kind == "cross" and (price > config.TAKER_MAX + _EPS
                                           or limit > config.TAKER_MAX + _EPS))
    crosses: dict = {}
    cross_before_cancel = 0
    cancelled: set = set()
    limit_hours = {o.hour for o in qt.orders.values() if o.kind == "limit"}
    for hour, ev, ts in qt.events:
        if ev == "cancel":
            cancelled.add(hour)
        if ev == "place_cross":
            crosses[hour] = crosses.get(hour, 0) + 1
            if hour in limit_hours and hour not in cancelled:
                cross_before_cancel += 1
    overfill = sum(1 for m in ms if m is not None and m.shares
                   and (m.filled_shares or 0) > m.shares + _EPS)
    return {
        "исполнение лимитки после H+11:55": late_maker,
        "цена или лимит добора выше TAKER_MAX": over_cap,
        "больше одного добора на час": sum(1 for v in crosses.values() if v > 1),
        "добор до отмены лимитки": cross_before_cancel,
        "набрано больше размера ставки": overfill,
    }


def _print(out: dict, label: str) -> None:
    print(f"\n=== реплей: {label} ===")
    for k, v in out.items():
        if isinstance(v, tuple):
            mean, ci, t = v
            print(f"  {k:42s} {mean:+.4f} ± {ci:.4f}  (t = {t:.2f})")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk:38s} {vv}")
        elif isinstance(v, float):
            print(f"  {k:42s} {v:.4f}" if abs(v) < 2 else f"  {k:42s} {v:.2f}")
        else:
            print(f"  {k:42s} {v}")


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python main.py replay",
                                 description="Реплей схемы исполнения по записи стакана.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--fill", choices=("queue", "touch"), default="queue")
    ap.add_argument("--sell-only", action="store_true",
                    help="очередь съедают только продажи нашего токена (строже)")
    ap.add_argument("--scheme", choices=("new", "old"), default="new")
    ap.add_argument("--hours", type=int, default=0, help="взять только первые N часов")
    ap.add_argument("--json", type=Path, default=None, help="куда записать сводку")
    args = ap.parse_args(argv)

    if not config.DRY_RUN:
        print("Реплей работает только при DRY_RUN = True в config.py.")
        return 2
    try:
        pred = _load_pred(args.pred)
        print(f"читаю запись монитора из {args.data} …")
        tape = Tape(args.data)
    except ReplayError as e:
        print(f"Реплей не запущен: {e}")
        return 2

    hours = sorted(h for h in tape.by_hour if h in pred)
    if args.hours:
        hours = hours[:args.hours]
    if not hours:
        print("Реплей не запущен: у записи монитора и прогнозов нет общих часов.")
        return 2

    _apply_scheme(args.scheme)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()                      # реплей всегда с чистой базы
    config.DB_PATH = args.db

    from loguru import logger
    logger.remove()
    # Час сразу за концом записи бот честно пытается решить и теряет — это не
    # сбой схемы, а край данных; в консоль такие строки не пускаем.
    logger.add(sys.stderr, level="ERROR",
               filter=lambda r: "нет записи монитора" not in r["message"])
    logger.add(args.db.with_suffix(".log"), level="INFO", encoding="utf-8")

    _install_fake_model(pred)
    clock = Clock(hours[0] - pd.Timedelta(seconds=150 - TICK_PHASE_SECONDS))
    qt = _patch_world(tape, clock, set(hours), args.fill, sell_only=args.sell_only)

    from app import storage
    from app.runner import Bot
    storage._Session = None
    bot = Bot()
    label = (f"схема {args.scheme}, исполнение {args.fill}"
             + (" (только продажи)" if args.sell_only else "")
             + f", {len(hours)} часов {hours[0]:%d.%m} – {hours[-1]:%d.%m}")
    print(f"{label}; прогноз: {args.pred.name}")
    ticks = _simulate(bot, clock, hours)
    out = _summary(hours, tape, qt, pred)
    out["тиков"] = ticks
    _print(out, label)
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")
    print(f"\nбаза реплея: {args.db}")
    return 0
