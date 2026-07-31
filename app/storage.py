"""Хранилище: SQLite через SQLAlchemy.

Одна строка `Market` на каждый часовой рынок — это и журнал решений, и бухгалтерия,
и защита от повторного выставления ордера. Ключ — `hour_start`, момент начала часа
в UTC: он однозначно определяет рынок и не зависит от того, как Polymarket назвал слаг.

Отдельная таблица `Quote` пишет стакан в момент решения. Она нужна не для торговли,
а чтобы ответить на главный незакрытый вопрос исследования: по каким ценам ордера
реально можно было бы исполнить.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String, Text,
                        create_engine, func, select)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

import config
from app.log import get

log = get("storage")

Base = declarative_base()

#: Стадии жизни рынка. Переход только вперёд, что упрощает восстановление после падения.
NEW = "new"                 # строка создана, решение ещё не принято
SKIPPED = "skipped"         # решили не ставить (риск-лимит, NaN, фильтр)
PENDING = "pending"         # решение принято, ордер ещё не удалось поставить пассивно
ORDERED = "ordered"         # ордер отправлен, ждём исполнения
FILLED = "filled"           # исполнен полностью или частично
UNFILLED = "unfilled"       # час кончился, так и не исполнился
RESOLVED = "resolved"       # исход известен, P&L посчитан
ERROR = "error"


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True)
    #: Начало часа рынка (UTC). Уникален — на один час одна ставка.
    hour_start = Column(DateTime(timezone=True), unique=True, nullable=False, index=True)
    status = Column(String(16), default=NEW, nullable=False, index=True)

    # --- рынок
    slug = Column(String(128))
    condition_id = Column(String(80))
    token_up = Column(String(90))
    token_down = Column(String(90))

    # --- решение
    decided_at = Column(DateTime(timezone=True))
    proba_up = Column(Float)
    side = Column(String(4))                 # "Up" / "Down"
    price = Column(Float)
    shares = Column(Float)
    ev_usd = Column(Float)
    reason = Column(Text)                    # почему пропустили или что пошло не так

    # --- ордер
    order_id = Column(String(80), index=True)
    ordered_at = Column(DateTime(timezone=True))
    #: Сколько раз пытались поставить пассивный ордер, прежде чем он лёг в книгу.
    place_attempts = Column(Integer, default=0)
    filled_shares = Column(Float, default=0.0)
    #: Средневзвешенная цена реального исполнения, не наша заявка. Может быть лучше
    #: заявки: лимит 0.50 при предложении 0.47 покупается по 0.47.
    fill_price = Column(Float)
    filled_at = Column(DateTime(timezone=True))
    #: "MAKER" / "TAKER" / "MIXED" — от этого зависит комиссия.
    fill_role = Column(String(8))
    #: Комиссия тейкера в долларах, уже уплаченная (положительное = расход).
    fee_usd = Column(Float, default=0.0)

    # --- исход
    outcome = Column(String(4))              # фактический "Up"/"Down"
    won = Column(Boolean)
    pnl_usd = Column(Float)
    resolved_at = Column(DateTime(timezone=True))

    dry_run = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return (f"<Market {self.hour_start:%Y-%m-%d %H:%M}Z {self.status} "
                f"{self.side or '-'}@{self.price or 0:.3f}>")


class Quote(Base):
    """Снимок стакана. Пишется в момент решения и в конце часа."""

    __tablename__ = "quotes"

    id = Column(Integer, primary_key=True)
    hour_start = Column(DateTime(timezone=True), nullable=False, index=True)
    taken_at = Column(DateTime(timezone=True), nullable=False)
    tag = Column(String(16))                 # "decision" / "close"
    best_bid_up = Column(Float)
    best_ask_up = Column(Float)
    best_bid_down = Column(Float)
    best_ask_down = Column(Float)
    spread_up = Column(Float)


class Event(Base):
    """Журнал заметных событий — ошибки, тревоги, остановки."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    level = Column(String(16))
    kind = Column(String(32))
    message = Column(Text)


_engine = None
_Session: sessionmaker | None = None


def init(path=None) -> None:
    global _engine, _Session
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def _migrate(engine) -> None:
    """Досоздать колонки, появившиеся после того, как БД была создана.

    `create_all` таблицы создаёт, но существующие не меняет, а история сделок
    ценнее удобства — поэтому добавляем недостающие колонки на месте.
    """
    from sqlalchemy import inspect, text

    for table in (Market, Quote, Event):
        have = {c["name"] for c in inspect(engine).get_columns(table.__tablename__)}
        missing = [c for c in table.__table__.columns if c.name not in have]
        if not missing:
            continue
        with engine.begin() as conn:
            for col in missing:
                kind = col.type.compile(engine.dialect)
                conn.execute(text(f"ALTER TABLE {table.__tablename__} "
                                  f"ADD COLUMN {col.name} {kind}"))
        log.info("БД: добавлены колонки %s в %s",
                 ", ".join(c.name for c in missing), table.__tablename__)


@contextmanager
def session() -> Iterator[Session]:
    if _Session is None:
        init()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# --------------------------------------------------------------------------
# Операции
# --------------------------------------------------------------------------

def as_utc(value) -> dt.datetime:
    """Время из SQLite приходит без зоны — возвращаем его в UTC.

    Без этого любое сравнение прочитанного времени с `now()` падает на
    «can't compare offset-naive and offset-aware datetimes».
    """
    v = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(str(value))
    return v.replace(tzinfo=dt.timezone.utc) if v.tzinfo is None else v.astimezone(dt.timezone.utc)


def get_market(s: Session, hour_start: dt.datetime) -> Market | None:
    return s.scalar(select(Market).where(Market.hour_start == hour_start))


def get_or_create(s: Session, hour_start: dt.datetime) -> Market:
    """Идемпотентность: на один час — одна строка, повторный вызов вернёт её же."""
    m = get_market(s, hour_start)
    if m is None:
        m = Market(hour_start=hour_start, status=NEW, dry_run=config.DRY_RUN)
        s.add(m)
        s.flush()
    return m


def open_positions(s: Session) -> list[Market]:
    """Рынки, по которым мы уже в игре, но исход ещё не записан."""
    return list(s.scalars(select(Market).where(Market.status.in_([ORDERED, FILLED]))))


def pending_resolution(s: Session, before: dt.datetime) -> list[Market]:
    """Что пора резолвить: час кончился, а исход не записан."""
    return list(s.scalars(
        select(Market)
        .where(Market.status.in_([ORDERED, FILLED, UNFILLED]))
        .where(Market.hour_start < before)
        .order_by(Market.hour_start)))


def add_quote(s: Session, hour_start: dt.datetime, tag: str, book: dict) -> None:
    s.add(Quote(hour_start=hour_start, taken_at=dt.datetime.now(dt.timezone.utc),
                tag=tag, **book))


def log_event(level: str, kind: str, message: str) -> None:
    with session() as s:
        s.add(Event(level=level, kind=kind, message=message))


def pnl_since(s: Session, since: dt.datetime) -> float:
    """Реализованный результат с указанного момента — для риск-лимитов."""
    total = s.scalar(select(func.sum(Market.pnl_usd))
                     .where(Market.resolved_at >= since)
                     .where(Market.dry_run == config.DRY_RUN))
    return float(total or 0.0)


def stats(s: Session, since: dt.datetime | None = None) -> dict:
    q = select(Market).where(Market.status == RESOLVED,
                             Market.dry_run == config.DRY_RUN)
    if since is not None:
        q = q.where(Market.resolved_at >= since)
    rows = list(s.scalars(q))
    filled = [m for m in rows if m.filled_shares]
    wins = [m for m in filled if m.won]

    # Пропуски живут отдельно: у них нет resolved_at, отбираем по часу рынка.
    sq = select(func.count()).select_from(Market).where(
        Market.status == SKIPPED, Market.dry_run == config.DRY_RUN)
    if since is not None:
        sq = sq.where(Market.hour_start >= since)

    return {
        "рынков": len(rows),
        "ставок": len(filled),
        "угадано": len(wins),
        "точность, %": (len(wins) / len(filled) * 100) if filled else float("nan"),
        "P&L, $": sum(m.pnl_usd or 0.0 for m in filled),
        "комиссии, $": sum(m.fee_usd or 0.0 for m in filled),
        "пассивных": sum(m.fill_role == "MAKER" for m in filled),
        "тейкером": sum(m.fill_role == "TAKER" for m in filled),
        "пропущено": int(s.scalar(sq) or 0),
        "ставок на Up, %": (sum(m.side == "Up" for m in filled) / len(filled) * 100)
                           if filled else float("nan"),
    }
