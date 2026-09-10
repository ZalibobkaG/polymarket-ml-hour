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

#: Чем набрана позиция (`Market.fill_role`) — по тому, какой кусок дал больше шейров.
MAKER = "maker"                        # лимитку налили из книги
IMMEDIATE_TAKER = "immediate_taker"    # аск уже был ≤ цены при постановке — сразу
CROSS = "cross"                        # добор тейкером на CROSS_AFTER_SECONDS
NO_FILL = "none"                       # ставку решили сделать, но ничего не налили
ROLES = (MAKER, IMMEDIATE_TAKER, CROSS, NO_FILL)
#: Строки, записанные до новой схемы исполнения: роль там — сторона биржи.
#: Тейкером старая схема исполнялась только в момент постановки.
_LEGACY_ROLES = {"MAKER": MAKER, "TAKER": IMMEDIATE_TAKER, "MIXED": "mixed"}


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
    #: Когда заявка по этому часу впервые ушла на биржу.
    place_ts = Column(DateTime(timezone=True))
    #: Сколько раз пытались поставить ордер, прежде чем биржа его приняла.
    place_attempts = Column(Integer, default=0)
    filled_shares = Column(Float, default=0.0)
    #: Средневзвешенная цена реального исполнения всех кусков (лимитка + добор),
    #: не наша заявка. Может быть лучше заявки: лимит 0.51 при аске 0.47 — по 0.47.
    fill_price = Column(Float)
    filled_at = Column(DateTime(timezone=True))
    #: maker / immediate_taker / cross / none — см. константы выше. У строк до
    #: новой схемы здесь "MAKER" / "TAKER" / "MIXED"; читать через `role_of`.
    fill_role = Column(String(16))
    #: Комиссия тейкера по формуле FEE_RATE·p·(1−p) — консервативная оценка, по
    #: ней считается P&L. Сколько биржа списала на самом деле — `fee_charged_usd`.
    fee_usd = Column(Float, default=0.0)
    #: Комиссия, которую биржа фактически начислила по нашим кускам (по данным
    #: сделки). None — биржа не сообщила; 0.0 — сообщила, что не брала.
    fee_charged_usd = Column(Float)

    # --- добор тейкером (runner.cross_remaining)
    #: Когда добор состоялся или от него отказались. Пишется ДО отправки заявки:
    #: непустое значение означает «второго добора по этому часу не будет».
    cross_ts = Column(DateTime(timezone=True))
    cross_order_id = Column(String(80))
    #: Что произошло с добором: цена, отказ по потолку, пауза, опоздание.
    cross_note = Column(Text)

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
    """Снимок стакана. Пишется в момент решения и перед добором тейкером."""

    __tablename__ = "quotes"

    id = Column(Integer, primary_key=True)
    hour_start = Column(DateTime(timezone=True), nullable=False, index=True)
    taken_at = Column(DateTime(timezone=True), nullable=False)
    tag = Column(String(16))                 # "decision" / "cross"
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


class ShareChange(Base):
    """Лента изменений размера ставки (`/shares` в Telegram).

    Одна таблица закрывает две задачи: «сколько шейров ставим сейчас» — это
    просто последняя строка, и аудит-лог для вертикальных линий на equity
    curve — отдельного хранилища для «текущего значения» не нужно.
    """

    __tablename__ = "share_changes"

    id = Column(Integer, primary_key=True)
    changed_at = Column(DateTime(timezone=True), nullable=False, index=True,
                        server_default=func.now())
    old_shares = Column(Float, nullable=False)
    new_shares = Column(Float, nullable=False)


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
        log.info("DB: added columns {} to {}",
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


def add_quote(s: Session, hour_start: dt.datetime, tag: str, book: dict,
              at: dt.datetime | None = None) -> None:
    s.add(Quote(hour_start=hour_start, tag=tag,
                taken_at=at or dt.datetime.now(dt.timezone.utc), **book))


def log_event(level: str, kind: str, message: str) -> None:
    with session() as s:
        s.add(Event(level=level, kind=kind, message=message))


def current_shares(s: Session) -> float:
    """Сколько шейров ставим прямо сейчас — последнее изменение, а если его ещё
    не было (свежая база), дефолт из `config.py`."""
    row = s.scalar(select(ShareChange).order_by(ShareChange.changed_at.desc()).limit(1))
    return row.new_shares if row else float(config.SHARES)


def set_shares(s: Session, new_shares: float) -> ShareChange | None:
    """Записать изменение размера ставки. `None`, если значение не поменялось."""
    old = current_shares(s)
    if abs(old - new_shares) < 1e-9:
        return None
    change = ShareChange(old_shares=old, new_shares=new_shares)
    s.add(change)
    s.flush()
    return change


def share_changes(s: Session) -> list[ShareChange]:
    """Вся лента изменений по порядку времени — для вертикальных линий на графике."""
    return list(s.scalars(select(ShareChange).order_by(ShareChange.changed_at)))


def pnl_since(s: Session, since: dt.datetime) -> float:
    """Реализованный результат с указанного момента — для риск-лимитов."""
    total = s.scalar(select(func.sum(Market.pnl_usd))
                     .where(Market.resolved_at >= since)
                     .where(Market.dry_run == config.DRY_RUN))
    return float(total or 0.0)


def role_of(m: Market) -> str | None:
    """Чем набрана позиция — в словаре новой схемы, в том числе у старых строк.

    None — ставку по часу не решали (пропуск, ошибка). Строка до новой схемы с
    "MIXED" возвращается как "mixed": из неё не восстановить, какой кусок был каким.
    """
    if not m.side:
        return None
    if not m.filled_shares:
        return NO_FILL
    role = m.fill_role or ""
    return _LEGACY_ROLES.get(role, role or None)


def execution(rows: list[Market]) -> dict:
    """Сводка исполнения по строкам с решённой ставкой: доли ролей, цена, комиссии.

    Доли считаются от всех часов, где ставку решили сделать, — включая те, где
    ничего не налили (`none`). Средняя цена входа — по шейрам: сколько в среднем
    заплатили за шейр по всем исполненным кускам.
    """
    bets = [m for m in rows if m.side]
    roles = [role_of(m) for m in bets]
    filled = [m for m in bets if m.filled_shares]
    shares = sum(m.filled_shares for m in filled)
    known = [m for m in filled if m.fee_charged_usd is not None]
    out = {"решено ставить": len(bets)}
    for role, label in ((MAKER, "мейкер"), (IMMEDIATE_TAKER, "сразу тейкер"),
                        (CROSS, "добор"), (NO_FILL, "без исполнения")):
        n = sum(r == role for r in roles)
        out[f"{label}, %"] = n / len(bets) * 100 if bets else float("nan")
    legacy = sum(r == "mixed" for r in roles)
    if legacy:
        out["смешанных (старые строки)"] = legacy
    out["средняя цена входа"] = (sum((m.fill_price or m.price) * m.filled_shares
                                     for m in filled) / shares) if shares else float("nan")
    out["комиссий списано биржей, $"] = sum(m.fee_charged_usd for m in known)
    out["комиссия неизвестна, ставок"] = len(filled) - len(known)
    return out


def stats(s: Session, since: dt.datetime | None = None) -> dict:
    q = select(Market).where(Market.status == RESOLVED,
                             Market.dry_run == config.DRY_RUN)
    if since is not None:
        q = q.where(Market.resolved_at >= since)
    rows = list(s.scalars(q))
    filled = [m for m in rows if m.filled_shares]
    wins = [m for m in filled if m.won]
    roles = [role_of(m) for m in filled]

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
        # По формуле FEE_RATE — ею же уменьшен P&L; фактическое списание ниже.
        "комиссии, $": sum(m.fee_usd or 0.0 for m in filled),
        # Ключи читает дневная сводка в Telegram: «пассивно N, тейкером M».
        "пассивных": sum(r == MAKER for r in roles),
        "тейкером": sum(r in (IMMEDIATE_TAKER, CROSS) for r in roles),
        "пропущено": int(s.scalar(sq) or 0),
        "ставок на Up, %": (sum(m.side == "Up" for m in filled) / len(filled) * 100)
                           if filled else float("nan"),
        **execution(rows),
    }
