"""Логирование: в файл с ротацией по суткам и в консоль."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import config

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-12s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False

#: Сколько байт с конца читать под хвост. Ротация суточная, за сутки набегает
#: около 50 КБ, так что этого хватает на любой разумный день с запасом — а
#: заодно защищает от чтения в память гигабайта, если логи вдруг разбухнут.
_MAX_TAIL_BYTES = 8 * 1024 * 1024


def setup(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or "INFO").upper()))

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    fmt.converter = __import__("time").gmtime          # всё в UTC, как и торговля

    fh = TimedRotatingFileHandler(config.LOG_DIR / "bot.log", when="midnight",
                                  backupCount=30, encoding="utf-8", utc=True)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # httpx болтлив на уровне INFO — каждый запрос отдельной строкой.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _configured = True


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(name)


def path() -> Path:
    """Текущий лог-файл (без суффикса ротации — тот, в который пишем сейчас)."""
    return config.LOG_DIR / "bot.log"


def tail(lines: int | None = 2000) -> bytes:
    """Хвост текущего лога как UTF-8 байты. `lines=None` — файл целиком.

    Читается с конца, а не целиком: файл открыт на запись живым процессом и
    расти может как угодно. Первая строка прочитанного куска отбрасывается,
    если читали не с начала файла — она почти наверняка обрезана посередине,
    в том числе посреди многобайтового символа.
    """
    p = path()
    if not p.exists():
        raise FileNotFoundError(f"лог-файла ещё нет: {p}")

    with p.open("rb") as fh:
        size = fh.seek(0, 2)
        start = 0 if lines is None else max(0, size - _MAX_TAIL_BYTES)
        if lines is None and size > _MAX_TAIL_BYTES:
            start = size - _MAX_TAIL_BYTES
        fh.seek(start)
        data = fh.read()

    if start:
        data = data.partition(b"\n")[2]
    if lines is not None:
        rows = data.splitlines(keepends=True)
        data = b"".join(rows[-lines:])
    return data
