"""Логирование: в файл с ротацией по суткам и в консоль."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

import config

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-12s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


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
