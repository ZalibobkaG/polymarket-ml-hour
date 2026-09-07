"""Logging: loguru with daily-rotating file, console mirror, and Telegram tail.

Why loguru over stdlib `logging`: call-site `module:function:line` comes for
free in the format string, no boilerplate `logging.getLogger(__name__)` dance,
and level-based icons double as a colour code for scanning the log — see
`_FORMAT` below.

Level = status colour, chosen deliberately per call site, not derived from
"how bad is this technically":

    SUCCESS (✅ green)   — a good outcome: fill, win, order placed, retrain ok.
    INFO    (ℹ️ neutral) — plain facts: a decision was made, a request came in.
    DEBUG   (🐞 quiet)   — internal plumbing: WS trigger, a single retry attempt.
    WARNING (⚠️ yellow)  — a setback we expected and handled: a loss, a
                            reconnect, an unfilled hour, a retry.
    ERROR   (❌ red)     — needs a human: lost hour, resolve mismatch, a bug.

So a losing trade is WARNING, not ERROR — losing is a normal, priced-in
outcome of the strategy. ERROR is reserved for the bot actually malfunctioning.

Everything lands in one file, including DEBUG: `/log` in Telegram sends this
same file, and nothing is hidden from it — filtering happens by scanning the
level/icon, not by dropping lines at write time.
"""

from __future__ import annotations

import datetime as dt
import inspect
import logging
import sys
from pathlib import Path

from loguru import logger

import config

# "!UTC" — вся торговля идёт в UTC, и время в логе должно совпадать с ним
# независимо от часового пояса машины, на которой крутится бот.
_FORMAT = ("{time:YYYY-MM-DD HH:mm:ss.SSS!UTC} | {level.icon} {level: <8} | "
          "{extra[mod]}:{function}:{line} | {message}")
#: Ротация по UTC-полуночи, а не по местной, — та же причина.
_ROTATE_AT = dt.time(0, 0, tzinfo=dt.timezone.utc)
_configured = False

#: How many bytes from the end to read for the tail. Rotation is daily and a
#: DEBUG-inclusive day runs a few hundred KB, so this covers any reasonable
#: day with headroom — and caps memory if the log ever balloons.
_MAX_TAIL_BYTES = 16 * 1024 * 1024


class _InterceptHandler(logging.Handler):
    """Redirects stdlib `logging` records into loguru, level and all.

    Standard loguru recipe: walk back up the stack past stdlib's own logging
    frames so `{function}:{line}` in the log line points at the real caller
    (e.g. inside httpx), not at this handler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.bind(mod=record.name).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage())


def setup(file_level: str | None = None, console_level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()  # drop loguru's default stderr sink, we set our own below

    logger.add(config.LOG_DIR / "bot.log", format=_FORMAT,
               level=(file_level or "DEBUG").upper(), rotation=_ROTATE_AT,
               retention="30 days", encoding="utf-8", enqueue=True)
    logger.add(sys.stdout, format=_FORMAT,
               level=(console_level or "INFO").upper(), colorize=True)

    # Third-party libraries (httpx and everything under it — httpcore, hpack,
    # h2 — plus urllib3, websocket-client, py_clob_client_v2) go through stdlib
    # `logging`, routed here into loguru. Default threshold is WARNING: our own
    # DEBUG-level detail comes from *our* calls to `get()`, not from network
    # libraries dumping every frame/header at DEBUG. py_clob_client_v2 additionally
    # logs its own HTTP errors on every failed request, including ones our code
    # already handles and logs meaningfully (e.g. a post-only reject) — muted
    # fully to avoid a duplicate, uninformative line for every such request.
    root = logging.getLogger()
    root.handlers = [_InterceptHandler()]
    root.setLevel(logging.WARNING)
    logging.getLogger("py_clob_client_v2.http_helpers.helpers").setLevel(logging.CRITICAL)

    _configured = True


def get(name: str):
    """A logger bound to a short module tag, e.g. `get("runner")`.

    `.info`/`.warning`/`.error`/`.exception`/`.success`/`.debug` all take
    `{}`-style placeholders (loguru's `str.format`, not stdlib's `%s`).
    """
    setup()
    return logger.bind(mod=name)


def path() -> Path:
    """Current log file (no rotation suffix — the one being written to now)."""
    return config.LOG_DIR / "bot.log"


def tail(lines: int | None = 2000) -> bytes:
    """Tail of the current log as UTF-8 bytes. `lines=None` — the whole file.

    Read from the end, not loaded whole: the file is open for writing by a
    live process and can grow arbitrarily. The first line of a chunk read from
    the middle is dropped — it is almost certainly cut mid-line, possibly
    mid-multibyte-character.
    """
    p = path()
    if not p.exists():
        raise FileNotFoundError(f"log file does not exist yet: {p}")

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
