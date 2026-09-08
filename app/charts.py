"""Графики для Telegram (`/stats`): equity curve, уверенность модели, угадывания.

Рисуются в память (PNG в `BytesIO`), на диск ничего не пишется. Backend `Agg` —
headless, сервер без дисплея. matplotlib — единственная тяжёлая зависимость
проекта, но гонять торговые данные во внешний сервис отрисовки (quickchart.io
и подобные) хуже: и утечка данных на сторону, и зависимость от чужого аптайма.

Каждая функция возвращает `None`, если рисовать нечего (свежая база после
сброса статистики, например) — вызывающий код тогда шлёт текст вместо картинки.
"""

from __future__ import annotations

import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from app import storage

# На коротком диапазоне (первые дни после сброса статистики) AutoDateLocator не
# находит «круглый» интервал под наши minticks/maxticks и падает на дефолтный
# шаг — сам по себе не проблема, просто предупреждает. Тиков это не портит
# (см. тест на реальных данных), а в логе как ложная тревога не нужно.
warnings.filterwarnings("ignore", message="AutoDateLocator was unable to pick.*",
                        category=UserWarning)

# Тёмная тема на весь модуль — эти графики только и делают, что открываются
# картинкой в Telegram, конкурировать за rcParams с чем-то ещё в процессе не с чем.
plt.style.use("dark_background")

# Крупнее, чем дефолт: график открывают на весь экран телефона — мелкий текст
# и тонкие линии на таком масштабе становятся кашей.
_FIGSIZE = (11, 6.5)
_DPI = 170
# На чёрном фоне тёмно-зелёный/тёмно-красный/тёмно-синий из светлой темы гаснут —
# те же цвета, но на тон-два ярче.
_GREEN, _RED, _BLUE, _GREY, _AMBER = (
    "#66bb6a", "#ef5350", "#42a5f5", "#9e9e9e", "#ffb300")


def _finish(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _dated_axes(ax) -> None:
    # Локатор по умолчанию на плотных часовых данных иногда кладёт две метки
    # одной даты подряд — фиксируем разумное число тиков явно.
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=0.25)


def equity_curve() -> bytes | None:
    """Кумулятивный P&L по резолвнутым ставкам (только заполненные), во времени.

    Изменения размера ставки (`/shares`) отмечаются вертикальной пунктирной
    линией с подписью в легенде — без этого скачок наклона equity curve легко
    принять за смену везения, а не размера позиции.
    """
    with storage.session() as s:
        rows = [(storage.as_utc(m.hour_start), m.pnl_usd or 0.0)
                for m in s.query(storage.Market)
                .filter(storage.Market.status == storage.RESOLVED,
                        storage.Market.filled_shares > 0)
                .order_by(storage.Market.hour_start)]
        changes = [(storage.as_utc(c.changed_at), c.old_shares, c.new_shares)
                   for c in storage.share_changes(s)]
    if not rows:
        return None

    times = [r[0] for r in rows]
    cum, total = [], 0.0
    for _, pnl in rows:
        total += pnl
        cum.append(total)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(times, cum, color=_GREEN if cum[-1] >= 0 else _RED, linewidth=2)
    ax.axhline(0, color=_GREY, linewidth=0.8, linestyle="--")
    for changed_at, old, new in changes:
        if times[0] <= changed_at <= times[-1]:
            ax.axvline(changed_at, color=_AMBER, linewidth=1.3, linestyle=":",
                      label=f"{old:g}→{new:g} shares ({changed_at:%d.%m})")
    if any(times[0] <= c[0] <= times[-1] for c in changes):
        ax.legend(fontsize=9)
    ax.set_title("Equity curve")
    ax.set_ylabel("Кумулятивный P&L, $")
    _dated_axes(ax)
    fig.autofmt_xdate()
    return _finish(fig)


def confidence_curve() -> bytes | None:
    """P(Up) модели на каждый час, где принималось решение — включая пропуски."""
    with storage.session() as s:
        rows = [(storage.as_utc(m.hour_start), m.proba_up)
                for m in s.query(storage.Market)
                .filter(storage.Market.proba_up.isnot(None))
                .order_by(storage.Market.hour_start)]
    if not rows:
        return None

    times = [r[0] for r in rows]
    proba = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(times, proba, color=_BLUE, linewidth=1.2)
    ax.axhline(0.5, color=_GREY, linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_title("Уверенность модели, P(Up)")
    _dated_axes(ax)
    fig.autofmt_xdate()
    return _finish(fig)


def guess_walk() -> bytes | None:
    """Кумулятивные ±1 (угадали/нет) — своя ось X у каждой линии.

    «По всем» — любой резолвнутый рынок с решённой стороной, даже если ордер
    не встал. «По заполненным» — только там, где реально стояла ставка.
    Разница между линиями показывает, сколько «съедает» именно исполнение,
    а не качество прогноза.
    """
    with storage.session() as s:
        rows = [(bool(m.won), (m.filled_shares or 0.0) > 0)
                for m in s.query(storage.Market)
                .filter(storage.Market.won.isnot(None))
                .order_by(storage.Market.hour_start)]
    if not rows:
        return None

    def walk(subset):
        total, out = 0, []
        for won in subset:
            total += 1 if won else -1
            out.append(total)
        return out

    all_won = [won for won, _ in rows]
    filled_won = [won for won, was_filled in rows if was_filled]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if all_won:
        ax.plot(range(1, len(all_won) + 1), walk(all_won),
               color=_GREY, linewidth=1.2, label=f"по всем ({len(all_won)})")
    if filled_won:
        ax.plot(range(1, len(filled_won) + 1), walk(filled_won),
               color=_BLUE, linewidth=1.8,
               label=f"по заполненным ({len(filled_won)})")
    ax.axhline(0, color=_GREY, linewidth=0.8, linestyle="--")
    ax.set_title("Угадывания")
    ax.set_xlabel("№ рынка")
    ax.set_ylabel("Кумулятивно ±1")
    ax.legend()
    ax.grid(alpha=0.25)
    return _finish(fig)
