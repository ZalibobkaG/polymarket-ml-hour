"""Оформление графиков для отчётов: палитра и обёртки над matplotlib.

Палитра — первые три слота категориальной схемы; проверены на светлой подложке
(разделимость при дальтонизме, контраст). Больше трёх серий на одном графике не
делаем: четвёртый слот в этой схеме перестаёт надёжно отличаться от второго.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
#: Светлый оттенок основного цвета — для столбиков-подложки под линией.
FILL = "#9ec5f4"


def fig(w=9.0, h=4.6, nrows=1, **kw):
    return plt.subplots(nrows, 1, figsize=(w, h), facecolor=SURFACE, **kw)


def style(ax, title=None, ylabel=None, xlabel=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)


def note(ax, text: str):
    """Пояснение под заголовком — над областью графика, чтобы ничего не перекрывать.

    Заголовок при этом отодвигается выше: иначе пояснение легло бы прямо на него.
    """
    ax.set_title(ax.get_title(loc="left"), loc="left", color=INK, fontsize=12, pad=26)
    ax.text(0, 1.02, text, transform=ax.transAxes, color=MUTED, fontsize=8.5,
            va="bottom", ha="left")


def legend(ax, **kw):
    return ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)


def save(figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return f"{path.parent.name}/{path.name}"
