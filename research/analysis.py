"""Разбор переобученной модели в деньгах: три цены входа, графики, таблицы.

Правила расчёта те же, что в исследовании (`money.py`): на каждый часовой рынок
ставится фиксированная сумма STAKE, то есть покупается STAKE/цена шейров; выигравший
шейр гасится за $1, проигравший за $0; лимитка считается исполненной.

Три режима входа — 48¢, 49¢, 50¢. Разница между ними велика, потому что точка
безубыточности численно равна цене входа: купил по 48 — хватит 48% угаданных,
купил по 50 — нужно 50%.

Запуск:
    python research/analysis.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import money  # noqa: E402
import stats as st  # noqa: E402
import viz  # noqa: E402
from viz import FILL, INK2, MUTED, SERIES  # noqa: E402

FIGS = HERE.parent / "reports"
FIGS.mkdir(exist_ok=True)
CACHE = HERE / "cache"

#: Размер ставки на один рынок. Отдельно от боевого размера (5 шейров) — здесь
#: единая база для сравнения с отчётами исследования.
STAKE = 10.0
PRICES = (0.48, 0.49, 0.50)
ET = "America/New_York"


def load(cutoff: int = 2) -> pd.DataFrame:
    res = pickle.loads((CACHE / f"wf_cutoff{cutoff}.pkl").read_bytes())
    d = res["oos"].sort_index()
    d["contra"] = res["contra"].reindex(d.index).values
    return d


def outcomes(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Булевы векторы «угадал» для модели и базовых стратегий."""
    y = d["y"].values
    return {
        "модель": d["pred"].values == y,
        "против прошлого часа": d["contra"].values == y,
        "всегда Up": y == 1,
    }


# --------------------------------------------------------------------------
# Таблицы
# --------------------------------------------------------------------------

def summary(d: pd.DataFrame) -> pd.DataFrame:
    win = outcomes(d)["модель"]
    months = pd.DatetimeIndex(d.index).tz_convert(ET).to_period("M")
    rows = []
    for p in PRICES:
        ev = money.evaluate(win, p, STAKE, months=months)
        rows.append({
            "цена входа": f"{int(p*100)}¢",
            "ставок": ev["ставок"],
            "угадано, %": round(ev["угадано, %"], 2),
            "нужно, %": ev["нужно, %"],
            "запас, п.п.": round(ev["запас, п.п."], 2),
            "выигрыш даёт, $": round(ev["выигрыш даёт, $"], 2),
            "итог, $": round(ev["итог, $"]),
            "на ставку, $": round(ev["на ставку, $"], 3),
            "ROI, %": round(ev["ROI, %"], 2),
            "просадка, $": round(ev["просадка, $"]),
            "месяцев в минусе": f"{ev['месяцев в минусе']} из {ev['месяцев всего']}",
        })
    return pd.DataFrame(rows).set_index("цена входа")


def baselines(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, win in outcomes(d).items():
        for p in PRICES:
            ev = money.evaluate(win, p, STAKE)
            rows.append({"стратегия": name, "цена": f"{int(p*100)}¢",
                         "угадано, %": round(ev["угадано, %"], 2),
                         "итог, $": round(ev["итог, $"]),
                         "ROI, %": round(ev["ROI, %"], 2)})
    return pd.DataFrame(rows).pivot(index="стратегия", columns="цена",
                                    values=["угадано, %", "итог, $"])


def monthly_table(d: pd.DataFrame) -> pd.DataFrame:
    win = outcomes(d)["модель"]
    months = pd.DatetimeIndex(d.index).tz_convert(ET).to_period("M")
    out = pd.DataFrame(index=sorted(set(months)))
    for p in PRICES:
        pnl = money.pnl(win, p, STAKE)
        out[f"{int(p*100)}¢"] = pd.Series(pnl).groupby(np.asarray(months)).sum()
    out["ставок"] = pd.Series(win).groupby(np.asarray(months)).size()
    out["точность, %"] = (pd.Series(win).groupby(np.asarray(months)).mean() * 100).round(2)
    # Огрызки на краях периода в графики не идут: месяц из 50 ставок ни о чём.
    return out[out["ставок"] >= money.MIN_MONTH_BETS]


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------

def plot_cumulative(d: pd.DataFrame) -> str:
    """Накопленная прибыль по трём ценам входа плюс базовая линия."""
    win = outcomes(d)["модель"]
    contra = outcomes(d)["против прошлого часа"]
    x = pd.DatetimeIndex(d.index)

    f, ax = viz.fig(10.0, 5.2)
    for color, p in zip(SERIES, PRICES):
        eq = money.equity_curve(win, p, STAKE)
        ax.plot(x, eq, color=color, linewidth=1.7,
                label=f"модель, вход по {int(p*100)}¢ → ${eq[-1]:+,.0f}")
    eq_c = money.equity_curve(contra, 0.50, STAKE)
    ax.plot(x, eq_c, color=MUTED, linewidth=1.2, linestyle="--",
            label=f"против прошлого часа, 50¢ → ${eq_c[-1]:+,.0f}")
    ax.axhline(0, color=viz.AXIS, linewidth=1)

    viz.style(ax, "Накопленная прибыль", ylabel="$", xlabel="")
    viz.note(ax, f"ставка ${STAKE:.0f} на каждый из {len(d):,} часовых рынков; "
                 "точка безубыточности численно равна цене входа")
    viz.legend(ax, loc="upper left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+,.0f}")
    return viz.save(f, FIGS / "cumulative.png")


def plot_monthly(d: pd.DataFrame) -> str:
    """Прибыль по месяцам: столбики для трёх цен рядом."""
    t = monthly_table(d)
    labels = [str(p) for p in t.index]
    x = np.arange(len(t))
    width = 0.27

    f, ax = viz.fig(10.0, 5.0)
    for i, (color, p) in enumerate(zip(SERIES, PRICES)):
        col = f"{int(p*100)}¢"
        ax.bar(x + (i - 1) * width, t[col], width, color=color, label=f"вход по {col}")
    ax.axhline(0, color=viz.AXIS, linewidth=1)

    viz.style(ax, "Прибыль по месяцам", ylabel="$")
    neg = {f"{int(p*100)}¢": int((t[f'{int(p*100)}¢'] < 0).sum()) for p in PRICES}
    viz.note(ax, "минусовых месяцев: " +
             ", ".join(f"{k} — {v} из {len(t)}" for k, v in neg.items()))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    viz.legend(ax, loc="upper left", ncol=3)
    return viz.save(f, FIGS / "monthly.png")


def plot_drawdown(d: pd.DataFrame) -> str:
    win = outcomes(d)["модель"]
    x = pd.DatetimeIndex(d.index)
    f, ax = viz.fig(10.0, 4.0)
    for color, p in zip(SERIES, PRICES):
        eq = money.equity_curve(win, p, STAKE)
        mdd, dd = money.drawdown(eq)
        ax.plot(x, dd, color=color, linewidth=1.3,
                label=f"{int(p*100)}¢ — худшая ${mdd:,.0f}")
    viz.style(ax, "Просадка от предыдущего максимума", ylabel="$")
    viz.note(ax, "одна историческая траектория, а не оценка будущего риска")
    viz.legend(ax, loc="lower left")
    return viz.save(f, FIGS / "drawdown.png")


def plot_accuracy_by_month(d: pd.DataFrame) -> str:
    t = monthly_table(d)
    x = np.arange(len(t))
    f, ax = viz.fig(10.0, 4.2)
    ax.bar(x, t["точность, %"], color=FILL, label="точность модели")
    ax.axhline(50, color=MUTED, linewidth=1.2, linestyle="--", label="монетка")
    for color, p in zip(SERIES, PRICES):
        ax.axhline(p * 100, color=color, linewidth=1.1,
                   label=f"безубыточность при {int(p*100)}¢")
    viz.style(ax, "Точность по месяцам против точек безубыточности", ylabel="%")
    viz.note(ax, "столбик выше линии своего цвета — этот месяц по этой цене прибылен")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in t.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(45, 60)
    viz.legend(ax, loc="upper right", ncol=2)
    return viz.save(f, FIGS / "accuracy_monthly.png")


def plot_calibration(d: pd.DataFrame) -> str:
    """Калибровка: что обещала модель против того, что случилось."""
    q = d["proba"].values
    y = d["y"].values
    bins = np.quantile(q, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    idx = np.clip(np.digitize(q, bins[1:-1]), 0, len(bins) - 2)
    pred, act, cnt = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 50:
            continue
        pred.append(q[m].mean() * 100)
        act.append(y[m].mean() * 100)
        cnt.append(m.sum())

    f, (ax, ax2) = viz.fig(9.0, 6.4, nrows=2,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    lim = [min(pred + act) - 1, max(pred + act) + 1]
    ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1, label="идеальная калибровка")
    ax.plot(pred, act, color=SERIES[0], marker="o", markersize=5, linewidth=1.8,
            label="модель")
    viz.style(ax, "Калибровка", ylabel="фактически Up, %", xlabel="предсказано, %")
    viz.note(ax, "линия положе диагонали — модель переуверена: обещает больше, "
                 "чем сбывается (наклон 0.717)")
    viz.legend(ax, loc="upper left")

    ax2.hist(q * 100, bins=60, color=FILL)
    viz.style(ax2, "Распределение предсказанной вероятности", ylabel="часов", xlabel="%")
    viz.note(ax2, f"размах {q.min()*100:.1f}–{q.max()*100:.1f}%, "
                  f"среднее отклонение от 50% — {np.abs(q-0.5).mean()*100:.1f} п.п.")
    return viz.save(f, FIGS / "calibration.png")


def plot_by_hour(d: pd.DataFrame) -> str:
    """Точность по часам суток ET — проверка сессионного расщепления."""
    et = pd.DatetimeIndex(d.index).tz_convert(ET)
    win = outcomes(d)["модель"]
    t = pd.DataFrame({"h": et.hour, "win": win}).groupby("h")["win"].agg(["mean", "size"])
    t["mean"] *= 100
    band = 1.96 * np.sqrt(50 * 50 / t["size"].mean())

    f, ax = viz.fig(10.0, 4.2)
    colors = [SERIES[0] if 13 <= h < 23 else MUTED for h in t.index]
    ax.bar(t.index, t["mean"] - 50, bottom=50, color=colors, width=0.75)
    ax.axhline(50, color=viz.AXIS, linewidth=1)
    ax.axhspan(50 - band, 50 + band, color=viz.GRID, alpha=0.55, zorder=0)
    us = win[(et.hour >= 13) & (et.hour < 23)].mean() * 100
    other = win[~((et.hour >= 13) & (et.hour < 23))].mean() * 100
    viz.style(ax, "Точность по часам суток (нью-йоркское время)", ylabel="%", xlabel="час ET")
    viz.note(ax, f"синим — американская сессия 13:00–23:00 ET: {us:.2f}%; "
                 f"остальные часы: {other:.2f}%; серая полоса — шум")
    ax.set_ylim(46, 58)
    return viz.save(f, FIGS / "by_hour.png")


def plot_price_sensitivity(d: pd.DataFrame) -> str:
    """Итог в деньгах как функция цены входа — почему цент решает так много."""
    win = outcomes(d)["модель"]
    contra = outcomes(d)["против прошлого часа"]
    grid = np.arange(0.44, 0.581, 0.005)
    f, ax = viz.fig(9.5, 4.4)
    for name, w, color in (("модель", win, SERIES[0]),
                           ("против прошлого часа", contra, SERIES[1])):
        totals = [money.evaluate(w, p, STAKE)["итог, $"] for p in grid]
        ax.plot(grid * 100, totals, color=color, linewidth=1.8, label=name)
    ax.axhline(0, color=viz.AXIS, linewidth=1)
    for color, p in zip(SERIES, PRICES):
        ax.axvline(p * 100, color=color, linewidth=0.9, linestyle=":")
    acc = win.mean() * 100
    ax.axvline(acc, color=MUTED, linewidth=1.1, linestyle="--")
    ax.annotate(f"точность модели {acc:.2f}%", xy=(acc, 0), xytext=(acc + 0.6, 2000),
                color=INK2, fontsize=8.5)
    viz.style(ax, "Сколько приносит стратегия при разной цене входа", ylabel="$",
              xlabel="цена входа, центов")
    viz.note(ax, "стратегия в нуле ровно там, где цена равна точности: w = p")
    viz.legend(ax, loc="upper right")
    return viz.save(f, FIGS / "price_sensitivity.png")


def plot_quarterly(d: pd.DataFrame) -> str:
    """Точность модели и базовой линии по кварталам."""
    o = outcomes(d)
    t = pd.DataFrame({"fold": d["fold"].values, "модель": o["модель"],
                      "правило": o["против прошлого часа"],
                      "всегда Up": o["всегда Up"]}).groupby("fold").mean() * 100
    x = np.arange(len(t))
    width = 0.27
    f, ax = viz.fig(9.5, 4.4)
    for i, (col, color) in enumerate(zip(["модель", "правило", "всегда Up"], SERIES)):
        ax.bar(x + (i - 1) * width, t[col] - 50, width, bottom=50, color=color, label=col)
    ax.axhline(50, color=viz.AXIS, linewidth=1)
    viz.style(ax, "Точность по кварталам", ylabel="%")
    wins = int((t["модель"] > t["правило"]).sum())
    viz.note(ax, f"модель обыгрывает правило в {wins} кварталах из {len(t)}")
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, fontsize=9)
    ax.set_ylim(47, 56)
    viz.legend(ax, loc="upper right", ncol=3)
    return viz.save(f, FIGS / "quarterly.png")


# --------------------------------------------------------------------------

def main() -> None:
    d = load(2)
    print(f"строк вне обучения: {len(d):,}  "
          f"({d.index[0]:%Y-%m-%d} → {d.index[-1]:%Y-%m-%d})\n")

    print("=== ИТОГ ПО ЦЕНАМ ВХОДА (ставка $10) ===")
    s = summary(d)
    print(s.to_string(), "\n")

    print("=== ПРОТИВ СТРАТЕГИЙ БЕЗ МОДЕЛИ ===")
    print(baselines(d).to_string(), "\n")

    print("=== ПО МЕСЯЦАМ ===")
    m = monthly_table(d)
    print(m.to_string(), "\n")

    for fn in (plot_cumulative, plot_monthly, plot_drawdown, plot_accuracy_by_month,
               plot_calibration, plot_by_hour, plot_price_sensitivity, plot_quarterly):
        print("график:", fn(d))

    s.to_pickle(CACHE / "summary.pkl")
    m.to_pickle(CACHE / "monthly.pkl")


if __name__ == "__main__":
    main()
