"""Признаки режима поверх дневных и часовых свечей из `btc_data`.

Зачем: базовая модель (см. MODEL_REPORT.md) умела только повторять преобладающий
класс квартала. Отличить растущий режим от падающего ей было нечем — все её признаки
описывали ровно один предыдущий день. Здесь добавляется то, по чему режим виден:
положение цены относительно скользящих, расхождение самих скользящих между собой,
волатильность, положение в канале и внутридневная структура из часовых свечей.

Два правила, которые здесь соблюдаются жёстко:

1.  Всё относительное. Уровни цены в признаки не идут: «65 000» в 2024 и в 2026 —
    разные вещи, и правило вида «если close > 100000» в новом режиме не сработает
    никогда. Только проценты, доли и z-оценки.

2.  Никакой утечки. Строка market_date = D описывает рынок, который резолвится в D.
    Каждый признак считается по данным, закрытым к моменту старта этих суток, то есть
    по D-1 включительно, и назван с суффиксом `_prev`. Взятие предыдущего дня всегда
    календарное (reindex на D-1), а не shift(1): при простое биржи shift сравнил бы
    через дыру.

Скользящие считаются двумя способами — SMA и EMA — чтобы сравнить их на одних и тех
же фолдах, а не выбирать умозрительно (`ma="sma"` / `ma="ema"`).

Пример:
    >>> import btc_features as bf
    >>> feats = bf.build_features(start="2023-07-27", ma="sma")
    >>> X, y = bf.make_xy(feats)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import btc_data as bd

#: Длины скользящих. 7 — неделя, 14 и 30 — как в базовой таблице, 100 — длинный тренд.
MA_WINDOWS = (7, 14, 30, 100)
#: На сколько дней раньше грузить историю. Не 100, а сильно больше: одна неполная свеча
#: (простой биржи хоть на минуту) обнуляет окно на следующие 100 дней, и без запаса
#: такой день внутри разгона съел бы начало выборки.
WARMUP_DAYS = 300

VOL_WINDOWS = (14, 30)
ATR_WINDOW = 14
CHANNEL_WINDOW = 20
UP_SHARE_WINDOW = 20
VOLUME_Z_WINDOW = 30
#: Окна для асимметрии «средний рост против среднего падения».
ASYM_WINDOWS = (10, 20)
#: Окна RSI и короткий канал — признаки возврата к среднему.
RSI_WINDOWS = (7, 14)
SHORT_CHANNEL = 10

TARGET = "y"
#: Колонки, которые не являются признаками (таргет и справочные).
NON_FEATURES = ["outcome", "y"]

#: Набор признаков базовой модели (MODEL_REPORT.md) — чтобы сравнивать на тех же строках.
BASE_FEATURES = [
    "change_pct_prev", "y_prev", "volume_change_pct_prev",
    "ma14_change_pct_prev", "ma30_change_pct_prev",
]

#: Смысловые блоки признаков. Байесовский поиск включает и выключает их целиком:
#: 34 отдельных флага — это 2^34 вариантов, TPE в них не разберётся за сотню попыток,
#: а на уровне блоков сразу видно, какая группа вообще что-то даёт.
FEATURE_GROUPS = {
    "тренд": [
        "ma7_change_pct_prev", "ma14_change_pct_prev", "ma30_change_pct_prev",
        "ma100_change_pct_prev", "ma7_dist_prev", "ma14_dist_prev", "ma30_dist_prev",
        "ma100_dist_prev", "ma7_vs_ma30_prev", "ma30_vs_ma100_prev", "ma7_vs_ma100_prev",
        "ma30_dist_delta3_prev", "ma30_dist_delta7_prev", "ma30_dist_delta14_prev",
    ],
    "волатильность": ["vol14_prev", "vol30_prev", "vol_ratio_prev", "atr14_prev",
                      "range_pct_prev"],
    "положение": ["close_pos_prev", "channel_pos_prev", "up_share20_prev",
                  "asym10_prev", "asym20_prev"],
    "возврат": ["rsi7_prev", "rsi14_prev", "channel_pos10_prev",
                "dist_high10_prev", "dist_low10_prev", "streak_prev"],
    "объём": ["volume_z30_prev", "volume_change_pct_prev"],
    "вчера": ["change_pct_prev", "y_prev"],
    "часы": ["hours_up_share_prev", "hour_vol_prev", "hour_of_high_prev",
             "hour_of_low_prev", "ret3h_prev", "ret6h_prev", "ret12h_prev"],
    "календарь": ["dow"],
}


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------

def _calendar(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Разворачивает таблицу на сплошной календарь: в дырах будет NaN.

    Нужно, чтобы окна не съезжали. Без этого rolling(100) в месте простоя биржи
    молча захватил бы более старые цены и выдал бы значение, которого не было.
    """
    idx = pd.DatetimeIndex(df.index)
    return df.reindex(pd.date_range(idx[0], idx[-1], freq="D"))


def _ma(series: pd.Series, window: int, kind: str) -> pd.Series:
    """SMA или EMA. Обе возвращают NaN там, где в окне не хватает дней."""
    sma = series.rolling(window, min_periods=window).mean()
    if kind == "sma":
        return sma
    if kind == "ema":
        ema = series.ewm(span=window, min_periods=window, ignore_na=False).mean()
        # Гасим EMA там, где SMA не определена: иначе EMA тихо перешагнула бы дыру
        # в данных, и два варианта считались бы на разном объёме истории.
        return ema.where(sma.notna())
    raise ValueError(f"ma должно быть 'sma' или 'ema', а не {kind!r}")


def _dist_pct(price: pd.Series, ma: pd.Series) -> pd.Series:
    """На сколько процентов цена выше своей скользящей."""
    return (price / ma - 1) * 100


def _asymmetry(ret: pd.Series, window: int) -> pd.Series:
    """Средний рост делить на среднее падение за окно. >1 — растёт резче, чем падает.

    Это не то же самое, что наклон скользящей. Наклон — это итог `p*рост - (1-p)*падение`,
    то есть частота и амплитуда, схлопнутые в одно число. Здесь вторая половина
    разложения, и она почти не связана с первой: корреляция с долей растущих дней 0.06.
    Аддитивным моделям (наивный байес, логрегрессия) отношение надо давать готовым —
    произведений признаков они не строят.
    """
    up = (ret > 0).astype("float64").where(ret.notna())
    n_up = up.rolling(window, min_periods=window).sum()
    mean_up = ret.where(ret > 0, 0.0).rolling(window, min_periods=window).sum() / n_up
    mean_dn = (-ret).where(ret < 0, 0.0).rolling(window, min_periods=window).sum() / (window - n_up)
    return (mean_up / mean_dn).replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI по Уайлдеру: 0 — перепроданность, 100 — перекупленность.

    Классическая мера возврата к среднему. Добавлена потому, что связь доли растущих
    дней с исходом оказалась контрарной: после серии роста следующий день чаще падает.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, ignore_na=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, ignore_na=False).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    # Гасим там, где в окне есть дыра, — как и для EMA.
    return rsi.where(close.rolling(window, min_periods=window).mean().notna())


def _streak(ret: pd.Series) -> pd.Series:
    """Длина текущей серии одинаковых дней со знаком: +3 — третий рост подряд."""
    sign = np.sign(ret)
    group = (sign != sign.shift()).cumsum()
    return sign * (sign.groupby(group).cumcount() + 1)


# --------------------------------------------------------------------------
# Дневные признаки
# --------------------------------------------------------------------------

def _daily_history(m1: pd.DataFrame) -> pd.DataFrame:
    """OHLCV суток рынка на сплошном календаре — история, по которой считаются окна.

    Собирается из минуток, а не из дневной таблицы, и условие мягче: достаточно, чтобы
    сутки закрывались расчётной минуткой полудня ET. Дневная таблица дополнительно
    требует все 1440 минут и целиком выбрасывает день, потерявший хоть одну, — для
    таргета это правильно, а для истории губительно: 80 минут простоя биржи 2023-03-24
    обнулили бы SMA(100) на следующие сто дней. Close у такого дня настоящий,
    искажены только High/Low/Volume, и то на величину пропущенных минут.
    """
    keyed = bd.add_market_time(m1)
    grouped = keyed.groupby("market_date")
    hist = grouped.agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"}).astype("float64")
    last_minute = grouped.apply(lambda g: g.index[-1], include_groups=False)
    settle = bd._settle_timestamps(hist.index)
    hist = hist[last_minute.values == settle.values]
    return _calendar(hist)


def _daily_indicators(hist: pd.DataFrame, ma: str) -> pd.DataFrame:
    """Индикаторы по состоянию на конец каждых суток. Сдвиг на день — позже."""
    close, high, low, volume = hist["close"], hist["high"], hist["low"], hist["volume"]
    ret = (close / close.shift(1) - 1) * 100
    out = pd.DataFrame(index=hist.index)

    # --- скользящие: наклон, расстояние до цены и расстояние между собой
    mas = {w: _ma(close, w, ma) for w in MA_WINDOWS}
    for w in MA_WINDOWS:
        out[f"ma{w}_change_pct"] = (mas[w] / mas[w].shift(1) - 1) * 100
        out[f"ma{w}_dist"] = _dist_pct(close, mas[w])
    for short, long in ((7, 30), (30, 100), (7, 100)):
        out[f"ma{short}_vs_ma{long}"] = _dist_pct(mas[short], mas[long])

    # Не «где мы относительно средней», а «приближаемся или отрываемся»: то же
    # расстояние в сравнении с тем, каким оно было несколько дней назад.
    for lag in (3, 7, 14):
        out[f"ma30_dist_delta{lag}"] = out["ma30_dist"] - out["ma30_dist"].shift(lag)

    # --- волатильность и размах
    for w in VOL_WINDOWS:
        out[f"vol{w}"] = ret.rolling(w, min_periods=w).std()
    out["vol_ratio"] = out[f"vol{VOL_WINDOWS[0]}"] / out[f"vol{VOL_WINDOWS[1]}"]

    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr14"] = true_range.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean() / close * 100
    out["range_pct"] = (high - low) / close * 100
    # Где закрылись внутри своей же свечи: 1 — у максимума дня, 0 — у минимума.
    out["close_pos"] = ((close - low) / (high - low)).replace([np.inf, -np.inf], np.nan)

    # --- положение в канале за 20 дней: 1 — на максимуме, 0 — на минимуме
    hi20 = high.rolling(CHANNEL_WINDOW, min_periods=CHANNEL_WINDOW).max()
    lo20 = low.rolling(CHANNEL_WINDOW, min_periods=CHANNEL_WINDOW).min()
    out["channel_pos"] = ((close - lo20) / (hi20 - lo20)).replace([np.inf, -np.inf], np.nan)

    # --- локальная базовая ставка: какая доля последних суток закрылась в плюс
    up = (ret > 0).astype("float64").where(ret.notna())
    out["up_share20"] = up.rolling(UP_SHARE_WINDOW, min_periods=UP_SHARE_WINDOW).mean()
    # ...и вторая половина того же разложения: на сколько в среднем растёт против падает
    for w in ASYM_WINDOWS:
        out[f"asym{w}"] = _asymmetry(ret, w)

    # --- возврат к среднему: перекупленность, короткий канал, длина серии
    for w in RSI_WINDOWS:
        out[f"rsi{w}"] = _rsi(close, w)
    hi_s = high.rolling(SHORT_CHANNEL, min_periods=SHORT_CHANNEL).max()
    lo_s = low.rolling(SHORT_CHANNEL, min_periods=SHORT_CHANNEL).min()
    out[f"channel_pos{SHORT_CHANNEL}"] = (
        (close - lo_s) / (hi_s - lo_s)).replace([np.inf, -np.inf], np.nan)
    out[f"dist_high{SHORT_CHANNEL}"] = _dist_pct(close, hi_s)
    out[f"dist_low{SHORT_CHANNEL}"] = _dist_pct(close, lo_s)
    out["streak"] = _streak(ret)

    # --- объём в стандартных отклонениях: устойчивее, чем изменение в процентах,
    # которое скачет на сотни процентов от одного тихого дня
    mean_v = volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).mean()
    std_v = volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).std()
    out["volume_z30"] = (volume - mean_v) / std_v
    out["volume_change_pct"] = (volume / volume.shift(1) - 1) * 100

    # --- сам день
    out["change_pct"] = ret
    out["y"] = np.where(ret > 0, 1.0, np.where(ret < 0, 0.0, np.nan))
    return out


# --------------------------------------------------------------------------
# Часовые признаки
# --------------------------------------------------------------------------

def _hourly_indicators(hourly: pd.DataFrame) -> pd.DataFrame:
    """Внутридневная структура суток рынка, индекс — market_date этих суток.

    Часовая таблица заканчивается ровно расчётной минуткой полудня ET, поэтому всё
    посчитанное здесь по суткам D-1 известно ровно к старту суток D.
    """
    close = hourly["close"].sort_index()
    full = close.reindex(pd.date_range(close.index[0], close.index[-1], freq="h"))
    hret = (full / full.shift(1) - 1) * 100

    h = hourly.copy()
    h["ret"] = hret.reindex(pd.DatetimeIndex(h.index)).values
    g = h.groupby("market_date")

    out = pd.DataFrame(index=g.size().index)
    out["hours_up_share"] = g["ret"].apply(lambda s: (s > 0).mean())
    out["hour_vol"] = g["ret"].std()
    # В какой час суток пришлись максимум и минимум: 1..24, где 24 — час резолва.
    out["hour_of_high"] = g.apply(
        lambda d: d.loc[d["high"].idxmax(), "hour_in_day"], include_groups=False)
    out["hour_of_low"] = g.apply(
        lambda d: d.loc[d["low"].idxmin(), "hour_in_day"], include_groups=False)

    # Разгон перед самым резолвом: последние N часов суток.
    settle = h[h["is_settle_hour"]].reset_index().set_index("market_date")["hour_end"]
    settle = pd.DatetimeIndex(settle.reindex(out.index))
    last = full.reindex(settle).values
    for n in (3, 6, 12):
        base = full.reindex(settle - pd.Timedelta(hours=n)).values
        out[f"ret{n}h"] = (last / base - 1) * 100
    return out


# --------------------------------------------------------------------------
# Сборка
# --------------------------------------------------------------------------

def add_regime_features(daily: pd.DataFrame, hourly: pd.DataFrame, m1: pd.DataFrame,
                        ma: str = "sma") -> pd.DataFrame:
    """Таблица признаков: строка = рынок суток D, все признаки посчитаны по D-1.

    Возвращает признаки с суффиксом `_prev`, плюс `outcome`/`y` — исход суток D.
    """
    hist = _daily_history(m1)
    ind = _daily_indicators(hist, ma)
    hind = _hourly_indicators(hourly)

    index = pd.DatetimeIndex(daily.index)
    prev_index = index - pd.Timedelta(days=1)

    out = pd.DataFrame(index=index)
    for col in ind.columns:
        out[f"{col}_prev"] = ind[col].reindex(prev_index).values
    for col in hind.columns:
        out[f"{col}_prev"] = hind[col].reindex(prev_index).values

    # День недели известен заранее и утечкой не является.
    out["dow"] = index.dayofweek

    out["outcome"] = daily["outcome"].values
    out[TARGET] = daily[TARGET].values
    out.index.name = "market_date"

    out = out.dropna()
    out[TARGET] = out[TARGET].astype("int8")
    out["y_prev"] = out["y_prev"].astype("int8")
    return out


def feature_names(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in NON_FEATURES]


def make_xy(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_names(feats)
    return feats[cols].astype("float64"), feats[TARGET].astype("int8")


def build_features(start=None, end=None, ma: str = "sma",
                   cache_dir: Path | str = bd.DEFAULT_CACHE,
                   workers: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Грузит свечи с запасом истории и возвращает признаки режима с `start`.

    start — дата первых суток рынка в итоговой таблице. История берётся на
    WARMUP_DAYS раньше, иначе первые сто строк отвалились бы из-за длинной скользящей.
    """
    first = None if start is None else pd.Timestamp(start).normalize()
    load_start = None if first is None else first - pd.Timedelta(days=WARMUP_DAYS)

    daily, hourly, m1 = bd.build_dataset(start=load_start, end=end, cache_dir=cache_dir,
                                         workers=workers, verbose=verbose)
    feats = add_regime_features(daily, hourly, m1, ma=ma)
    if first is not None:
        lost = (feats.index[0] - first).days if feats.index[0] > first else 0
        if lost > 0:
            print(f"  ВНИМАНИЕ: не хватило разгона, потеряно {lost} дней в начале "
                  f"выборки — увеличьте WARMUP_DAYS (сейчас {WARMUP_DAYS})")
        feats = feats[feats.index >= first]
    if verbose:
        print(f"  признаков: {len(feature_names(feats))}, строк: {len(feats):,} "
              f"({feats.index[0]:%Y-%m-%d} → {feats.index[-1]:%Y-%m-%d}), скользящие: {ma.upper()}")
    return feats
