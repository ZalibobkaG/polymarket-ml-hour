"""Свечи BTC/USDT с Binance, пересобранные в сетку времени Polymarket.

Polymarket-рынки "BTC Up or Down" резолвятся так:

    Рынок резолвится в "Up", если Close 1-минутной свечи Binance BTC/USDT
    за <день D-1> 12:00 ET (полдень) НИЖЕ, чем финальный Close свечи
    за <день D> 12:00 ET. В "Down" — если выше. Если Close равны — 50/50.

Из этого следует два неочевидных момента, которые здесь и реализованы:

1.  Точка отсчёта — не 12:00 ET, а 12:01 ET. Резолвит *закрытие* свечи,
    открытой в 12:00, то есть цена в момент 12:00:59.999.
    Значит "день" рынка = минутные свечи с 12:01 ET (D-1) по 12:00 ET (D)
    включительно, а не с 12:00 по 11:59.

2.  ET — это America/New_York с переводом часов. В дни перехода на летнее/
    зимнее время сутки рынка длятся 23 или 25 часов, а не 24. Здесь они так
    и остаются — граница привязана к полудню по нью-йоркским часам, а не к
    фиксированному сдвигу от UTC.

Часовые свечи собираются по той же сетке со сдвигом в минуту (HH:01 → HH+1:00),
поэтому ровно 24 часовые свечи укладываются в одни сутки рынка, а последняя
из них закрывается той самой минутой, по которой считается резолв.

Строка обеих таблиц — это свеча, которую предсказываем: её OHLC и её исход
(outcome/y). Признаки — всё, что с суффиксом `_prev`: они описывают предыдущую,
уже закрытую свечу, то есть единственное, что известно к началу этой.

Внимание: OHLC самой свечи в признаки брать нельзя. `y` — это в точности знак
`close` минус `close` предыдущей свечи, так что любая модель, увидевшая `close`,
получит 100% точности и ноль смысла. Эти колонки — справочные, чтобы таблицу можно
было глазами сверить с биржей.

Данные берутся из архивов data.binance.vision (месячные и дневные zip),
хвост последних суток — из REST API. Всё кэшируется в parquet.

Модуль назван по BTC, но к нему не привязан: `load_1m` и `build_dataset` принимают
`symbol`, а кэш разложен по именам символов. Сетка времени, разметка исхода и правила
про неполные свечи от инструмента не зависят — у ETH тот же рынок с тем же резолвом.

Пример:
    >>> import btc_data
    >>> m1 = btc_data.load_1m(start="2024-01-01")
    >>> eth = btc_data.load_1m(start="2024-01-01", symbol="ETHUSDT")
    >>> daily = btc_data.build_daily(m1)
    >>> hourly = btc_data.build_hourly(m1)
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Константы
# --------------------------------------------------------------------------

#: Символ по умолчанию. Всё ниже принимает `symbol` параметром — кэш и архивы
#: разложены по именам символов, поэтому BTC и ETH не мешают друг другу.
SYMBOL = "BTCUSDT"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

#: Час ET, в котором открывается расчётная минутная свеча Polymarket.
SETTLE_HOUR = 12
#: Минута ET, в которую эта свеча открывается (закрывается она в SETTLE_MINUTE+1).
SETTLE_MINUTE = 0

VISION_URL = "https://data.binance.vision/data/spot"
REST_URL = "https://api.binance.com/api/v3/klines"

#: Первый день торгов BTC/USDT на Binance.
FIRST_DAY = date(2017, 8, 17)

DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "cache"

RAW_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KEEP_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
]
PRICE_COLS = ["open", "high", "low", "close"]

# Агрегируем только то, что идёт в итоговые таблицы. Остальные поля клайнов
# (quote_volume, trades, taker_buy_*) лежат в кэше минуток и доступны, если позже
# понадобятся под новые признаки.
_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

#: Длины скользящих средних по Close, изменение которых идёт в признаки.
MA_WINDOWS = (14, 30)


# --------------------------------------------------------------------------
# Разбор сырых клайнов
# --------------------------------------------------------------------------

def _timestamp_unit(values: pd.Series) -> str:
    """Binance с 2025 года отдаёт время в микросекундах, раньше — в миллисекундах."""
    return "us" if float(values.max()) > 1e14 else "ms"


def _parse_kline_csv(blob: bytes) -> pd.DataFrame:
    """CSV из архива Binance → DataFrame с UTC-индексом по времени открытия."""
    df = pd.read_csv(io.BytesIO(blob), header=None, names=RAW_COLS)

    # В части архивов первая строка — заголовок; она станет NaN и отвалится.
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[df["open_time"].notna()]

    idx = pd.to_datetime(df["open_time"], unit=_timestamp_unit(df["open_time"]), utc=True)
    out = df[KEEP_COLS].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(idx, name="open_time")
    return _downcast(out)


def _parse_rest_klines(rows: Sequence[Sequence]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RAW_COLS)
    df["open_time"] = pd.to_numeric(df["open_time"])
    idx = pd.to_datetime(df["open_time"], unit=_timestamp_unit(df["open_time"]), utc=True)
    out = df[KEEP_COLS].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(idx, name="open_time")
    return _downcast(out)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Цены держим в float64, объёмы — в float32: 5 млн минут иначе занимают лишнее."""
    for col in df.columns:
        if col in PRICE_COLS:
            df[col] = df[col].astype("float64")
        elif col == "trades":
            df[col] = df[col].fillna(0).astype("int32")
        else:
            df[col] = df[col].astype("float32")
    return df


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "btc-data/1.0"
    return s


def _fetch_zip(url: str, session: requests.Session, retries: int = 3) -> bytes | None:
    """Возвращает содержимое единственного файла в zip, либо None если архива нет."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            resp = session.get(url, timeout=90)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                return zf.read(zf.namelist()[0])
        except Exception as exc:  # сеть или битый архив — пробуем ещё раз
            last = exc
    raise RuntimeError(f"не удалось скачать {url}") from last


def _months(start: date, end: date) -> list[tuple[int, int]]:
    year, month = start.year, start.month
    out: list[tuple[int, int]] = []
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _load_month(ym: tuple[int, int], cache_dir: Path, session: requests.Session,
                symbol: str = SYMBOL) -> pd.DataFrame | None:
    year, month = ym
    cache = cache_dir / "monthly" / f"{symbol}-1m-{year:04d}-{month:02d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{VISION_URL}/monthly/klines/{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"
    blob = _fetch_zip(url, session)
    if blob is None:
        return None  # месяц ещё не закрыт — доберём дневными архивами

    df = _parse_kline_csv(blob)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _load_day(day: date, cache_dir: Path, session: requests.Session,
              symbol: str = SYMBOL) -> pd.DataFrame | None:
    cache = cache_dir / "daily" / f"{symbol}-1m-{day.isoformat()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{VISION_URL}/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
    blob = _fetch_zip(url, session)
    if blob is None:
        return None  # архив ещё не выложен — доберём через REST

    df = _parse_kline_csv(blob)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _load_rest(start: datetime, end: datetime, session: requests.Session,
               symbol: str = SYMBOL) -> pd.DataFrame:
    """Свежие минуты, которых ещё нет в архивах."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[Sequence] = []
    cursor = start_ms

    while cursor < end_ms:
        resp = session.get(
            REST_URL,
            params={"symbol": symbol, "interval": "1m", "startTime": cursor,
                    "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 60_000
        if _timestamp_unit(pd.Series([batch[-1][0]])) == "us":
            nxt = int(batch[-1][0]) // 1000 + 60_000
        if nxt <= cursor:
            break
        cursor = nxt

    if not rows:
        return pd.DataFrame(columns=KEEP_COLS, index=pd.DatetimeIndex([], tz=UTC, name="open_time"))
    return _parse_rest_klines(rows)


def _as_datetime(value, default: datetime) -> datetime:
    if value is None:
        return default
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return ts.to_pydatetime()


def load_1m(
    start=None,
    end=None,
    cache_dir: Path | str = DEFAULT_CACHE,
    workers: int = 8,
    verbose: bool = True,
    symbol: str = SYMBOL,
) -> pd.DataFrame:
    """Минутные свечи `symbol` за период [start, end) с UTC-индексом.

    start/end — что угодно, что понимает pd.Timestamp (строка, date, datetime).
    Наивное время трактуется как UTC. По умолчанию: вся история до текущего момента.
    """
    cache_dir = Path(cache_dir)
    start_dt = _as_datetime(start, datetime.combine(FIRST_DAY, datetime.min.time(), UTC))
    end_dt = _as_datetime(end, datetime.now(UTC))
    if end_dt <= start_dt:
        raise ValueError(f"end ({end_dt}) должен быть позже start ({start_dt})")

    session = _new_session()
    months = _months(start_dt.date(), end_dt.date())
    if verbose:
        print(f"1m {symbol}: {start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M} UTC "
              f"({len(months)} мес.)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        monthly = list(pool.map(
            lambda ym: _load_month(ym, cache_dir, session, symbol), months))

    parts = [df for df in monthly if df is not None]
    open_months = [ym for ym, df in zip(months, monthly) if df is None]

    # Незакрытые месяцы добираем дневными архивами.
    pending_days: list[date] = []
    for year, month in open_months:
        day = date(year, month, 1)
        while day.month == month and day <= end_dt.date():
            if day >= start_dt.date():
                pending_days.append(day)
            day += timedelta(days=1)

    if pending_days:
        if verbose:
            print(f"  месячных архивов нет за {len(open_months)} мес. — "
                  f"качаю {len(pending_days)} дневных")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            daily = list(pool.map(
                lambda d: _load_day(d, cache_dir, session, symbol), pending_days))
        parts.extend(df for df in daily if df is not None)
        missing_days = [d for d, df in zip(pending_days, daily) if df is None]
    else:
        missing_days = []

    # Самый хвост (сегодня/вчера) — только через REST.
    if missing_days:
        rest_start = datetime.combine(min(missing_days), datetime.min.time(), UTC)
        if verbose:
            print(f"  дневных архивов нет с {min(missing_days)} — добираю через REST API")
        parts.append(_load_rest(rest_start, end_dt, session, symbol))

    if not parts:
        raise RuntimeError("не удалось загрузить ни одной свечи")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index >= pd.Timestamp(start_dt)) & (df.index < pd.Timestamp(end_dt))]

    # Текущую минутку REST отдаёт недоформированной: она закрывается только через
    # минуту после открытия. Держать её в данных нельзя — её OHLCV ещё меняется.
    df = df[df.index + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz=UTC)]

    if verbose:
        print(f"  готово: {len(df):,} минутных свечей, "
              f"{df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC")
    return df


# --------------------------------------------------------------------------
# Сетка времени Polymarket
# --------------------------------------------------------------------------

def add_market_time(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к минуткам ключи сетки Polymarket.

    market_date — сутки рынка: свечи с 12:01 ET предыдущего дня по 12:00 ET
                  этого дня включительно. Резолв рынка помечен этой датой.
    hour_end     — конец часовой корзины (HH:01 → HH+1:00), UTC.
    """
    if df.index.tz is None:
        raise ValueError("ожидается индекс с таймзоной UTC")

    out = df.copy()
    et = out.index.tz_convert(ET)

    # Минута 12:00 ET ещё принадлежит текущим суткам рынка (её close и есть резолв),
    # а 12:01 — уже следующим.
    after_settle = (et.hour > SETTLE_HOUR) | (
        (et.hour == SETTLE_HOUR) & (et.minute > SETTLE_MINUTE)
    )
    day = pd.to_datetime(pd.Index(et.date))  # наивные даты ET, без сдвигов DST
    out["market_date"] = day + pd.to_timedelta(after_settle.astype("int8"), unit="D")

    # Сдвиг ET от UTC всегда кратен часу, поэтому округление вверх можно делать в UTC.
    out["hour_end"] = out.index.ceil("h")
    return out


def _settle_timestamps(market_dates: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    """UTC-момент закрытия расчётной свечи (12:00:59.999 ET) для каждых суток рынка."""
    naive = pd.DatetimeIndex(market_dates) + pd.Timedelta(hours=SETTLE_HOUR, minutes=SETTLE_MINUTE)
    return naive.tz_localize(ET).tz_convert(UTC)


def _outcome(diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Разметка свечи по правилу Polymarket: Up / Down / 50-50 и то же числом.

    diff — Close свечи минус Close предыдущей. NaN (предыдущей свечи нет в данных)
    даёт None/NaN, а не ложное '50-50'.
    """
    outcome = np.where(diff > 0, "Up", np.where(diff < 0, "Down", "50-50"))
    y = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, np.nan))
    unknown = pd.isna(diff)
    return np.where(unknown, None, outcome), np.where(unknown, np.nan, y)


def _ma_change_pct(close: pd.Series, window: int, freq: str) -> pd.Series:
    """Изменение SMA(window) по Close к предыдущей свече, %.

    Считается на сплошном календаре: если свеча выпала из данных, окно не должно
    молча съезжать и захватывать более старые цены. В таких местах выходит NaN.
    """
    full = close.reindex(pd.date_range(close.index[0], close.index[-1], freq=freq))
    ma = full.rolling(window, min_periods=window).mean()
    return (ma / ma.shift(1) - 1) * 100


def build_daily(df1m: pd.DataFrame, drop_partial: bool = True) -> pd.DataFrame:
    """Дневные свечи по сетке Polymarket (12:01 ET → 12:00 ET) с разметкой исхода.

    Строка — это рынок, который предсказываем: индекс `market_date` = дата резолва.

        open … close          — свеча ЭТИХ суток; справочные колонки, в признаки
                                брать нельзя (close и есть ответ)
        outcome / y           — 'Up'/'Down' и 1/0 для ЭТИХ суток, таргет
        change_pct_prev       — на сколько % предыдущие сутки закрылись выше позапрошлых
        outcome_prev / y_prev — как резолвились предыдущие сутки
        volume_change_pct_prev — на сколько % вырос объём предыдущих суток к позапрошлым
        maN_change_pct_prev   — на сколько % сдвинулась SMA(N) по Close на предыдущих сутках

    Признаки — только с суффиксом `_prev`: к моменту старта суток рынка известны
    ровно они.

    В таблицу попадают только целые свечи: незакрытые и обрезанные сутки, а также
    строки без полностью известного прошлого (начало выборки, простои биржи)
    отбрасываются — пропусков в таблице нет. Свечи с резолвом 50-50 (Close совпал
    с предыдущим до цента) тоже выпадают: бинарного таргета у них нет.
    """
    keyed = add_market_time(df1m)
    grouped = keyed.groupby("market_date", sort=True)

    daily = grouped.agg(_AGG)
    minutes = grouped.size()
    last_minute = grouped.apply(lambda g: g.index[-1], include_groups=False)

    # Сколько минут должно быть в сутках: 1440, а при переводе часов 1380 или 1500.
    settle = _settle_timestamps(daily.index)
    prev_settle = _settle_timestamps(pd.DatetimeIndex(daily.index) - pd.Timedelta(days=1))
    expected = (settle.values - prev_settle.values) / np.timedelta64(1, "m")

    if drop_partial:
        # Свеча годится только целой: закрывается расчётной минуткой 12:00 ET и не
        # потеряла ни одной минуты. Проверять одно закрытие мало — так проходят
        # обрезанные первые сутки выборки, у которых OHLCV посчитан по огрызку дня.
        daily = daily[(last_minute.values == settle.values) & (minutes.values == expected)]

    # Календарно-предыдущие сутки рынка. Именно по календарю, а не shift(1): при простое
    # биржи shift сравнил бы Close через дыру и выдал исход рынка, которого не было.
    prev_index = pd.DatetimeIndex(daily.index) - pd.Timedelta(days=1)

    # Изменения — всегда к предыдущим суткам. Нет предыдущих суток в данных → NaN,
    # и строка позже отбрасывается, вместо тихого сравнения через пропуск.
    close_prev = daily["close"].reindex(prev_index).values
    diff = daily["close"].values - close_prev
    daily["change_pct"] = diff / close_prev * 100
    daily["volume_change_pct"] = (
        daily["volume"].astype("float64").values
        / daily["volume"].astype("float64").reindex(prev_index).values - 1
    ) * 100

    daily["outcome"], daily["y"] = _outcome(diff)

    ma_change = {w: _ma_change_pct(daily["close"], w, "D") for w in MA_WINDOWS}

    # Признаки сдвигаем на сутки назад: строка рынка D несёт только закрытые сутки D-1.
    prev = daily.reindex(prev_index)
    out = pd.DataFrame(index=daily.index)
    for col in ("open", "high", "low", "close"):
        out[col] = daily[col].values
    out["outcome"] = daily["outcome"].values
    out["y"] = daily["y"].values
    out["change_pct_prev"] = prev["change_pct"].values
    out["outcome_prev"] = prev["outcome"].values
    out["y_prev"] = prev["y"].values
    out["volume_change_pct_prev"] = prev["volume_change_pct"].values
    for w in MA_WINDOWS:
        out[f"ma{w}_change_pct_prev"] = ma_change[w].reindex(prev_index).values

    out.index.name = "market_date"
    out = out.dropna()
    out["y"] = out["y"].astype("int8")
    out["y_prev"] = out["y_prev"].astype("int8")
    return out


def build_hourly(df1m: pd.DataFrame, drop_partial: bool = True) -> pd.DataFrame:
    """Часовые свечи по той же сетке: HH:01 → HH+1:00 ET включительно.

    Колонки те же, что в дневной таблице, и по тому же принципу: OHLC и outcome/y —
    про сам час, всё с суффиксом `_prev` — про предыдущий час.

    hour_in_day = 1..24, где 24 — час, закрывающийся расчётной минуткой полудня ET.
    В дни перевода часов в сутках 23 или 25 часовых свечей.

    Изредка (за 3 года — 7 раз) Close часа совпадает с предыдущим до цента: это резолв
    50-50, бинарного таргета нет. Такой час выпадает из таблицы, а с ним и следующий,
    которому этот час идёт в признаки.
    """
    keyed = add_market_time(df1m)
    grouped = keyed.groupby(["market_date", "hour_end"], sort=True)

    hourly = grouped.agg(_AGG)
    hourly["minutes"] = grouped.size()
    hourly = hourly.reset_index()

    if drop_partial:
        hourly = hourly[hourly["minutes"] == 60]

    hourly = hourly.set_index("hour_end")

    # Предыдущий час — по метке времени, а не по соседней строке: если час выпал из
    # данных, изменение не должно считаться через дыру.
    prev_index = pd.DatetimeIndex(hourly.index) - pd.Timedelta(hours=1)
    close_prev = hourly["close"].reindex(prev_index).values
    diff = hourly["close"].values - close_prev
    hourly["change_pct"] = diff / close_prev * 100
    hourly["volume_change_pct"] = (
        hourly["volume"].astype("float64").values
        / hourly["volume"].astype("float64").reindex(prev_index).values - 1
    ) * 100
    hourly["outcome"], hourly["y"] = _outcome(diff)

    ma_change = {w: _ma_change_pct(hourly["close"], w, "h") for w in MA_WINDOWS}

    # Номер часа внутри суток рынка — через время, прошедшее от полудня ET предыдущего
    # дня. Не через cumcount: тот сбился бы на сутках с дырами.
    day_start = _settle_timestamps(
        pd.DatetimeIndex(hourly["market_date"]) - pd.Timedelta(days=1)
    )
    hourly["hour_in_day"] = (
        (pd.DatetimeIndex(hourly.index).values - day_start.values) / np.timedelta64(1, "h")
    ).astype("int64")

    hour_et = pd.DatetimeIndex(hourly.index).tz_convert(ET)
    hourly["hour_end_et"] = hour_et
    hourly["is_settle_hour"] = (hour_et.hour == SETTLE_HOUR) & (hour_et.minute == SETTLE_MINUTE)

    prev = hourly.reindex(prev_index)
    out = hourly[[
        "market_date", "hour_in_day", "hour_end_et", "is_settle_hour",
        "open", "high", "low", "close", "outcome", "y",
    ]].copy()
    out["change_pct_prev"] = prev["change_pct"].values
    out["outcome_prev"] = prev["outcome"].values
    out["y_prev"] = prev["y"].values
    out["volume_change_pct_prev"] = prev["volume_change_pct"].values
    for w in MA_WINDOWS:
        out[f"ma{w}_change_pct_prev"] = ma_change[w].reindex(prev_index).values

    out = out.dropna()
    out["y"] = out["y"].astype("int8")
    out["y_prev"] = out["y_prev"].astype("int8")
    return out


# --------------------------------------------------------------------------
# Верхнеуровневый вход и проверки
# --------------------------------------------------------------------------

def build_dataset(
    start=None,
    end=None,
    cache_dir: Path | str = DEFAULT_CACHE,
    workers: int = 8,
    verbose: bool = True,
    symbol: str = SYMBOL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает минутки и возвращает (daily, hourly, m1) по сетке Polymarket.

    start — дата ПЕРВЫХ суток рынка, а не начало выборки минуток. Минутки берутся
    с запасом назад: самой длинной скользящей нужно MA_WINDOWS дней истории до
    предыдущих суток — иначе первые строки вышли бы пустыми.
    """
    first_date = None
    if start is not None:
        ts = pd.Timestamp(start)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(ET).tz_localize(None)
        first_date = ts.normalize()

    warmup = pd.Timedelta(days=max(MA_WINDOWS) + 5)
    load_start = None if first_date is None else first_date - warmup
    m1 = load_1m(start=load_start, end=end, cache_dir=cache_dir, workers=workers,
                 verbose=verbose, symbol=symbol)
    daily = build_daily(m1)
    hourly = build_hourly(m1)

    if first_date is not None:
        daily = daily[daily.index >= first_date]

    # Часы держим ровно по тем суткам, что есть в дневной таблице: иначе часовая
    # уезжает вперёд в ещё не закрытый рынок и таблицы перестают биться друг с другом.
    hourly = hourly[hourly["market_date"].isin(daily.index)]

    if verbose:
        print(f"  дневных свечей: {len(daily):,}   часовых: {len(hourly):,}")
    return daily, hourly, m1


def audit(daily: pd.DataFrame, hourly: pd.DataFrame, df1m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Сводка качества: покрытие календаря, пропуски, баланс классов.

    df1m передавать необязательно; с ним дополнительно проверяется полнота минуток.
    """
    idx = pd.DatetimeIndex(daily.index)
    calendar = pd.date_range(idx[0], idx[-1], freq="D")
    hours_per_day = hourly.groupby("market_date").size()
    up = (daily["outcome"] == "Up")
    down = (daily["outcome"] == "Down")

    rows = [
        ("период", f"{idx[0]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}"),
        ("суток рынка", f"{len(daily):,}"),
        ("пропущено суток", f"{len(calendar.difference(idx)):,}"),
        ("часовых свечей", f"{len(hourly):,}"),
        ("суток с 24 часовыми свечами", f"{int((hours_per_day == 24).sum()):,}"),
        ("суток с 23/25 часами (перевод часов)",
         f"{int(hours_per_day.isin([23, 25]).sum()):,}"),
    ]

    if df1m is not None:
        minutes = add_market_time(df1m).groupby("market_date").size().reindex(idx)
        rows += [
            ("минуток загружено", f"{len(df1m):,}"),
            ("суток с ровно 1440 минутами", f"{int((minutes == 1440).sum()):,}"),
            ("суток с дырами в минутках",
             f"{int((~minutes.isin([1380, 1440, 1500])).sum()):,}"),
        ]

    rows += [
        ("пропусков (NaN) в дневной таблице", f"{int(daily.isna().sum().sum()):,}"),
        ("пропусков (NaN) в часовой таблице", f"{int(hourly.isna().sum().sum()):,}"),
        ("Up", f"{int(up.sum()):,} ({up.mean() * 100:.2f}%)"),
        ("Down", f"{int(down.sum()):,} ({down.mean() * 100:.2f}%)"),
    ]
    return pd.DataFrame(rows, columns=["показатель", "значение"]).set_index("показатель")


def save_dataset(daily: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path | str = None) -> list[Path]:
    """Сохраняет обе таблицы в parquet и csv рядом с модулем (data/)."""
    out_dir = Path(out_dir or Path(__file__).resolve().parent / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in (("btc_daily_polymarket", daily), ("btc_hourly_polymarket", hourly)):
        for suffix, writer in ((".parquet", "to_parquet"), (".csv", "to_csv")):
            path = out_dir / f"{name}{suffix}"
            getattr(df, writer)(path)
            written.append(path)
    return written
