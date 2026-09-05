"""Минутные свечи Binance BTC/USDT: загрузка с кэшем на диск.

Нужны только для переобучения. Боевому прогнозу (`kit/predict.py`) этот модуль
не нужен вовсе: там минутки приходят снаружи, из вашей инфраструктуры.

Откуда берутся данные, в порядке убывания предпочтения:

1.  Месячные архивы `data.binance.vision` — по одному zip на месяц. Самый
    дешёвый способ: один запрос на 40 тысяч минуток.
2.  Дневные архивы того же хранилища — для текущего, ещё не закрытого месяца.
3.  REST `api.binance.com/api/v3/klines` — самый хвост, за который архивов
    ещё не выложили.

Ключи API не нужны ни на одном из трёх уровней: всё публичное.

Кэш — parquet, по файлу на архив:

    <cache_dir>/monthly/BTCUSDT-1m-2026-08.parquet
    <cache_dir>/daily/BTCUSDT-1m-2026-09-01.parquet

Имена файлов и типы колонок совпадают с теми, что использовались при обучении
модели, поэтому готовый кэш можно просто скопировать в `--cache-dir` и ничего
не перекачивать. Полная история с 2019 года — около 210 МБ.

Хвост из REST не кэшируется никогда, и это правильно: последние минуты ещё
могут дозаписаться.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

UTC = timezone.utc

SYMBOL = "BTCUSDT"
VISION_URL = "https://data.binance.vision/data/spot"
REST_URL = "https://api.binance.com/api/v3/klines"

#: Раньше этой даты на Binance пары BTCUSDT не было.
FIRST_DAY = date(2017, 8, 17)

#: Колонки клайна Binance, как они приходят в архиве и в REST.
RAW_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

#: Что оставляем. `trades` обязателен: по нему считается признак `trrel`.
KEEP_COLS = ["open", "high", "low", "close", "volume",
             "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]

PRICE_COLS = ["open", "high", "low", "close"]


# --------------------------------------------------------------------------
# разбор сырых клайнов
# --------------------------------------------------------------------------

def _timestamp_unit(values: pd.Series) -> str:
    """Binance с 2025 года отдаёт время в микросекундах, раньше — в миллисекундах.

    Разница на глаз не видна, а ошибка в единицах уводит все метки времени на
    тысячи лет. Поэтому единицы определяются по величине самого числа.
    """
    return "us" if float(values.max()) > 1e14 else "ms"


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Типы колонок ровно те же, что были при обучении модели.

    Цены — float64, число сделок — int32, остальные объёмы — float32.
    Менять это не следует: типы попадают в кэш, а из кэша — в признаки.
    """
    for col in df.columns:
        if col in PRICE_COLS:
            df[col] = df[col].astype("float64")
        elif col == "trades":
            df[col] = df[col].fillna(0).astype("int32")
        else:
            df[col] = df[col].astype("float32")
    return df


def _parse_kline_csv(blob: bytes) -> pd.DataFrame:
    """CSV из архива Binance -> DataFrame с UTC-индексом по времени открытия."""
    df = pd.read_csv(io.BytesIO(blob), header=None, names=RAW_COLS)

    # В части архивов первая строка — заголовок. После to_numeric она станет
    # NaN и отвалится сама; без этой строчки разбор падал бы на ней.
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


# --------------------------------------------------------------------------
# загрузка
# --------------------------------------------------------------------------

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "poly-hourly-kit/1.0"
    return s


def _fetch_zip(url: str, session: requests.Session, retries: int = 3) -> bytes | None:
    """Содержимое единственного файла в zip. None — если архива нет (404)."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            resp = session.get(url, timeout=90)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                return zf.read(zf.namelist()[0])
        except Exception as exc:      # сеть или битый архив — пробуем ещё раз
            last = exc
    raise RuntimeError(
        f"Не удалось скачать {url} за {retries} попытки.\n"
        f"Последняя ошибка: {type(last).__name__}: {last}\n"
        "Проверьте доступ в интернет и что data.binance.vision открыт из вашей сети."
    ) from last


def _months(start: date, end: date) -> list[tuple[int, int]]:
    year, month = start.year, start.month
    out: list[tuple[int, int]] = []
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _load_month(ym: tuple[int, int], cache_dir: Path, session: requests.Session,
                symbol: str) -> pd.DataFrame | None:
    year, month = ym
    cache = cache_dir / "monthly" / f"{symbol}-1m-{year:04d}-{month:02d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{VISION_URL}/monthly/klines/{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"
    blob = _fetch_zip(url, session)
    if blob is None:
        return None                   # месяц ещё не закрыт — доберём дневными

    df = _parse_kline_csv(blob)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _load_day(day: date, cache_dir: Path, session: requests.Session,
              symbol: str) -> pd.DataFrame | None:
    cache = cache_dir / "daily" / f"{symbol}-1m-{day.isoformat()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{VISION_URL}/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
    blob = _fetch_zip(url, session)
    if blob is None:
        return None                   # архива ещё нет — доберём через REST

    df = _parse_kline_csv(blob)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _load_rest(start: datetime, end: datetime, session: requests.Session,
               symbol: str) -> pd.DataFrame:
    """Свежие минутки, которых ещё нет в архивах. В кэш не пишутся."""
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
        if nxt <= cursor:             # курсор не двинулся — выходим, а не зацикливаемся
            break
        cursor = nxt

    if not rows:
        return pd.DataFrame(columns=KEEP_COLS,
                            index=pd.DatetimeIndex([], tz=UTC, name="open_time"))
    return _parse_rest_klines(rows)


def _as_datetime(value, default: datetime) -> datetime:
    if value is None:
        return default
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return ts.to_pydatetime()


def load_1m(start=None, end=None, cache_dir: Path | str = "data/cache/binance_1m",
            workers: int = 8, verbose: bool = True, symbol: str = SYMBOL) -> pd.DataFrame:
    """Минутные свечи `symbol` за период [start, end) с UTC-индексом.

    start / end — что угодно, что понимает pandas.Timestamp. Наивное время
    трактуется как UTC. По умолчанию — вся история до текущего момента.

    Возвращает DataFrame с индексом-временем открытия минутки и колонками
    KEEP_COLS. Дубликаты сняты, порядок по возрастанию времени.
    """
    cache_dir = Path(cache_dir)
    start_dt = _as_datetime(start, datetime.combine(FIRST_DAY, datetime.min.time(), UTC))
    end_dt = _as_datetime(end, datetime.now(UTC))
    if end_dt <= start_dt:
        raise ValueError(
            f"Конец периода ({end_dt}) должен быть позже начала ({start_dt})."
        )

    session = _new_session()
    months = _months(start_dt.date(), end_dt.date())
    if verbose:
        print(f"  минутки {symbol}: {start_dt:%Y-%m-%d %H:%M} -> {end_dt:%Y-%m-%d %H:%M} "
              f"UTC ({len(months)} мес.), кэш {cache_dir}")

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
            print(f"  дневных архивов нет с {min(missing_days)} — добираю через REST")
        parts.append(_load_rest(rest_start, end_dt, session, symbol))

    if not parts:
        raise RuntimeError(
            "Не удалось загрузить ни одной минутной свечи.\n"
            "Проверьте доступ в интернет либо укажите --cache-dir на готовый кэш."
        )

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index >= pd.Timestamp(start_dt)) & (df.index < pd.Timestamp(end_dt))]

    # Текущая минутка ещё не закрыта: её OHLCV продолжает меняться. Держать её
    # в данных нельзя — признаки по ней получатся не те, что будут через минуту.
    df = df[df.index + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz=UTC)]

    if verbose:
        print(f"  готово: {len(df):,} минуток, "
              f"{df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M} UTC"
              .replace(",", " "))
    return df
