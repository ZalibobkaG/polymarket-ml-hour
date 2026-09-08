"""Часовые свечи из минуток, в том числе обрезанные моментом решения.

Почему свечи собираются, а не берутся у биржи готовыми. Решение принимается за
2 минуты до начала часа H. В этот момент свеча H-1 известна не целиком: минутки
[:00 ... :57] уже прожиты, а [:58] и [:59] — ещё нет. Готовая часовая свеча
Binance содержит все 60 минут, то есть подглядывает в будущее относительно
момента решения. Поэтому вся история признаков строится из одинаково обрезанных
свечей — ряда, который целиком доступен в момент решения.

Обрезка — часть модели, а не настройка. Менять `CUTOFF_MINUTES` нельзя: модель
обучена на свечах из минуток :00..:57, и на любых других она считает другое.
"""
from __future__ import annotations

import pandas as pd

#: Сколько последних минут часа отбрасывается. Ровно эта обрезка была при обучении.
CUTOFF_MINUTES = 2

#: Колонки готовой свечи. Ровно их берёт расчёт признаков.
CANDLE_COLS = ["open_time", "open", "high", "low", "close", "volume", "trades"]

#: Колонки, которые обязаны быть у минуток на входе.
REQUIRED_MINUTE_COLS = ["open", "high", "low", "close", "volume", "trades"]

PRICE_COLS = ["open", "high", "low", "close"]

#: Типы минутных колонок, в которых считалось обучение.
#:
#: Это не косметика. Объём при обучении хранился во float32, и сумма 60 значений
#: во float32 отличается от суммы тех же значений во float64 в 7-8 знаке. Через
#: признак `volrel` эта мелочь доходит до модели. Поэтому колонки приводятся
#: к обучающим типам ДО агрегации, и результат перестаёт зависеть от того,
#: в каком виде минутки пришли от вашей инфраструктуры.
MINUTE_DTYPES = {"open": "float64", "high": "float64", "low": "float64",
                 "close": "float64", "volume": "float32", "trades": "float64"}


def hourly(minutes: pd.DataFrame, cutoff: int = CUTOFF_MINUTES) -> pd.DataFrame:
    """Часовые свечи из минуток. `cutoff` — сколько последних минут часа отбросить.

    На входе: DataFrame минутных баров с индексом-временем (UTC, tz-aware) и
    колонками open/high/low/close/volume/trades.

    На выходе: DataFrame с колонками CANDLE_COLS. Метка свечи — время её начала
    (HH:00), как у Binance. Часы, в которых не оказалось ни одной минутки,
    из результата выпадают — дырку лучше увидеть отсутствующей строкой, чем
    строкой из NaN.

    `trades` суммируется наравне с объёмом: этого числа требует признак `trrel`.
    Сумма целых после группировки приезжает как float64 (пустые часы дают NaN
    до отбрасывания), поэтому округляется и приводится к целому — округление
    здесь только убирает хвост представления, числа и так целые.
    """
    missing = [c for c in REQUIRED_MINUTE_COLS if c not in minutes.columns]
    if missing:
        raise ValueError(
            "В минутных барах нет обязательных колонок: " + ", ".join(missing) + ".\n"
            "Нужны: " + ", ".join(REQUIRED_MINUTE_COLS) + ". Колонка `trades` "
            "(число сделок внутри минутки) обязательна — по ней считается признак "
            "`trrel`, и без неё модель работать не может."
        )
    if not isinstance(cutoff, int) or not 0 <= cutoff <= 59:
        raise ValueError(f"cutoff должен быть целым от 0 до 59, передано {cutoff!r}.")

    m = minutes[REQUIRED_MINUTE_COLS]
    if cutoff:
        m = m[m.index.minute <= (59 - cutoff)]
    m = m.astype(MINUTE_DTYPES)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "trades": "sum"}
    out = m.resample("1h", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["close"]).reset_index()
    out = out.rename(columns={out.columns[0]: "open_time"})

    out["trades"] = out["trades"].round().astype("Int64")
    for c in PRICE_COLS + ["volume"]:
        out[c] = out[c].astype("float64")
    return out[CANDLE_COLS].sort_values("open_time").reset_index(drop=True)


def minute_counts(minutes: pd.DataFrame, cutoff: int = CUTOFF_MINUTES) -> pd.Series:
    """Сколько минуток попало в каждый час. Индекс — начало часа.

    Нужно проверкам целостности: час, собранный из половины минуток, — это не
    та свеча, на которой училась модель.
    """
    m = minutes[minutes.index.minute <= (59 - cutoff)] if cutoff else minutes
    return m.resample("1h", label="left", closed="left").size()
