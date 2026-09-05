"""Прогноз на один часовой рынок Polymarket. Наружу — ровно одна функция.

    from kit.predict import predict
    ans = predict(minutes_df, "2026-09-04 15:00")
    # {'p_up': 0.5513..., 'side': 'Up', 'model_id': 'ca2589be653a',
    #  'decision_time_utc': '2026-09-04T14:58:00+00:00',
    #  'feature_hour_utc': '2026-09-04T14:00:00+00:00'}

Всё остальное в этом модуле — внутренности, с подчёркиванием в начале имени.
Так задумано: чем меньше способов позвать модель, тем меньше способов позвать
её неправильно.

Что именно предсказывается: закроется ли часовая свеча Binance BTCUSDT за час H
НЕ НИЖЕ своего открытия (close >= open, равенство считается за Up). Решение
принимается в H минус 2 минуты по свече H-1, собранной из минуток :00..:57.

Функция сама отрезает всё, что позже момента решения. Передать минуток больше,
чем нужно, безопасно; заглянуть в будущее — невозможно.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kit import _featload as _fl
from kit import candles as _candles

__all__ = ["predict"]

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_TXT = _ROOT / "model" / "poly_hourly.txt"
_MODEL_JSON = _ROOT / "model" / "poly_hourly.json"

#: Жёсткий минимум истории в часах. Меньше — исключение, а не приблизительный ответ.
#:
#: Откуда число. Самому длинному окну признаков (z168, std168) нужно 169 свечей.
#: Сверх того streak_signed формально смотрит на всю историю сразу — практически
#: же самая длинная серия одинаковых по знаку свечей за 7 лет равна 15. Замер на
#: 800 случайных часах периода оценки: начиная со 169 часов истории вероятность
#: совпадает с расчётом по полной семилетней истории ТОЧНО, до последнего бита,
#: и ни разу не меняет сторону. 240 часов взяты с запасом к этому порогу.
_MIN_HISTORY_HOURS = 240

#: Рекомендованный запас. Больше — не вредно: лишнее функция отрежет сама.
_RECOMMENDED_HISTORY_HOURS = 720

#: Сколько минуток должно быть в часе признаков при обрезке на 2 минуты.
_MINUTES_PER_TRUNCATED_HOUR = 60 - _candles.CUTOFF_MINUTES

#: Насколько неполным может быть час признаков. Час, собранный из половины
#: минуток, — уже не та свеча, на которой училась модель.
_MIN_MINUTES_IN_FEATURE_HOUR = 50

_HOUR = pd.Timedelta(hours=1)
_MINUTE = pd.Timedelta(minutes=1)

_model_cache: dict | None = None


# --------------------------------------------------------------------------
# модель
# --------------------------------------------------------------------------

def _load_model() -> dict:
    """Бустер и его паспорт. Читается один раз за процесс.

    Контрольная сумма файла модели сверяется с паспортом ВСЕГДА. Не совпала —
    считать нельзя: это другая модель, и измеренные числа к ней не относятся.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    for path, what in ((_MODEL_TXT, "файл модели"), (_MODEL_JSON, "паспорт модели")):
        if not path.exists():
            raise RuntimeError(
                f"Не найден {what}: {path}\n"
                "В комплекте оба файла лежат в папке model/. Если вы меняли модель, "
                "положите туда оба сразу — и .txt, и .json."
            )

    meta = json.loads(_MODEL_JSON.read_text(encoding="utf-8"))
    want = meta.get("sha256")
    got = hashlib.sha256(_MODEL_TXT.read_bytes()).hexdigest()
    if want != got:
        raise RuntimeError(
            "Файл model/poly_hourly.txt не соответствует своему паспорту.\n"
            f"  ожидался sha256: {want}\n"
            f"  получен  sha256: {got}\n"
            "Считать по нему нельзя: либо файл повреждён, либо .txt и .json взяты "
            "от разных версий модели. Положите в model/ оба файла из одного комплекта."
        )

    # Импорт здесь, а не наверху: если lightgbm не той версии или вовсе не стоит,
    # ошибка всплывёт рядом с моделью, а не при импорте пакета.
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(_MODEL_TXT))

    passport_features = list(meta.get("features") or [])
    booster_features = list(booster.feature_name())
    if passport_features != _fl.FEATURES or booster_features != _fl.FEATURES:
        raise RuntimeError(
            "Списки признаков разошлись: модель ждёт не то, что считает kit/features.py.\n"
            f"  kit/features.py: {_fl.FEATURES}\n"
            f"  паспорт модели:  {passport_features}\n"
            f"  файл модели:     {booster_features}\n"
            "Так бывает, когда заменили модель, а файл признаков — нет (или наоборот)."
        )

    _model_cache = {"booster": booster, "meta": meta,
                    "model_id": meta.get("model_id", "?")}
    return _model_cache


# --------------------------------------------------------------------------
# разбор аргументов
# --------------------------------------------------------------------------

def _as_hour(value) -> pd.Timestamp:
    """Приводит переданный час рынка к tz-aware UTC и требует ровного начала часа."""
    if not isinstance(value, (str, datetime, pd.Timestamp, np.datetime64)):
        raise TypeError(
            f"Час рынка передан типом {type(value).__name__}, а нужен str, datetime "
            "или pandas.Timestamp. Например: predict(df, '2026-09-04 15:00')."
        )
    try:
        ts = pd.Timestamp(value)
    except Exception as e:
        raise ValueError(
            f"Не удалось разобрать час рынка {value!r}: {e}\n"
            "Ожидается строка вида '2026-09-04 15:00', datetime или pandas.Timestamp."
        ) from e

    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if (ts.minute, ts.second, ts.microsecond, ts.nanosecond) != (0, 0, 0, 0):
        raise ValueError(
            f"Час рынка должен быть ровным началом часа, передано {ts.isoformat()}.\n"
            "Это метка рынка — час H, чей исход предсказываем, — а не момент решения. "
            f"Для этого часа решение принимается в {(ts - 2 * _MINUTE).isoformat()}."
        )
    return ts


def _as_minutes(df) -> pd.DataFrame:
    """Проверяет минутные бары и приводит индекс к tz-aware UTC."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Минутные бары переданы типом {type(df).__name__}, а нужен "
            "pandas.DataFrame с индексом-временем и колонками "
            + ", ".join(_candles.REQUIRED_MINUTE_COLS) + "."
        )
    if df.empty:
        raise ValueError("Минутные бары пусты: считать не по чему.")

    missing = [c for c in _candles.REQUIRED_MINUTE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "В минутных барах нет колонок: " + ", ".join(missing) + ".\n"
            "Нужны: " + ", ".join(_candles.REQUIRED_MINUTE_COLS) + ". Колонка trades "
            "(число сделок внутри минутки) обязательна: по ней считается признак "
            "trrel. У Binance это девятое поле ответа /api/v3/klines."
        )

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(
            f"Индекс минутных баров должен быть DatetimeIndex, а он "
            f"{type(idx).__name__}.\nИндекс — время ОТКРЫТИЯ минутки в UTC."
        )

    out = df.copy()
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if out.index.has_duplicates:
        n_dup = int(out.index.duplicated().sum())
        sample = ", ".join(str(x) for x in out.index[out.index.duplicated()].unique()[:5])
        raise ValueError(
            f"В минутных барах {n_dup} повторяющихся меток времени, например: "
            f"{sample}.\nУберите дубликаты перед вызовом: одна минутка — одна строка."
        )
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out


# --------------------------------------------------------------------------
# проверки достаточности данных
# --------------------------------------------------------------------------

def _check_history(minutes: pd.DataFrame, hour: pd.Timestamp,
                   decision: pd.Timestamp, feature_hour: pd.Timestamp) -> None:
    """Хватает ли истории и нет ли в ней дыр.

    Три разные беды — три разных текста. Каждое сообщение говорит, чего именно
    не хватает, чтобы не пришлось гадать.
    """
    if minutes.empty:
        raise ValueError(
            f"После отсечения по моменту решения ({decision.isoformat()}) минуток "
            "не осталось совсем: все переданные бары — позже момента решения.\n"
            "Проверьте, что час рынка передан правильно и что данные не из будущего."
        )

    need_from = hour - pd.Timedelta(hours=_MIN_HISTORY_HOURS)
    first, last = minutes.index[0], minutes.index[-1]

    if first > need_from:
        have_hours = (decision - first) / _HOUR
        raise ValueError(
            "Не хватает истории.\n"
            f"  нужны минутки начиная с: {need_from.isoformat()} "
            f"(это {_MIN_HISTORY_HOURS} ч до часа рынка)\n"
            f"  самая ранняя переданная: {first.isoformat()}\n"
            f"  то есть на руках примерно {have_hours:.1f} ч из {_MIN_HISTORY_HOURS}.\n"
            f"Рекомендуемый запас — {_RECOMMENDED_HISTORY_HOURS} ч "
            f"({_RECOMMENDED_HISTORY_HOURS // 24} суток): лишнее функция отрежет сама."
        )

    if last < feature_hour:
        raise ValueError(
            "Данные обрываются раньше часа признаков.\n"
            f"  час признаков:                {feature_hour.isoformat()}\n"
            f"  последняя минутка до решения: {last.isoformat()}\n"
            "Свечу H-1 собирать не из чего."
        )

    window = minutes[minutes.index >= need_from]
    counts = _candles.minute_counts(window, cutoff=_candles.CUTOFF_MINUTES)
    grid = pd.date_range(need_from, feature_hour, freq="1h")
    holes = [h for h in grid if int(counts.get(h, 0)) == 0]
    if holes:
        shown = ", ".join(h.strftime("%Y-%m-%d %H:00") for h in holes[:10])
        more = f" и ещё {len(holes) - 10}" if len(holes) > 10 else ""
        raise ValueError(
            f"Дыры в минутках: {len(holes)} час(ов) в окне истории без единой минутки.\n"
            f"  {shown}{more}\n"
            "Час, которого нет вовсе, сдвигает все скользящие окна признаков. "
            "Догрузите пропущенное или начните окно раньше."
        )

    got = int(counts.get(feature_hour, 0))
    if got < _MIN_MINUTES_IN_FEATURE_HOUR:
        in_hour = window[(window.index >= feature_hour)
                         & (window.index < feature_hour + _HOUR)]
        last_min = in_hour.index[-1].strftime(":%M") if len(in_hour) else "ни одной"
        raise ValueError(
            f"Час признаков {feature_hour.isoformat()} собран из {got} минуток "
            f"из {_MINUTES_PER_TRUNCATED_HOUR} (последняя {last_min}).\n"
            f"Минимум — {_MIN_MINUTES_IN_FEATURE_HOUR}. По такому огрызку считать "
            "нельзя: свеча H-1 получится не та, на которой училась модель."
        )


# --------------------------------------------------------------------------
# расчёт
# --------------------------------------------------------------------------

def _features_row(minutes: pd.DataFrame, hour: pd.Timestamp) -> tuple[pd.Series, float]:
    """Строка из 22 признаков и вероятность Up. Внутренняя кухня predict().

    Отдельной функцией — чтобы самопроверка могла показать, ГДЕ разошлось:
    в сборке свечей или уже в модели. Публичным интерфейсом не является.
    """
    decision = hour - _candles.CUTOFF_MINUTES * _MINUTE
    feature_hour = hour - _HOUR

    # Отсечение по моменту решения. Делается здесь и всегда: последние две
    # минуты часа H-1 не должны попасть в признаки никаким путём.
    minutes = minutes[minutes.index < decision]
    _check_history(minutes, hour, decision, feature_hour)

    trunc = _candles.hourly(minutes, cutoff=_candles.CUTOFF_MINUTES)
    built = _fl.build(trunc, "1h")

    last_time = pd.Timestamp(built["df"]["open_time"].iloc[-1])
    if last_time != feature_hour:
        raise RuntimeError(
            "Последняя собранная свеча — не та, что нужна.\n"
            f"  ожидалась: {feature_hour.isoformat()} (час H-1)\n"
            f"  получена:  {last_time.isoformat()}\n"
            "Обычно это значит, что в минутках нет данных за час H-1."
        )

    row = built["X"].iloc[-1][_fl.FEATURES]
    bad = [name for name, v in row.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(
            "Признаки посчитались не полностью, нечисловые значения: "
            + ", ".join(bad) + ".\n"
            "Обычная причина — нулевой объём или ноль сделок в одном из предыдущих "
            "24 часов (простой биржи). Такой час пропускают, а не предсказывают наугад."
        )

    booster = _load_model()["booster"]
    p_up = float(booster.predict(row.to_numpy(dtype="float64").reshape(1, -1))[0])
    return row, p_up


def predict(minutes_df: pd.DataFrame, hour_start_utc) -> dict:
    """Прогноз на часовой рынок Polymarket «Bitcoin Up or Down».

    Аргументы
    ---------
    minutes_df : pandas.DataFrame
        Минутные бары Binance BTC/USDT. Индекс — время ОТКРЫТИЯ минутки
        (UTC; наивное время трактуется как UTC). Обязательные колонки:
        open, high, low, close, volume, trades.
        Истории нужно не меньше 240 часов до часа рынка, рекомендуется 720.
        Лишнее по времени функция отрезает сама, поэтому передавать «с запасом»
        безопасно и правильно.
    hour_start_utc : str | datetime | pandas.Timestamp
        Час рынка H — тот, чей исход предсказываем. Ровное начало часа.
        Решение по нему принимается в H минус 2 минуты.

    Возвращает
    ----------
    dict с полями:
        p_up               вероятность того, что close >= open часа H
        side               'Up' если p_up > 0.5, иначе 'Down'
        model_id           отпечаток модели из её паспорта
        decision_time_utc  момент решения, ISO-8601 (H минус 2 минуты)
        feature_hour_utc   час свечи признаков, ISO-8601 (H минус 1 час)

    Бросает исключение
    ------------------
    ValueError, TypeError, RuntimeError — если данных не хватает, если в них
    дыры или если файлы модели не сходятся с паспортом. Приблизительного ответа
    функция не возвращает никогда: пропустить час дешевле, чем поставить вслепую.
    """
    hour = _as_hour(hour_start_utc)
    minutes = _as_minutes(minutes_df)
    _, p_up = _features_row(minutes, hour)
    return {
        "p_up": p_up,
        "side": "Up" if p_up > 0.5 else "Down",
        "model_id": _load_model()["model_id"],
        "decision_time_utc": (hour - _candles.CUTOFF_MINUTES * _MINUTE).isoformat(),
        "feature_hour_utc": (hour - _HOUR).isoformat(),
    }
