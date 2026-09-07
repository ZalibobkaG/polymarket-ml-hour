"""Минутки для прогноза — доставка сырых данных биржи до `poly_hourly_kit`.

Расчёт признаков и вероятности теперь целиком внутри `poly_hourly_kit/` (комплект
напарника, обученный отдельно и герметичный — его `kit/features.py` нельзя
трогать, контрольная сумма сверяется при каждом импорте). Здесь остаётся ровно
то, что снаружи комплекта: загрузка минуток Binance и проверка их свежести —
что бы ни отдал источник, ничего новее момента решения в расчёт попасть не
должно, а если источник отстал, лучше отказаться от ставки, чем считать по
дырявому ряду.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import config
from app.log import get

ROOT = Path(__file__).resolve().parents[1]
#: Загрузка минуток лежит в `research/` — единственное, что оттуда ещё нужно
#: боевому пути (расчёт признаков теперь целиком в `poly_hourly_kit/`).
sys.path.insert(0, str(ROOT / "research"))

import btc_data as bd  # noqa: E402

log = get("features")


def decision_time(hour_start: pd.Timestamp, cutoff: int | None = None) -> pd.Timestamp:
    """Момент, когда признаки для рынка `hour_start` становятся известны."""
    cutoff = config.CUTOFF_MINUTES if cutoff is None else cutoff
    return pd.Timestamp(hour_start) - pd.Timedelta(minutes=cutoff)


def load_minutes(hour_start: pd.Timestamp, cutoff: int | None = None,
                 days: int = config.HISTORY_DAYS) -> pd.DataFrame:
    """Минутки для расчёта, обрезанные ровно моментом решения.

    Обрезка — главная защита от подглядывания: что бы ни отдал источник, ничего
    новее момента решения в расчёт не попадёт. `poly_hourly_kit.predict()` и сам
    режет по моменту решения, так что двойная обрезка безопасна и правильна.
    """
    cut = decision_time(hour_start, cutoff)
    start = cut - pd.Timedelta(days=days)
    m1 = bd.load_1m(start=start, cache_dir=config.CACHE_DIR, verbose=False)
    return m1[m1.index < cut]


def for_market(hour_start: pd.Timestamp, cutoff: int | None = None) -> pd.DataFrame:
    """Скачать минутки для одного рынка — вход `app.model.predict()`."""
    m1 = load_minutes(hour_start, cutoff)
    if m1.empty:
        raise ValueError("не получены минутки")
    last = m1.index[-1]
    expected = decision_time(hour_start, cutoff) - pd.Timedelta(minutes=1)
    if last < expected:
        # Данные отстают: биржа или сеть не успели. `poly_hourly_kit` эту дыру
        # заметил бы, только если она съест больше 8 минуток часа признаков —
        # здесь же ловим саму задержку, до того как она успеет вырасти.
        raise ValueError(f"минутки отстают: последняя {last:%H:%M}Z, "
                         f"ожидалась {expected:%H:%M}Z")
    return m1


# --------------------------------------------------------------------------
# Самопроверка живого пути
# --------------------------------------------------------------------------

def verify(n: int = 3) -> dict:
    """Смоук-тест живого пути: своя загрузка минуток + прогноз, на последних часах.

    В отличие от `python -m kit.selfcheck` (сверка с эталоном на фиксированных
    данных, допуск ноль) здесь дёргается сегодняшний живой путь целиком: наша
    загрузка минуток из Binance, наш кэш, и только затем — комплект. Ловит то,
    чего фикстура не видит: дыры в кэше, отставание биржи, разъехавшиеся колонки.
    Возвращает ответ `predict()` по самому свежему из проверенных часов.
    """
    from app import model

    now = pd.Timestamp.now(tz="UTC")
    last_full_hour = now.floor("h")
    m1 = load_minutes(last_full_hour)

    ans = None
    for k in range(n, 0, -1):
        hour = last_full_hour - pd.Timedelta(hours=k)
        ans = model.predict(m1, hour)
        log.debug("selfcheck {} -> p_up={:.4f}", hour, ans["p_up"])
    if ans is None:
        raise ValueError("n должно быть положительным")
    return ans
