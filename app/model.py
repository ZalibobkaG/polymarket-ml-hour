"""Прогноз часовой модели — комплект под монету из `config.COIN`.

Модель, признаки и их проверка контрольной суммой живут в комплекте
(`poly_hourly_kit/` для BTC, `poly_hourly_kit_eth/` для ETH) — герметичном
и не подлежащем редактированию (см. его `README.md`). Здесь — только: путь до
него, сверка обрезки свечи и монеты с конфигом при старте и hot-reload модели,
чтобы переобучение (`main.py retrain`) подхватывалось без перезапуска бота.

Комплект добавляется в `sys.path`, и его пакет называется `kit` — одинаково у
всех монет. Поэтому один процесс работает ровно с одной монетой: второй
комплект в том же процессе молча вернул бы первый.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import config
from app.log import get

log = get("model")

ROOT = Path(__file__).resolve().parents[1]
KIT_DIR = config.KIT_DIR
if not KIT_DIR.exists():
    raise FileNotFoundError(
        f"нет комплекта {KIT_DIR} для монеты COIN={config.COIN!r} — "
        "положите его рядом с ботом или поправьте COIN в config.py")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import kit.candles as _kc  # noqa: E402  — комплект напарника, не модуль этого проекта
import kit.predict as _kp  # noqa: E402

_MODEL_TXT = KIT_DIR / "model" / f"{config.MODEL_STEM}.txt"
_MODEL_JSON = KIT_DIR / "model" / f"{config.MODEL_STEM}.json"

# Провал здесь — ровно тот случай, когда бот молча торговал бы не той моделью:
# признаки в бою обрезаны на CUTOFF_MINUTES, комплект обучен на обрезке kit.candles
# CUTOFF_MINUTES, и если числа разошлись, вход не совпадёт с обучением ни на бит.
if _kc.CUTOFF_MINUTES != config.CUTOFF_MINUTES:
    raise ValueError(
        f"комплект {KIT_DIR.name} обучен на обрезке {_kc.CUTOFF_MINUTES} мин, а в "
        f"конфиге "
        f"CUTOFF_MINUTES={config.CUTOFF_MINUTES} — вход не совпадёт с обучением")

_last_sha: str | None = None


def _verify_symbol() -> None:
    """Паспорт модели обязан говорить про ту же монету, что и конфиг.

    Это главная страховка всего переключения монет. Минутки любой пары приходят
    в одном формате, и модель ETH, накормленная минутками BTC, вернёт не ошибку,
    а правдоподобную вероятность — ошибку заметили бы только по деньгам. Паспорт
    читается как JSON, без загрузки бустера, поэтому проверка дешёвая и делается
    при каждой перезагрузке модели, в том числе после переобучения.
    """
    meta = json.loads(_MODEL_JSON.read_text(encoding="utf-8"))
    # symbol появился в паспортах не сразу; у старых комплектов монета указана
    # только в правиле резолва ("... полной часовой свечи Binance BTCUSDT ...").
    haystack = f"{meta.get('symbol') or ''} {meta.get('target_rule') or ''} " \
               f"{meta.get('market') or ''}"
    if config.SYMBOL not in haystack:
        raise ValueError(
            f"комплект {KIT_DIR.name} обучен не на {config.SYMBOL}: в паспорте "
            f"{_MODEL_JSON.name} нет упоминания этой пары "
            f"(symbol={meta.get('symbol')!r}, target_rule={meta.get('target_rule')!r}).\n"
            f"Либо COIN={config.COIN!r} указывает не на тот комплект, либо в model/ "
            "лежит модель другой монеты. Считать по ней нельзя: ошибки не будет, "
            "будет молча неверный прогноз."
        )


def reload_if_changed() -> bool:
    """Перечитать модель, если файлы в `<KIT_DIR>/model/` обновились.

    `kit/predict.py` кэширует бустер в собственном модульном кэше навсегда —
    расcчитано на короткоживущий процесс. У нас процесс живёт неделями, а
    `main.py retrain` кладёт новую модель в тот же каталог, не трогая бота, —
    поэтому кэш сбрасывается отсюда, по контрольной сумме файла модели.
    Возвращает True, если модель только что сменилась (не при первой загрузке).
    """
    global _last_sha
    if not _MODEL_TXT.exists():
        return False
    sha = hashlib.sha256(_MODEL_TXT.read_bytes()).hexdigest()
    if sha == _last_sha:
        return False
    changed = _last_sha is not None
    _last_sha = sha
    _kp._model_cache = None
    _verify_symbol()
    if changed:
        log.success("model reloaded after retraining: {}", sha[:12])
    return changed


reload_if_changed()  # первая загрузка процесса — не «переобучение», просто старт


def predict(minutes: pd.DataFrame, hour_start: pd.Timestamp) -> dict:
    """Прогноз на час `hour_start`. Тонкая обёртка над `kit.predict.predict()`.

    Возвращает словарь с полями `p_up`, `side`, `model_id`, `decision_time_utc`,
    `feature_hour_utc` — см. `<KIT_DIR>/kit/predict.py`.
    """
    reload_if_changed()
    return _kp.predict(minutes, hour_start)


def card() -> dict:
    """Паспорт текущей модели (`<KIT_DIR>/model/<MODEL_STEM>.json`)."""
    return _kp._load_model()["meta"]
