"""Загрузка обученной модели и прогноз.

Модель собирается `research/retrain.py` и лежит в `models/model.joblib` вместе
со списком признаков и карточкой. Здесь — только применение.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

import config
from app.log import get

log = get("model")


@lru_cache(maxsize=1)
def bundle() -> dict:
    path = config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"нет {path} — соберите модель: python research/retrain.py --cutoff "
            f"{config.CUTOFF_MINUTES}")
    b = joblib.load(path)
    card = b.get("card", {})
    trained_cutoff = b.get("cutoff", card.get("обрезка_минут"))
    if trained_cutoff != config.CUTOFF_MINUTES:
        # Это ровно тот случай, когда бот молча торговал бы не той моделью.
        raise ValueError(
            f"модель обучена на обрезке {trained_cutoff} мин, а в конфиге "
            f"{config.CUTOFF_MINUTES} — вход не совпадёт с обучением")
    log.info("модель загружена: %d признаков, обрезка %s мин, обучена по %s",
             len(b["features"]), trained_cutoff,
             card.get("обучено", {}).get("по", "?"))
    return b


def predict(feats: pd.Series) -> float:
    """Вероятность того, что часовая свеча закроется не ниже, чем открылась."""
    b = bundle()
    missing = [c for c in b["features"] if c not in feats.index]
    if missing:
        raise ValueError(f"в признаках не хватает колонок: {missing}")
    x = feats[b["features"]].to_frame().T.astype("float64")
    if x.isna().any().any():
        raise ValueError(f"NaN в признаках: {list(x.columns[x.isna().any()])}")
    return float(b["model"].predict_proba(x)[0, 1])


def card() -> dict:
    return bundle().get("card", {})
