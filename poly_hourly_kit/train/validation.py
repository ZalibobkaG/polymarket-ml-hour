"""Разбиение по времени: скользящий walk-forward.

Схема зафиксирована и менять её нельзя — именно на ней измерены все числа
из README:

    скользящее обучение 1460 дней -> зазор 7 дней -> проверка 30 дней, шаг 30 дней
    период оценки 1095 дней, окна проверки встык и не пересекаются
    порог уверенности подбирается только на последних 90 днях обучающего окна

Почему обучающее окно именно 1460 дней (4 года): это единственная длина, при
которой в КАЖДОМ обучающем окне присутствуют все три фазы рынка. При окне в год
существуют окна вообще без медвежьей фазы, и модель, обученная на таком окне,
не видела падений ни разу.

Почему зазор 7 дней: он перекрывает самое длинное окно признаков (168 часов).
Без зазора последние бары обучения и первые бары проверки делили бы одни и те
же исходные свечи.

Никакого перемешивания, никакого KFold, никакого train_test_split. Только время.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scheme:
    """Параметры нарезки. Все длины — в днях, кроме двух страховок в барах."""
    train_days: int = 1460        # скользящее обучающее окно, 4 года
    embargo_days: int = 7         # зазор между обучением и проверкой
    val_days: int = 30            # длина окна проверки
    step_days: int = 30           # шаг; равен val_days, поэтому окна идут встык
    eval_days: int = 1095         # период оценки, 3 года
    inner_days: int = 90          # внутренняя валидация: хвост обучающего окна
    min_fold_days: int = 5        # огрызок короче этого фолдом не считается
    min_train_bars: int = 800     # страховки от вырожденных фолдов
    min_val_bars: int = 50


@dataclass(frozen=True)
class Fold:
    """Один фолд: чему учимся и на чём проверяемся."""
    n: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train: pd.Series              # булева маска по строкам датафрейма
    val: pd.Series

    @property
    def embargo(self) -> pd.Timedelta:
        return self.val_start - self.train_end


def _days(n: int) -> pd.Timedelta:
    return pd.Timedelta(days=n)


def eval_period(t: pd.Series, scheme: Scheme) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Границы периода оценки: последние eval_days дней, конец округлён вниз до полуночи."""
    end = t.max().normalize()
    return end - _days(scheme.eval_days), end


def val_windows(t: pd.Series, scheme: Scheme) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Окна проверки: встык, шагом step_days, огрызок короче min_fold_days отбрасывается."""
    start, end = eval_period(t, scheme)
    out, s = [], start
    while s < end:
        e = min(s + _days(scheme.val_days), end)
        if e - s >= _days(scheme.min_fold_days):
            out.append((s, e))
        s = s + _days(scheme.step_days)
    return out


def make_folds(t: pd.Series, usable: pd.Series, scheme: Scheme) -> list[Fold]:
    """Список фолдов. `usable` — строки, пригодные и для обучения, и для проверки."""
    folds: list[Fold] = []
    for vs, ve in val_windows(t, scheme):
        te = vs - _days(scheme.embargo_days)
        ts = te - _days(scheme.train_days)
        train = usable & (t >= ts) & (t < te)
        val = usable & (t >= vs) & (t < ve)
        if train.sum() < scheme.min_train_bars or val.sum() < scheme.min_val_bars:
            continue
        folds.append(Fold(len(folds), ts, te, vs, ve, train, val))
    return folds


def inner_split(t: pd.Series, train: pd.Series, scheme: Scheme,
                train_end: pd.Timestamp | None = None) -> tuple[pd.Series, pd.Series]:
    """Вложенный контур внутри обучающего окна.

    Последние inner_days дней окна — внутренняя валидация. Только на ней
    допустимо подбирать порог. Проверочная часть фолда сюда не попадает никогда:
    иначе порог был бы подобран по тем самым барам, на которых потом меряется
    результат.
    """
    end = t[train].max() if train_end is None else train_end
    cut = end - _days(scheme.inner_days)
    return train & (t <= cut), train & (t > cut)


def future_correlations(X: pd.DataFrame, fwd: pd.Series,
                        mask: pd.Series | None = None) -> pd.Series:
    """Корреляция каждого признака с доходностью следующего бара.

    Защита от утечки будущего. Настоящий признак знать будущее не может, поэтому
    его корреляция обязана быть маленькой (у нынешних 22 признаков максимум 0.03).
    Случайно подставленная доходность следующего часа дала бы 1.0 и была бы
    поймана здесь же.
    """
    ok = fwd.notna() if mask is None else (mask & fwd.notna())
    ok = ok.to_numpy()
    f = fwd.to_numpy()[ok]
    out = {}
    for c in X.columns:
        v = X[c].to_numpy()[ok]
        good = np.isfinite(v)
        if good.sum() < 2 or np.std(v[good]) == 0:
            out[c] = 0.0
            continue
        out[c] = abs(float(np.corrcoef(v[good], f[good])[0, 1]))
    return pd.Series(out).sort_values(ascending=False)


def check_folds(folds: list[Fold], t: pd.Series, usable: pd.Series,
                scheme: Scheme) -> dict:
    """Целостность нарезки: пересечения, зазор, полнота покрытия.

    Главное здесь — «предсказано ровно один раз». Если хоть один час периода
    оценки предсказан дважды или ни разу, метрики считать нельзя: они посчитаны
    не на том множестве, на котором кажется.
    """
    start, end = eval_period(t, scheme)
    ev = usable & (t >= start) & (t < end)

    covered = np.zeros(len(t), dtype=int)
    overlaps = 0
    embargoes: list[float] = []
    for f in folds:
        overlaps += int((f.train & f.val).sum())
        embargoes.append(f.embargo / pd.Timedelta(days=1))
        covered[np.flatnonzero(f.val.to_numpy())] += 1

    ev_np = ev.to_numpy()
    val_overlap = sum(1 for a, b in zip(folds, folds[1:]) if b.val_start < a.val_end)

    return {
        "n_folds": len(folds),
        "n_eval_bars": int(ev_np.sum()),
        "eval_start": start,
        "eval_end": end,
        "train_val_overlap": overlaps,
        "val_windows_overlap": val_overlap,
        "min_embargo_days": min(embargoes) if embargoes else float("nan"),
        "predicted_once": int((covered[ev_np] == 1).sum()),
        "predicted_never": int((covered[ev_np] == 0).sum()),
        "predicted_twice_or_more": int((covered[ev_np] >= 2).sum()),
        "predicted_outside_eval": int(covered[~ev_np].sum()),
    }
