"""Схема обучения: та же, что в исследовании (claude/model_v2.py), с правками методики.

Оригинал не тронут — этот модуль самостоятельный, чтобы прогоны основного проекта
воспроизводились ровно как раньше. Отличий от `btc_model.py` семь, все они разобраны
в [ML_REVIEW.md](ML_REVIEW.md):

1.  **Гейт `balance` действует и на логрег** (§5). В оригинале `class_weight`
    предлагался логрегу всегда, независимо от `opts["balance"]`, из-за чего
    конфигурация plain у логрега не была plain, а аблация «окно+баланс» для него
    сравнивала только окно.

2.  **Elastic net вместо перебора блоков признаков** (§3). Восемь бинарных
    переключателей — это 256 дискретных ячеек, в которых TPE за 60 попыток
    выбирает шум (в отчёте это видно: блоки скачут от квартала к кварталу без
    системы). Здесь модель получает все признаки, а отбор делает L1-компонента
    внутри самой оптимизации — непрерывно и устойчиво. Ручек становится две
    (C, l1_ratio) вместо восьми переключателей плюс C.

3.  **Окно обучения и баланс убраны из пространства по умолчанию** (§3). Парный
    тест на 16 000 часов уже показал, что окно не даёт ничего (p 0.22–0.49);
    оставлять фальсифицированную ручку — это дисперсия процедуры без выигрыша.
    Механизм в коде сохранён (`train_windows`), просто не используется.

4.  **Внутренних фолдов шесть по месяцу вместо трёх по кварталу** (§3). Оценка,
    по которой выбираются параметры, усредняется по шести точкам вместо трёх,
    и при этом обучающая часть самого раннего внутреннего фолда становится
    длиннее, а не короче (6 месяцев валидации против 9).

5.  **Важности — по стандартизованным коэффициентам, не permutation на тесте** (§6).
    Оригинал считал permutation importance на тестовых данных, и по ней принимались
    дизайн-решения следующих заходов. Для линейной модели поверх StandardScaler
    коэффициент и есть важность — бесплатно, точно и без подглядывания в тест.

6.  **Калибровка на прошлых фолдах** (§7). Сильный L2 сжимает вероятности к 0.5;
    для accuracy это безразлично, а для торгового правила `q > цена` — нет.
    Калибратор обучается только на OOS-предсказаниях предыдущих фолдов.

7.  **Несколько сидов TPE** (§3). Выбор параметров — сам по себе случайная величина;
    `seeds` позволяет измерить её разброс и усреднить предсказания.

Всё остальное — разбиения, эмбарго, подбор строго внутри фолда, log loss как цель —
скопировано без изменений: оно и было правильным.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

#: Настройки прогона. Отличия от оригинала помечены — см. шапку модуля.
DEFAULT_OPTS = {
    "train_windows": (0,),      # §3: окно не подбирается, всегда вся история
    "balance": False,           # §3: баланс не подбирается
    "select_blocks": False,     # §2: блоки не перебираются, отбор делает elastic net
    #: Служебный флаг для аблации: воспроизводит исходное поведение, где class_weight
    #: предлагался логрегу независимо от `balance` (собственно §1). В рабочей схеме
    #: всегда False — иначе конфигурация «без баланса» опять перестанет быть таковой.
    "legacy_logreg_balance": False,
    "first_test": "2024-10-01",
    "freq": "QS",
    "embargo_days": 30,
    "min_train": 2000,
    "min_test": 200,
    "min_rolling_train": 500,
    "inner_folds": 6,           # §4: было 3
    "inner_months": 1,          # §4: было 3
}


def _opts(opts: dict | None) -> dict:
    return {**DEFAULT_OPTS, **(opts or {})}


# --------------------------------------------------------------------------
# Разбиения — без изменений относительно оригинала
# --------------------------------------------------------------------------

@dataclass
class Fold:
    label: str
    train: np.ndarray
    test: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(self.train.sum())

    @property
    def n_test(self) -> int:
        return int(self.test.sum())


def walk_forward_splits(index: pd.Index, opts: dict | None = None) -> list[Fold]:
    """Расширяющееся окно: train — всё до теста минус зазор, test — очередной квартал."""
    o = _opts(opts)
    idx = pd.DatetimeIndex(index)
    tz = idx.tz
    starts = pd.date_range(pd.Timestamp(o["first_test"], tz=tz), idx[-1], freq=o["freq"])
    folds: list[Fold] = []

    for i, t0 in enumerate(starts):
        t1 = starts[i + 1] if i + 1 < len(starts) else idx[-1] + pd.Timedelta(days=1)
        test = (idx >= t0) & (idx < t1)
        train = idx < t0 - pd.Timedelta(days=o["embargo_days"])
        if test.sum() < o["min_test"] or train.sum() < o["min_train"]:
            continue
        folds.append(Fold(label=f"{t0.year}Q{t0.quarter}", train=train, test=test,
                          test_start=idx[test][0], test_end=idx[test][-1]))
    return folds


def inner_splits(index: pd.Index, opts: dict | None = None) -> list[Fold]:
    """Walk-forward внутри обучающей части — для подбора параметров.

    §4: шесть фолдов по месяцу вместо трёх по кварталу. Оценка усредняется по шести
    точкам, а самый ранний внутренний фолд получает больше обучающих строк, а не
    меньше: под валидацию уходит 6 месяцев истории вместо 9.
    """
    o = _opts(opts)
    idx = pd.DatetimeIndex(index)
    end = idx[-1] + pd.Timedelta(days=1)
    n, months = o["inner_folds"], o["inner_months"]
    folds: list[Fold] = []

    for k in range(n, 0, -1):
        t0 = end - pd.DateOffset(months=months * k)
        t1 = end - pd.DateOffset(months=months * (k - 1))
        test = (idx >= t0) & (idx < t1)
        train = idx < t0 - pd.Timedelta(days=o["embargo_days"])
        if test.sum() < o["min_test"] or train.sum() < o["min_rolling_train"]:
            continue
        folds.append(Fold(label=f"inner{k}", train=train, test=test,
                          test_start=idx[test][0], test_end=idx[test][-1]))
    return folds


# --------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------

def _make_model(name: str, params: dict):
    """Собирает модель по имени и подобранным гиперпараметрам.

    §2: у логрега вместо чистого L2 — elastic net. При l1_ratio → 0 это ровно прежняя
    L2-регрессия, при l1_ratio → 1 — лассо, обнуляющее лишние признаки. То есть отбор
    признаков стал частью непрерывной оптимизации, а не отдельным перебором блоков.
    """
    if name == "logreg":
        if params.get("l1_ratio") is None:      # прежнее поведение, для сравнения
            clf = LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                                     max_iter=2000, random_state=SEED)
        else:
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=params["C"],
                l1_ratio=params["l1_ratio"], class_weight=params["class_weight"],
                max_iter=3000, random_state=SEED)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if name == "gbm":
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"], max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            max_features=params["max_features"], class_weight=params["class_weight"],
            early_stopping=False, random_state=SEED)
    if name == "gnb":
        priors = np.array([0.5, 0.5]) if params["equal_priors"] else None
        return Pipeline([("scale", StandardScaler()),
                         ("clf", GaussianNB(var_smoothing=params["var_smoothing"],
                                            priors=priors))])
    raise ValueError(name)


def _suggest(name: str, trial: optuna.Trial, groups: dict[str, list[str]],
             all_features: list[str], opts: dict) -> dict:
    """Пространство поиска.

    §1: гейт `balance` теперь общий для всех трёх моделей — в оригинале логрег
    получал `class_weight` даже в конфигурации, которая называлась «без баланса».

    §2: при `select_blocks=False` набор признаков не перебирается вовсе — модель
    получает все, а отбор делает L1-компонента elastic net.
    """
    params: dict = {}
    balance = opts["balance"]

    if name == "logreg":
        # Диапазон C сохранён как в оригинале, когда перебираются блоки: иначе шаг A
        # аблации сравнивал бы заодно и другое пространство поиска.
        lo = 1e-3 if opts["select_blocks"] else 1e-4
        params["C"] = trial.suggest_float("C", lo, 10.0, log=True)
        params["class_weight"] = (
            trial.suggest_categorical("class_weight", [None, "balanced"])
            if (balance or opts["legacy_logreg_balance"]) else None)
        params["l1_ratio"] = (trial.suggest_float("l1_ratio", 0.0, 1.0)
                              if not opts["select_blocks"] else None)
    elif name == "gbm":
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 30, 400)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 3, 31)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 120)
        params["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)
        params["max_features"] = trial.suggest_float("max_features", 0.4, 1.0)
        params["class_weight"] = (
            trial.suggest_categorical("class_weight", [None, "balanced"])
            if balance else None)
    elif name == "gnb":
        params["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)
        params["equal_priors"] = (
            trial.suggest_categorical("equal_priors", [True, False]) if balance else False)

    windows = opts["train_windows"]
    params["max_train_days"] = (
        trial.suggest_categorical("max_train_days", windows) if len(windows) > 1
        else windows[0])

    if opts["select_blocks"]:
        on = {g: trial.suggest_categorical(f"use_{g}", [True, False]) for g in groups}
        if not any(on.values()):
            raise optuna.TrialPruned()
        params["features"] = _expand(groups, on)
        params["groups_on"] = on
    else:
        params["features"] = list(all_features)
        params["groups_on"] = {g: True for g in groups}
    return params


def _expand(groups: dict[str, list[str]], on: dict[str, bool]) -> list[str]:
    out: list[str] = []
    for g, cols in groups.items():
        if on.get(g):
            out += [c for c in cols if c not in out]
    return out


def _fit_predict(name: str, params: dict, X_tr, y_tr, X_te) -> np.ndarray:
    cols = params["features"]
    model = _make_model(name, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr[cols], y_tr)
        return model.predict_proba(X_te[cols])[:, 1]


def rolling_train(X: pd.DataFrame, y: pd.Series, fold: Fold, max_train_days: int,
                  embargo_days: int = 30):
    if not max_train_days:
        return X[fold.train], y[fold.train]
    cutoff = fold.test_start - pd.Timedelta(days=max_train_days + embargo_days)
    keep = fold.train & (pd.DatetimeIndex(X.index) >= cutoff)
    return X[keep], y[keep]


def tune(name: str, X: pd.DataFrame, y: pd.Series, groups: dict[str, list[str]],
         n_trials: int = 60, seed: int = SEED,
         opts: dict | None = None) -> tuple[dict, optuna.Study]:
    """Байесовский подбор по внутренним walk-forward фолдам обучающей выборки.

    Цель — log loss, как в оригинале: она плавная и учитывает уверенность.
    """
    o = _opts(opts)
    folds = inner_splits(X.index, o)
    if not folds:
        raise RuntimeError("не хватает истории на внутренние фолды")
    all_features = list(X.columns)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(name, trial, groups, all_features, o)
        losses, weights = [], []
        for f in folds:
            X_tr, y_tr = rolling_train(X, y, f, params["max_train_days"], o["embargo_days"])
            if len(X_tr) < o["min_rolling_train"] or y_tr.nunique() < 2:
                raise optuna.TrialPruned()
            p = _fit_predict(name, params, X_tr, y_tr, X[f.test])
            losses.append(log_loss(y[f.test], p, labels=[0, 1]))
            weights.append(f.n_test)
        return float(np.average(losses, weights=weights))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    if o["select_blocks"]:
        on = {g: best.pop(f"use_{g}") for g in groups}
        best["features"] = _expand(groups, on)
        best["groups_on"] = on
    else:
        best["features"] = all_features
        best["groups_on"] = {g: True for g in groups}
        best.setdefault("l1_ratio", None)
    best.setdefault("max_train_days", o["train_windows"][0] if len(o["train_windows"]) == 1 else 0)
    best.setdefault("class_weight", None)
    if name == "gnb":
        best.setdefault("equal_priors", False)
    return best, study


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
    }
    out["roc_auc"] = roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else np.nan
    return out


def coef_importance(name: str, params: dict, X_tr, y_tr) -> dict:
    """Важность признаков по стандартизованным коэффициентам (§5).

    Признаки уже прошли StandardScaler, поэтому коэффициент напрямую сопоставим между
    признаками: это сдвиг логита на одно стандартное отклонение признака. В отличие от
    permutation importance считается на обучающих данных, то есть в тест не подглядывает
    вовсе, и не размазывается между коллинеарными признаками так, как permutation.
    """
    if name != "logreg":
        return {}
    cols = params["features"]
    model = _make_model(name, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr[cols], y_tr)
    return dict(zip(cols, model[-1].coef_[0]))


def run_walk_forward(X: pd.DataFrame, y: pd.Series, models: list[str],
                     groups: dict[str, list[str]] | None = None,
                     n_trials: int = 60, seeds: tuple[int, ...] = (SEED,),
                     verbose: bool = True, opts: dict | None = None) -> dict:
    """Полный прогон walk-forward. По каждому фолду: подбор на train, предсказание на test.

    §7: `seeds` — несколько сидов TPE. Выбор гиперпараметров сам по себе случаен;
    прогон с несколькими сидами показывает разброс этого выбора, а усреднение
    предсказаний снижает дисперсию процедуры. При одном сиде поведение прежнее.
    """
    o = _opts(opts)
    if groups is None:
        groups = {c: [c] for c in X.columns}
    folds = walk_forward_splits(X.index, o)
    rows, oos, chosen, coefs = [], [], [], []

    for f in folds:
        X_tr, y_tr = X[f.train], y[f.train]
        X_te, y_te = X[f.test], y[f.test]
        major = int(y_tr.mean() >= 0.5)
        persist = X_te["y_prev"].values.astype(int)

        fold_row = {
            "fold": f.label, "n_train": f.n_train, "n_test": f.n_test,
            "с": f.test_start.date(), "по": f.test_end.date(),
            "Up % в тесте": y_te.mean() * 100,
            "база: всегда Up": accuracy_score(y_te, np.full(f.n_test, major)),
            "база: как вчера": accuracy_score(y_te, persist),
        }

        for name in models:
            per_seed = []
            for s in seeds:
                params, _ = tune(name, X_tr, y_tr, groups, n_trials=n_trials,
                                 seed=s, opts=o)
                X_win, y_win = rolling_train(X, y, f, params["max_train_days"],
                                             o["embargo_days"])
                p = _fit_predict(name, params, X_win, y_win, X_te)
                per_seed.append(p)
                row = {"fold": f.label, "model": name, "seed": s,
                       "окно обучения": params["max_train_days"],
                       "C": params.get("C"), "l1_ratio": params.get("l1_ratio"),
                       "class_weight": params.get("class_weight"),
                       "точность сида": accuracy_score(y_te, (p >= 0.5).astype(int))}
                if name == "logreg":
                    c = coef_importance(name, params, X_win, y_win)
                    row["ненулевых признаков"] = int(sum(abs(v) > 1e-8 for v in c.values()))
                    coefs.append({"fold": f.label, "seed": s, **c})
                chosen.append(row)

            proba = np.mean(per_seed, axis=0)
            m = _metrics(y_te.values, proba)
            fold_row[name] = m["accuracy"]
            fold_row[f"{name}_auc"] = m["roc_auc"]
            fold_row[f"{name}_logloss"] = m["log_loss"]

            oos.append(pd.DataFrame({
                "fold": f.label, "model": name, "y": y_te.values, "proba": proba,
                "pred": (proba >= 0.5).astype(int), "base_major": major,
                "base_persist": persist,
            }, index=X_te.index))

        rows.append(fold_row)
        if verbose:
            best = " ".join(f"{n} {fold_row[n]*100:.1f}%" for n in models)
            print(f"  {f.label}: train {f.n_train:5d} / test {f.n_test:4d} | "
                  f"база {fold_row['база: всегда Up']*100:.1f}% | {best}")

    return {
        "folds": pd.DataFrame(rows).set_index("fold"),
        "oos": pd.concat(oos),
        "chosen": pd.DataFrame(chosen),
        "coefs": pd.DataFrame(coefs) if coefs else pd.DataFrame(),
    }


# --------------------------------------------------------------------------
# Калибровка (§6)
# --------------------------------------------------------------------------

def calibrate_oos(oos: pd.DataFrame, model: str, method: str = "isotonic") -> pd.Series:
    """Калибрует вероятности фолда по предсказаниям всех предыдущих фолдов.

    Зачем. Сильная регуляризация жмёт логиты к нулю, и вероятности прижимаются к 0.5.
    Для accuracy при пороге 0.5 это безразлично, а для торгового правила `q > цена` —
    решающе: при цене 51 центов сжатая вероятность почти никогда её не превысит,
    и модель «не хочет» ставить даже там, где преимущество есть.

    Данные калибратору идут только из прошлого: для фолда k он обучен на честных
    OOS-предсказаниях фолдов 1..k−1. Первый фолд остаётся некалиброванным — истории
    под него нет, и это отмечено в колонке `calibrated`.
    """
    d = oos[oos["model"] == model].sort_index()
    labels = list(dict.fromkeys(d["fold"]))
    out = pd.Series(np.nan, index=d.index, dtype="float64")
    done = pd.Series(False, index=d.index)

    for i, lab in enumerate(labels):
        mask = (d["fold"] == lab).values
        if i == 0:
            out[mask] = d["proba"].values[mask]
            continue
        past = d[d["fold"].isin(labels[:i])]
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(past["proba"].values, past["y"].values)
            out[mask] = cal.predict(d["proba"].values[mask])
        else:
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(past[["proba"]].values, past["y"].values)
            out[mask] = lr.predict_proba(d[["proba"]].values[mask])[:, 1]
        done[mask] = True

    return pd.DataFrame({"proba_cal": out, "calibrated": done}, index=d.index)
