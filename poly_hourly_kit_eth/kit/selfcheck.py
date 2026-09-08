"""Самопроверка комплекта. Это первое, что нужно запустить.

    python -m kit.selfcheck

Что она доказывает. В комплекте лежат 200 часов, посчитанных на нашей машине
настоящим прогоном: минутки, 22 признака и вероятность. Скрипт берёт те же
минутки, прогоняет их через `predict()` у вас и сверяет с эталоном.

ДОПУСК НОЛЬ. Не «почти совпало», не «в пределах 1e-9», а ровно то же число до
последнего бита. Так и задумано: любое расхождение означает, что где-то по пути
считается не то, и лучше узнать об этом сейчас, чем по деньгам через месяц.

Проверяется весь путь целиком:

    минутки -> обрезка :00..:57 -> часовые свечи -> 22 признака -> вероятность

Поэтому «ПРОВЕРКА ПРОЙДЕНА» означает не только что модель загрузилась, но и что
свечи вы собираете правильно и что тайминг не сдвинут.

Код возврата 0 — сошлось, 1 — нет.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
PASSPORT = ROOT / "passport.json"

W = 78

#: Версии, на которых модель обучалась и на которых снят эталон.
EXPECTED = {"lightgbm": "4.5.0", "numpy": "2.2.6", "pandas": "2.2.3"}


def _line(ch: str = "-") -> None:
    print(ch * W)


def _read_minutes(path: Path) -> pd.DataFrame:
    """Минутки фикстуры. float_precision='round_trip' — обязательно.

    Разбор чисел по умолчанию быстрее, но может ошибиться на последний бит.
    При допуске ноль такая ошибка сама создала бы расхождение на ровном месте.
    """
    df = pd.read_csv(path, float_precision="round_trip")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    return df.set_index("open_time").sort_index()


def run() -> int:
    _line("=")
    print("  САМОПРОВЕРКА КОМПЛЕКТА poly_hourly_kit_eth")
    _line("=")

    problems: list[str] = []

    # Импорт внутри функции, а не наверху: kit/features.py сверяется с паспортом
    # прямо при загрузке, и если файл правили, читать про это надо человеческим
    # текстом, а не в виде трассировки стека.
    try:
        from kit import _featload as _fl
        from kit import predict as _kp
    except Exception as e:
        print()
        print(str(e))
        print()
        _line("=")
        print("  ПРОВЕРКА НЕ ПРОЙДЕНА")
        _line("=")
        return 1

    # ---------- 1. окружение ----------
    import lightgbm as lgb

    versions = {"lightgbm": lgb.__version__, "numpy": np.__version__,
                "pandas": pd.__version__}
    print(f"  python    {sys.version.split()[0]}")
    for name, got in versions.items():
        want = EXPECTED[name]
        mark = "" if got == want else f"   <- ожидалась {want}"
        print(f"  {name:<9} {got}{mark}")
    mismatched = [n for n, g in versions.items() if g != EXPECTED[n]]
    if mismatched:
        print()
        print("  ВНИМАНИЕ: версии библиотек отличаются от тех, на которых снят эталон.")
        print("  Это не отказ — числа ниже скажут точнее любых предупреждений. Но если")
        print("  они разойдутся, начинать надо именно отсюда: pip install -r requirements.txt")
    print()

    # ---------- 2. контрольные суммы ----------
    print("  Контрольные суммы")
    if not PASSPORT.exists():
        print(f"    ОШИБКА: нет паспорта комплекта {PASSPORT}")
        return 1
    passport = json.loads(PASSPORT.read_text(encoding="utf-8"))

    # features.py уже сверен при импорте kit._featload — иначе бы сюда не дошли
    print(f"    kit/features.py           совпал  ({_fl.FEATURES_SHA256[:12]}...)")

    try:
        model = _kp._load_model()
    except Exception as e:
        print(f"    model/poly_hourly_eth.txt НЕ СОВПАЛ")
        print()
        print(str(e))
        return 1
    print(f"    model/poly_hourly_eth.txt совпал  (модель {model['model_id']})")

    fx_meta = passport.get("fixtures") or {}
    files = {"fixtures/minutes.csv.gz": fx_meta.get("minutes_csv_gz_sha256"),
             "fixtures/reference.csv": fx_meta.get("reference_csv_sha256")}
    for rel, want in files.items():
        path = ROOT / rel
        if not path.exists():
            print(f"    {rel:<25} НЕТ ФАЙЛА")
            problems.append(f"нет файла {rel}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if want and got != want:
            print(f"    {rel:<25} НЕ СОВПАЛ")
            problems.append(f"{rel}: sha256 {got}, а в паспорте {want}")
        else:
            print(f"    {rel:<25} совпал")
    if problems:
        print()
        for p in problems:
            print("  " + p)
        print()
        print("  Фикстуры изменены или повреждены. Возьмите их из комплекта заново.")
        return 1
    print()

    # ---------- 3. пересчёт 200 часов ----------
    minutes = _read_minutes(FIXTURES / "minutes.csv.gz")
    ref = pd.read_csv(FIXTURES / "reference.csv", float_precision="round_trip")
    hours = pd.to_datetime(ref["hour_utc"], utc=True, format="ISO8601")

    print(f"  Пересчёт {len(ref)} часов "
          f"({hours.iloc[0]:%Y-%m-%d %H:%M} .. {hours.iloc[-1]:%Y-%m-%d %H:%M} UTC)")
    print(f"  минуток в фикстуре: {len(minutes):,}".replace(",", " "))
    print("  допуск: НОЛЬ")
    print()

    bad_rows = []
    errors = []
    for i, hour in enumerate(hours):
        want_p = float(ref["p_up"].iloc[i])
        try:
            row, got_p = _kp._features_row(minutes, hour)
            answer = _kp.predict(minutes, hour)
        except Exception as e:
            errors.append((hour, f"{type(e).__name__}: {e}"))
            continue
        if got_p != want_p or answer["p_up"] != want_p:
            want_row = ref[_fl.FEATURES].iloc[i].to_numpy(dtype="float64")
            got_row = row.to_numpy(dtype="float64")
            diff_feats = [n for n, a, b in zip(_fl.FEATURES, got_row, want_row) if a != b]
            bad_rows.append((hour, want_p, got_p, diff_feats))

    # ---------- 4. вывод ----------
    if errors:
        _line()
        print(f"  ИСКЛЮЧЕНИЙ: {len(errors)}. Первые пять:")
        for hour, msg in errors[:5]:
            print(f"    {hour:%Y-%m-%d %H:%M} UTC — {msg}")
        print()

    if bad_rows:
        _line()
        print(f"  РАСХОЖДЕНИЙ: {len(bad_rows)} из {len(ref)}. Первые десять:")
        print()
        print(f"    {'час UTC':<17} {'эталон':>12} {'получено':>12} {'разница':>11}  "
              "разошлись признаки")
        for hour, want_p, got_p, feats in bad_rows[:10]:
            names = ", ".join(feats) if feats else "признаки совпали (дело в модели)"
            print(f"    {hour:%Y-%m-%d %H:%M}  {want_p:12.10f} {got_p:12.10f} "
                  f"{got_p - want_p:+11.3e}  {names}")
        print()
        print("  Как это читать:")
        print("    признаки разошлись  -> расходится сборка свечей или обрезка;")
        print("                           проверьте минутки и колонку trades;")
        print("    признаки совпали    -> расходится сама модель: почти всегда")
        print("                           это другая версия lightgbm.")

    if errors or bad_rows:
        _line("=")
        print("  ПРОВЕРКА НЕ ПРОЙДЕНА")
        _line("=")
        return 1

    _line("=")
    print("  ПРОВЕРКА ПРОЙДЕНА")
    _line("=")
    print(f"  {len(ref)} часов из {len(ref)} совпали с эталоном точно, до последнего бита.")
    print("  Значит правильно собираются свечи, правильно считаются 22 признака")
    print("  и правильно загрузилась модель. Комплект подключён верно.")
    print()
    print("  Дальше: README.md, раздел «Тайминг» — это единственное, что самопроверка")
    print("  подтвердить не может, потому что зависит от вашего планировщика.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
