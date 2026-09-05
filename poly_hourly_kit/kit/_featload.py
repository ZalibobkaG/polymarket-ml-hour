"""Загрузка `kit/features.py` — побитовой копии файла признаков из проекта.

Зачем такая возня вместо обычного `import`.

`kit/features.py` скопирован байт в байт и править его нельзя: модель обучена
именно на этой версии признаков, любая правка делает её другой моделью. А внутри
у копии есть строка

    from src.data import INTERVAL_MS

— след проекта, из которого она пришла. Папки `src` в комплекте нет и быть не
должно, поэтому нужный модуль собирается прямо в памяти и подставляется в
`sys.modules` ровно на время загрузки, после чего прежнее содержимое `sys.modules`
возвращается на место.

Побочный смысл этой возни: даже если комплект запустить внутри исходного проекта,
где настоящий пакет `src` есть, признаки всё равно посчитаются нашей копией,
а не чужим файлом. Комплект герметичен.

`INTERVAL_MS` — единственное, что копии нужно снаружи. Значения взяты из
`src/data.py` проекта и для часовой модели важно только `"1h"`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent
ROOT = KIT_DIR.parent
FEATURES_PATH = KIT_DIR / "features.py"
PASSPORT_PATH = ROOT / "passport.json"

#: Длительность бара в миллисекундах. Копия из `src/data.py` исходного проекта.
INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


def sha256_of(path: Path) -> str:
    """sha256 файла в шестнадцатеричном виде."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_features_file() -> str:
    """Сверяет `kit/features.py` с паспортом комплекта. Возвращает его sha256."""
    if not FEATURES_PATH.exists():
        raise RuntimeError(
            f"Не найден файл признаков {FEATURES_PATH}.\n"
            "Он входит в комплект и обязателен: без него считать признаки нечем."
        )
    if not PASSPORT_PATH.exists():
        raise RuntimeError(
            f"Не найден паспорт комплекта {PASSPORT_PATH}.\n"
            "В нём лежат контрольные суммы; без него нельзя убедиться, что файлы целы."
        )
    passport = json.loads(PASSPORT_PATH.read_text(encoding="utf-8"))
    want = passport.get("features_sha256")
    got = sha256_of(FEATURES_PATH)
    if want != got:
        raise RuntimeError(
            "Файл kit/features.py изменён.\n"
            f"  ожидался sha256: {want}\n"
            f"  получен  sha256: {got}\n"
            "Модель обучена ровно на этой версии признаков. Любая правка — даже\n"
            "переставленный пробел — делает её другой моделью, и измеренные числа\n"
            "перестают к ней относиться. Верните файл из комплекта."
        )
    return got


def _load() -> types.ModuleType:
    """Импортирует `kit/features.py` с подставленной заглушкой `src.data`."""
    spec = importlib.util.spec_from_file_location("poly_kit_features", FEATURES_PATH)
    module = importlib.util.module_from_spec(spec)

    stub_pkg = types.ModuleType("src")
    stub_pkg.__path__ = []                      # пустой пакет: подпакетов у него нет
    stub_data = types.ModuleType("src.data")
    stub_data.INTERVAL_MS = dict(INTERVAL_MS)
    stub_pkg.data = stub_data

    saved = {name: sys.modules.get(name) for name in ("src", "src.data")}
    sys.modules["src"] = stub_pkg
    sys.modules["src.data"] = stub_data
    try:
        spec.loader.exec_module(module)
    finally:
        # Возвращаем sys.modules ровно в то состояние, в котором взяли.
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


FEATURES_SHA256 = _check_features_file()
_features = _load()

#: Список из 22 признаков в том порядке, в каком их ждёт модель.
FEATURES = list(_features.FEATURES)
#: Самое длинное окно признаков в барах (`z168`, `std168`).
MAX_LOOKBACK_BARS = int(_features.MAX_LOOKBACK_BARS)
#: Расчёт признаков. Сигнатура: build(df, timeframe) -> dict.
build = _features.build
