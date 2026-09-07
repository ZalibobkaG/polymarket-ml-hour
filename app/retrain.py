"""Периодическое переобучение модели `poly_hourly_kit` на новых данных.

    python main.py retrain

Модель нужно обновлять раз в интервал обучения, иначе её граница знаний
(`data_cutoff_utc` в паспорте) стареет, а сигнал, по измерениям самого
комплекта, и так слабеет год от года (см. `poly_hourly_kit/README.md` §6).
Комплект обучен на скользящем окне 1460 дней с шагом 30 — раз в месяц вполне
соответствует этой схеме; чаще — просто лишняя закачка минуток без выгоды.

Что делает, по шагам:

1.  Бэкапит текущие `model/*.{txt,json}` и `fixtures/*` — на случай отката.
2.  Запускает `python -m train.retrain --replace` внутри комплекта: это его
    родная команда (walk-forward → обучение боевой модели → замена файлов →
    пересборка фикстур), см. `poly_hourly_kit/train/retrain.py`.
3.  Прогоняет `python -m kit.selfcheck` — комплект сверяет новую модель сам
    с собой на пересобранном эталоне. Не прошла — откат к бэкапу.
4.  Шлёт итог в Telegram: было/стало по ключевым метрикам, либо причину отказа.

Живой бот подхватывает новую модель сам, без перезапуска: `app/model.py`
сверяет sha256 файла модели перед каждым прогнозом (`reload_if_changed`).

Расписание — не отсюда, у процесса бота (`main.py run`) нет своего планировщика
для этого. Позвать эту команду раз в месяц нужно снаружи, например cron на сервере:

    0 4 1 * *  cd /path/to/bot && .venv/bin/python main.py retrain >> logs/retrain.log 2>&1

Время в 04:00 UTC — не случайно: это провал активности что в Азии, что в США,
и до и после переобучения бот всё равно продолжает торговать старой моделью,
пока `--replace` не отработает (обычно минута-две, если минутки уже в кэше).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

from app import notify
from app.log import get

log = get("retrain")

ROOT = Path(__file__).resolve().parents[1]
KIT_DIR = ROOT / "poly_hourly_kit"
MODEL_DIR = KIT_DIR / "model"
FIXTURES_DIR = KIT_DIR / "fixtures"
BACKUP_ROOT = KIT_DIR / "model_backup"

#: Сколько последних бэкапов хранить на диске — остальные подчищаются.
KEEP_BACKUPS = 3

#: Сколько последних строк вывода класть в лог и в сообщение об ошибке —
#: полный вывод переобучения многословный, в Telegram он не нужен целиком.
LOG_TAIL = 4000
NOTIFY_TAIL = 500


def _backup() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = BACKUP_ROOT / stamp
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MODEL_DIR, dest / "model", dirs_exist_ok=True)
    shutil.copytree(FIXTURES_DIR, dest / "fixtures", dirs_exist_ok=True)
    return dest


def _restore(backup: Path) -> None:
    shutil.copytree(backup / "model", MODEL_DIR, dirs_exist_ok=True)
    shutil.copytree(backup / "fixtures", FIXTURES_DIR, dirs_exist_ok=True)


def _cleanup_backups() -> None:
    if not BACKUP_ROOT.exists():
        return
    backups = sorted((p for p in BACKUP_ROOT.iterdir() if p.is_dir()), reverse=True)
    for old in backups[KEEP_BACKUPS:]:
        shutil.rmtree(old, ignore_errors=True)


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    """Дочерний процесс `python -m <module> <args>` с cwd внутри комплекта.

    Отдельный процесс, а не импорт функций комплекта в текущий, — намеренно:
    `train.retrain` и `kit.selfcheck` рассчитаны на запуск именно так (это их
    задокументированный интерфейс), и падение переобучения не должно унести
    с собой процесс, из которого его позвали.
    """
    return subprocess.run([sys.executable, "-m", module, *args], cwd=str(KIT_DIR),
                          capture_output=True, text=True)


def _read_passport() -> dict:
    p = MODEL_DIR / "poly_hourly.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def run() -> int:
    log.info("retraining: start")
    old_meta = _read_passport()
    backup = _backup()
    log.info("backed up previous model: {}", backup)

    res = _run("train.retrain", "--replace")
    log.debug("train.retrain --replace (code {}):\n{}", res.returncode,
             (res.stdout or "")[-LOG_TAIL:])
    if res.returncode != 0:
        log.error("retraining failed, rolling back to the previous model:\n{}",
                  (res.stderr or "")[-LOG_TAIL:])
        _restore(backup)
        notify.error("Переобучение не удалось",
                     f"train.retrain вернул код {res.returncode}, оставлена "
                     f"прежняя модель.\n{(res.stderr or res.stdout)[-NOTIFY_TAIL:]}")
        return 1

    check = _run("kit.selfcheck")
    log.debug("kit.selfcheck (code {}):\n{}", check.returncode,
             (check.stdout or "")[-LOG_TAIL:])
    if check.returncode != 0:
        log.error("new model failed selfcheck — rolling back")
        _restore(backup)
        notify.error("Переобучение отклонено",
                     "Новая модель не прошла самопроверку комплекта (допуск "
                     "ноль) — оставлена прежняя модель. Подробности в logs/.")
        return 1

    new_meta = _read_passport()
    _cleanup_backups()

    old_wf, new_wf = old_meta.get("walkforward", {}), new_meta.get("walkforward", {})
    lines = [
        "🔄 <b>Модель переобучена</b>",
        f"{old_meta.get('model_id', '—')} → {new_meta.get('model_id', '—')}",
        f"Точность: {old_wf.get('model_pct', 0):.2f}% → {new_wf.get('model_pct', 0):.2f}%",
        f"Разрыв над правилом: {old_wf.get('gap_pp', 0):+.2f} п.п. → "
        f"{new_wf.get('gap_pp', 0):+.2f} п.п.",
        f"Наклон калибровки: {old_wf.get('calibration_slope', 0):.3f} → "
        f"{new_wf.get('calibration_slope', 0):.3f}",
        f"Граница знаний: {new_meta.get('data_cutoff_utc', '—')}",
    ]
    notify.send("\n".join(lines), silent=True)
    log.success("retraining done: {} -> {}",
               old_meta.get("model_id"), new_meta.get("model_id"))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
