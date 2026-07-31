"""Приём команд из Telegram.

Поток-демон опрашивает getUpdates длинным поллингом и раздаёт команды обработчикам.
Исходящие сообщения живут в `notify`, здесь только входящие.

Безопасность. Доступ определяется белым списком `TG_ALLOWED_USER_IDS`, и правило
именно «запрещено по умолчанию»: пустой список — команды не принимаются ни от кого.
Обратное поведение (пусто = всем можно) означало бы, что любой, кто найдёт бота по
имени, останавливает торговлю, а токен бота — не секрет получателя, только отправителя.
Проверяется `from.id` автора сообщения, а не chat.id: в группе они разные, и
подставить чужой `from.id` нельзя.

Незнакомцам не отвечаем — молчание не подтверждает, что бот жив. Попытка пишется
в лог вместе с id, чтобы владельца можно было добавить в список осознанно.
"""

from __future__ import annotations

import threading
from typing import Callable

import httpx

import config
from app.log import get

log = get("commands")

_API = "https://api.telegram.org/bot{token}/{method}"
#: Длинный поллинг: соединение висит и ждёт сообщение, а не долбит API опросами.
_POLL_SECONDS = 25
_TIMEOUT = _POLL_SECONDS + 10


def allowed_ids() -> set[str]:
    """Кому можно командовать ботом."""
    ids = {x for x in config.TG_ALLOWED_USER_IDS if x}
    if not ids and config.TELEGRAM_CHAT_ID:
        # В личном чате chat_id совпадает с user_id, так что это тот же человек.
        ids = {str(config.TELEGRAM_CHAT_ID)}
    return ids


class Listener(threading.Thread):
    """Опрос Telegram и вызов обработчиков команд.

    `handlers` — словарь «/команда → функция(аргумент) -> текст ответа». Ответ
    отправляется автору; исключение внутри обработчика не роняет поток.
    """

    daemon = True

    def __init__(self, handlers: dict[str, Callable[[str], str]]) -> None:
        super().__init__(name="tg-commands")
        self.handlers = handlers
        self.allowed = allowed_ids()
        self._stop = threading.Event()
        self._offset: int | None = None

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _call(self, method: str, **params):
        r = httpx.post(_API.format(token=config.TELEGRAM_BOT_TOKEN, method=method),
                       json=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("result")

    def _reply(self, chat_id, text: str) -> None:
        try:
            self._call("sendMessage", chat_id=chat_id, text=text,
                       parse_mode="HTML", disable_web_page_preview=True)
        except Exception as e:                                # noqa: BLE001
            log.warning("не удалось ответить: %s", notify_safe(str(e)))

    def _drain_backlog(self) -> None:
        """Пропустить команды, накопившиеся до запуска.

        Иначе `/stop`, отправленный вчера, остановит бота через секунду после старта.
        """
        try:
            updates = self._call("getUpdates", timeout=0, offset=-1) or []
        except Exception as e:                                # noqa: BLE001
            log.warning("не удалось прочитать старые сообщения: %s", notify_safe(str(e)))
            return
        if updates:
            self._offset = updates[-1]["update_id"] + 1
            log.info("старые сообщения пропущены (%d)", len(updates))

    def run(self) -> None:
        if not config.TELEGRAM_BOT_TOKEN:
            log.info("токен не задан — команды из Telegram не принимаются")
            return
        if not self.allowed:
            log.warning("TG_ALLOWED_USER_IDS пуст — команды не принимаются ни от кого; "
                        "добавьте свой user_id в .env, чтобы включить управление")
            return
        log.info("команды из Telegram: %s, разрешено id %s",
                 " ".join(sorted(self.handlers)), ", ".join(sorted(self.allowed)))
        self._drain_backlog()

        while not self._stop.is_set():
            try:
                updates = self._call("getUpdates", timeout=_POLL_SECONDS,
                                     offset=self._offset,
                                     allowed_updates=["message"]) or []
            except Exception as e:                            # noqa: BLE001
                log.warning("опрос Telegram не удался: %s", notify_safe(str(e)))
                self._stop.wait(10)
                continue
            for u in updates:
                self._offset = u["update_id"] + 1
                if not self._stop.is_set():
                    self._dispatch(u)

    def _dispatch(self, update: dict) -> None:
        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        author = str((msg.get("from") or {}).get("id", ""))
        chat_id = (msg.get("chat") or {}).get("id")

        if author not in self.allowed:
            log.warning("команда %r от постороннего id %s — игнорирую",
                        text.split()[0], author or "?")
            return

        # /stop@botname arg -> ("stop", "arg")
        head, _, arg = text.partition(" ")
        cmd = head.split("@")[0].lstrip("/").lower()
        handler = self.handlers.get(cmd)
        if handler is None:
            self._reply(chat_id, "Не знаю такой команды. Доступны: "
                                 + ", ".join("/" + c for c in sorted(self.handlers)))
            return

        log.info("команда /%s от %s", cmd, author)
        try:
            answer = handler(arg.strip())
        except Exception as e:                                # noqa: BLE001
            log.exception("обработчик /%s упал", cmd)
            answer = f"Команда не выполнена: {e}"
        if answer:
            self._reply(chat_id, answer)


def notify_safe(text: str) -> str:
    """Токен в тексте ошибки — из URL httpx; в лог он попадать не должен."""
    tok = config.TELEGRAM_BOT_TOKEN
    return text.replace(tok, "***") if tok else text
