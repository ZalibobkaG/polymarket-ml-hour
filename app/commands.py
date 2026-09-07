"""Приём команд из Telegram.

Поток-демон опрашивает getUpdates длинным поллингом и раздаёт команды обработчикам.
Исходящие сообщения живут в `notify`, здесь только входящие.

Управлять ботом можно тремя способами, и все три ведут в один и тот же код:

* набрать команду руками — `/status`;
* выбрать её в меню ☰ рядом с полем ввода (список туда кладёт `_publish_menu`);
* нажать кнопку под сообщением `/menu`.

Кнопка несёт в `callback_data` ровно ту строку, которую можно набрать руками, —
поэтому отдельного реестра «действий меню» не существует, и меню не может
разъехаться с командами.

Безопасность. Доступ определяется белым списком `TG_ALLOWED_USER_IDS`, и правило
именно «запрещено по умолчанию»: пустой список — команды не принимаются ни от кого.
Обратное поведение (пусто = всем можно) означало бы, что любой, кто найдёт бота по
имени, останавливает торговлю, а токен бота — не секрет получателя, только отправителя.
Проверяется `from.id` автора — и у сообщения, и у нажатой кнопки, а не chat.id:
в группе они разные, и подставить чужой `from.id` нельзя.

Незнакомцам не отвечаем — молчание не подтверждает, что бот жив. Попытка пишется
в лог вместе с id, чтобы владельца можно было добавить в список осознанно. По той
же причине список команд публикуется не всем подряд, а точечно в чаты из белого
списка: постороннему незачем видеть, чем этот бот управляется.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import httpx

import config
from app.log import get

log = get("commands")


@dataclass
class Reply:
    """Ответ обработчика: текст и, если нужно, кнопки под ним.

    `buttons` — строки кнопок, каждая кнопка это `(подпись, команда)`, где
    команда пишется так же, как её набрали бы руками: `("📊 Статус", "/status")`.
    """

    text: str
    buttons: list[list[tuple[str, str]]] | None = None


@dataclass(frozen=True)
class Command:
    """Одна команда бота.

    `title` — подпись в меню ☰ Telegram. Пустая означает «команду можно вызвать,
    но в меню не показывать» — так живут служебные формы вроде `/stop yes`.
    """

    run: Callable[[str], "str | Reply | None"]
    title: str = ""

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

    `commands` — словарь «имя команды → `Command`». Ответ отправляется автору;
    исключение внутри обработчика не роняет поток.
    """

    daemon = True

    def __init__(self, commands: dict[str, Command]) -> None:
        super().__init__(name="tg-commands")
        self.commands = commands
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

    def _reply(self, chat_id, answer: "str | Reply") -> None:
        if isinstance(answer, str):
            answer = Reply(answer)
        params: dict = dict(chat_id=chat_id, text=answer.text, parse_mode="HTML",
                            disable_web_page_preview=True)
        if answer.buttons:
            params["reply_markup"] = {"inline_keyboard": [
                [{"text": label, "callback_data": cmd} for label, cmd in row]
                for row in answer.buttons]}
        try:
            self._call("sendMessage", **params)
        except Exception as e:                                # noqa: BLE001
            log.warning("failed to reply: {}", notify_safe(str(e)))

    def _publish_menu(self) -> None:
        """Положить список команд в меню ☰ рядом с полем ввода.

        Область — конкретные чаты из белого списка, а не «все»: бот приватный,
        и постороннему незачем видеть, чем он управляется. Своего состояния у
        этого вызова нет, он идемпотентен — Telegram просто держит последний
        присланный список, так что повторять его на каждом старте безопасно.
        """
        # Порядок словаря, а не алфавит: в меню он и будет виден, а полезность
        # команд по алфавиту не раскладывается.
        listed = [{"command": name, "description": c.title}
                  for name, c in self.commands.items() if c.title]
        if not listed:
            return
        for uid in sorted(self.allowed):
            try:
                self._call("setMyCommands", commands=listed,
                           scope={"type": "chat", "chat_id": int(uid)})
            except Exception as e:                            # noqa: BLE001
                log.warning("failed to publish the ☰ menu for {}: {}",
                            uid, notify_safe(str(e)))

    def _drain_backlog(self) -> None:
        """Пропустить команды, накопившиеся до запуска.

        Иначе `/stop`, отправленный вчера, остановит бота через секунду после старта.
        """
        try:
            updates = self._call("getUpdates", timeout=0, offset=-1) or []
        except Exception as e:                                # noqa: BLE001
            log.warning("failed to read old messages: {}", notify_safe(str(e)))
            return
        if updates:
            self._offset = updates[-1]["update_id"] + 1
            log.info("skipped {} backlog message(s)", len(updates))

    def run(self) -> None:
        if not config.TELEGRAM_BOT_TOKEN:
            log.info("no token set — Telegram commands are not accepted")
            return
        if not self.allowed:
            log.warning("TG_ALLOWED_USER_IDS is empty — no one can command the bot; "
                        "add your user_id to .env to enable control")
            return
        log.info("Telegram commands: {}, allowed ids {}",
                 " ".join(sorted(self.commands)), ", ".join(sorted(self.allowed)))
        self._drain_backlog()
        self._publish_menu()

        while not self._stop.is_set():
            try:
                updates = self._call("getUpdates", timeout=_POLL_SECONDS,
                                     offset=self._offset,
                                     allowed_updates=["message",
                                                      "callback_query"]) or []
            except Exception as e:                            # noqa: BLE001
                log.warning("Telegram poll failed: {}", notify_safe(str(e)))
                self._stop.wait(10)
                continue
            for u in updates:
                self._offset = u["update_id"] + 1
                if not self._stop.is_set():
                    self._dispatch(u)

    def _dispatch(self, update: dict) -> None:
        if update.get("callback_query"):
            self._on_button(update["callback_query"])
            return

        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        author = str((msg.get("from") or {}).get("id", ""))
        chat_id = (msg.get("chat") or {}).get("id")

        if author not in self.allowed:
            log.warning("command {!r} from unauthorized id {} — ignoring",
                        text.split()[0], author or "?")
            return
        self._run(text, chat_id, author)

    def _on_button(self, query: dict) -> None:
        """Нажатие кнопки под сообщением.

        На кнопке крутятся «часики», пока не ответишь на callback, поэтому
        отвечаем всегда и первым делом — даже если нажал посторонний и делать
        мы ничего не станем.
        """
        author = str((query.get("from") or {}).get("id", ""))
        chat_id = ((query.get("message") or {}).get("chat") or {}).get("id")
        try:
            self._call("answerCallbackQuery", callback_query_id=query.get("id"))
        except Exception as e:                                # noqa: BLE001
            # Часики погаснут сами через полминуты; ради этого не бросаем команду.
            log.debug("answerCallbackQuery failed: {}", notify_safe(str(e)))

        if author not in self.allowed:
            log.warning("button {!r} from unauthorized id {} — ignoring",
                        query.get("data"), author or "?")
            return
        self._run(query.get("data") or "", chat_id, author)

    def _run(self, text: str, chat_id, author: str) -> None:
        """Выполнить команду — хоть набранную, хоть пришедшую с кнопки."""
        # /stop@botname arg -> ("stop", "arg")
        head, _, arg = text.strip().partition(" ")
        cmd = head.split("@")[0].lstrip("/").lower()
        command = self.commands.get(cmd)
        if command is None:
            self._reply(chat_id, "Не знаю такой команды. Меню — /menu, "
                                 "список — /help.")
            return

        log.info("command /{} from {}", cmd, author)
        try:
            answer = command.run(arg.strip())
        except Exception as e:                                # noqa: BLE001
            log.exception("handler /{} crashed", cmd)
            answer = f"Команда не выполнена: {e}"
        if answer:
            self._reply(chat_id, answer)


def notify_safe(text: str) -> str:
    """Токен в тексте ошибки — из URL httpx; в лог он попадать не должен."""
    tok = config.TELEGRAM_BOT_TOKEN
    return text.replace(tok, "***") if tok else text
