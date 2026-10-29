"""Точка входа CLI Adaptive Intelligence."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

# Устанавливаем UTF-8 для вывода в консоль Windows (избегаем ошибок charmap)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .config import Config
from .main_impl import run_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-ai",
        description="Adaptive Intelligence — локальная саморазвивающаяся интеллектуальная система",
    )
    sub = parser.add_subparsers(dest="command", help="Команды")

    chat = sub.add_parser("chat", help="Обычный диалог с моделью")
    chat.add_argument("message", nargs="?", help="Сообщение (если не передано — интерактивный режим)")

    solve = sub.add_parser("solve", help="Выполнить задачу с инструментами")
    solve.add_argument("message", help="Текст задачи")

    sub.add_parser("status", help="Просмотр состояния и возможностей системы")
    sub.add_parser("history", help="Просмотр истории задач")

    mem_search = sub.add_parser("mem-search", help="Поиск по памяти")
    mem_search.add_argument("query", help="Поисковый запрос")

    mem_view = sub.add_parser("mem-view", help="Просмотр конкретной записи")
    mem_view.add_argument("id", help="Идентификатор записи")

    sub.add_parser("mem-clear", help="Очистить некритические временные записи (только архивация)")

    sub.add_parser("diagnostics", help="Запуск диагностических тестов")

    sub.add_parser("training-status", help="Просмотр статуса обучения и версий модели")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    config = Config.load()
    config.ensure_dirs()

    try:
        return run_cli(args, config)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
    except Exception as exc:  # noqa: BLE001 — верхний уровень CLI
        print(f"Ошибка: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
