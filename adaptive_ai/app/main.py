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

    mem_add = sub.add_parser("mem-add", help="Добавить факт в память")
    mem_add.add_argument("content", help="Содержимое факта")
    mem_add.add_argument("--type", default="fact", help="Тип факта (fact, hypothesis, etc.)")
    mem_add.add_argument("--status", default="user_claim", help="Статус достоверности")
    mem_add.add_argument("--source", default=None, help="Источник")
    mem_add.add_argument("--entity", default=None, help="Связанная сущность")

    mem_update = sub.add_parser("mem-update", help="Исправить запись")
    mem_update.add_argument("id", help="Идентификатор записи")
    mem_update.add_argument("content", help="Новое содержимое")

    mem_delete = sub.add_parser("mem-delete", help="Удалить запись")
    mem_delete.add_argument("id", help="Идентификатор записи")

    sub.add_parser("mem-clear", help="Очистить некритические временные записи (только архивация)")

    sub.add_parser("diagnostics", help="Запуск диагностических тестов")

    skills_parser = sub.add_parser("skills", help="Просмотр и управление навыками")
    skills_parser.add_argument("action", nargs="?", default="list", help="list, view, check")
    skills_parser.add_argument("skill_id", nargs="?", help="ID навыка")

    mcp_parser = sub.add_parser("mcp", help="Управление MCP-серверами")
    mcp_parser.add_argument("action", nargs="?", default="status", help="status, list")

    exp_parser = sub.add_parser("experience", help="Просмотр накопленного опыта")
    exp_parser.add_argument("action", nargs="?", default="stats", help="stats, list, confirmed, failed, training")
    exp_parser.add_argument("--limit", type=int, default=10, help="Лимит записей")

    eval_parser = sub.add_parser("eval", help="Запуск оценки качества")
    eval_parser.add_argument("suite", nargs="?", default="all", help="Название набора (all, russian_language, fact_extraction, memory, logic, tool_execution, error_handling, structured_output)")
    eval_parser.add_argument("--save", action="store_true", help="Сохранить результаты в файл")

    tools_parser = sub.add_parser("tools", help="Просмотр и тестирование инструментов")
    tools_parser.add_argument("action", nargs="?", default="list", help="list или test <tool_name>")
    tools_parser.add_argument("tool_name", nargs="?", help="Имя инструмента для теста")
    tools_parser.add_argument("tool_args", nargs="*", help="Аргументы инструмента")

    sub.add_parser("training-status", help="Просмотр статуса обучения и версий модели")

    train_parser = sub.add_parser("train", help="Управление обучением")
    train_parser.add_argument("action", nargs="?", default="status", help="status, check, prepare, run, eval, promote")
    train_parser.add_argument("--model", help="Путь к базовой модели")
    train_parser.add_argument("--dataset", help="Путь к обучающему набору")
    train_parser.add_argument("--artifact", help="Путь к артефакту кандидата")
    train_parser.add_argument("--exp-id", help="ID эксперимента")

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
