"""Реализация CLI (вынесена из main.py для чистоты)."""

from __future__ import annotations

import sys
from typing import Any

from .config import Config

try:  # зависимости ядра импортируются лениво, чтобы CLI запускался даже без модели
    from adaptive_ai.core.contracts import (
        FactStatus,
        ModelBackend,
        ModelCapabilities,
        Status,
        TaskResult,
    )
    from adaptive_ai.core.orchestrator import Orchestrator
    from adaptive_ai.memory.sqlite_store import SqliteMemoryStore
    from adaptive_ai.models.model_registry import load_backend
    import adaptive_ai.models.rwkv_backend  # noqa: F401 — регистрация бэкенда
    from adaptive_ai.tools.registry import ToolRegistry
except Exception:  # pragma: no cover — при отсутствии зависимостей CLI покажет not_supported
    Orchestrator = None  # type: ignore[assignment]
    SqliteMemoryStore = None  # type: ignore[assignment]
    load_backend = None  # type: ignore[assignment]
    ToolRegistry = None  # type: ignore[assignment]
    ModelBackend = None  # type: ignore[assignment]
    ModelCapabilities = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    TaskResult = None  # type: ignore[assignment]


class _FallbackBackend:
    """Заглушка модели для работы CLI без реальной модели.

    Возвращает простой эхо-ответ с пометкой [режим без модели].
    Используется на Этапе 2, пока пользователь не положит модель в ./models.
    """

    name = "fallback"

    def load(self, model_path: str | None = None) -> TaskResult:
        self._loaded = True
        return TaskResult(status=Status.OK, data={"mode": "fallback"})

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> TaskResult:
        # Простой эхо-ответ без реальной инференсной модели
        answer = f"[режим без модели] Получил ваше сообщение: {prompt[:200]}"
        if system_prompt:
            answer += f"\nСистемная инструкция: {system_prompt[:100]}"
        if context:
            answer += f"\nКонтекст из памяти: {context[:200]}"
        return TaskResult(status=Status.OK, data=answer)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name=self.name,
            supports_state=False,
            supports_streaming=False,
            supports_structured_output=False,
            supports_adapters=False,
            supports_training=False,
            supports_state_save=False,
            supports_state_restore=False,
            context_window=4096,
        )

    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


def _not_supported(feature: str) -> int:
    print(f"Не реализовано: {feature}. Статус: not_supported.")
    return 0


def run_cli(args: Any, config: Config) -> int:
    if Orchestrator is None:
        print("Ядро системы недоступно. Проверьте установку зависимостей.")
        return 1

    # Инициализация компонентов
    try:
        backend = load_backend(config.model_backend, config.model_path)
        if not backend.is_loaded():
            # Модель не загружена — покажем ошибку и переключимся на fallback
            print(f"Предупреждение: модель не загружена ({config.model_path}).")
            print("Используется режим fallback (без реальной инференсной модели).")
            backend = _FallbackBackend()
            backend.load(None)
        memory = SqliteMemoryStore(config.db_path)
        registry = ToolRegistry()

        # Регистрация встроенных инструментов
        from adaptive_ai.tools.builtin.calculator import CalculatorTool
        from adaptive_ai.tools.builtin.python_calc import PythonCalcTool
        from adaptive_ai.tools.builtin.read_file import ReadFileTool
        from adaptive_ai.tools.builtin.run_tests import RunTestsTool
        from adaptive_ai.tools.builtin.search_memory import SearchMemoryTool

        registry.register(SearchMemoryTool(memory))
        registry.register(CalculatorTool())
        registry.register(PythonCalcTool())
        registry.register(ReadFileTool(allowed_dirs=[str(config.data_dir), str(config.models_dir)]))
        registry.register(RunTestsTool(allowed_dirs=[str(config.data_dir)]))

        from adaptive_ai.core.error_monitor import ErrorMonitor
        from adaptive_ai.core.guard_engine import GuardEngine
        from adaptive_ai.core.policies import SafetyPolicy, ResourcePolicy

        orch = Orchestrator(
            model=backend,
            memory=memory,
            tools=registry,
            safety=SafetyPolicy(),
            resources=ResourcePolicy(
                max_steps=config.max_steps,
                max_retries_per_action=config.max_retries,
                max_context_tokens=config.max_context_tokens,
                max_ram_mb=config.max_ram_mb,
            ),
            error_monitor=ErrorMonitor(),
            guard_engine=GuardEngine(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка инициализации: {exc}")
        return 1

    cmd = args.command

    if cmd == "chat":
        return _cmd_chat(orch, args, config)
    if cmd == "solve":
        return _cmd_solve(orch, args)
    if cmd == "status":
        return _cmd_status(orch, config)
    if cmd == "history":
        return _cmd_history(orch)
    if cmd == "mem-search":
        return _cmd_mem_search(orch, args)
    if cmd == "mem-view":
        return _cmd_mem_view(orch, args)
    if cmd == "mem-add":
        return _cmd_mem_add(orch, args)
    if cmd == "mem-update":
        return _cmd_mem_update(orch, args)
    if cmd == "mem-delete":
        return _cmd_mem_delete(orch, args)
    if cmd == "mem-clear":
        return _not_supported("mem-clear (архивация)")
    if cmd == "diagnostics":
        return _cmd_diagnostics(orch)
    if cmd == "eval":
        return _cmd_eval(orch, args, config)
    if cmd == "tools":
        return _cmd_tools(orch, args)
    if cmd == "training-status":
        return _cmd_training_status(config)

    print(f"Неизвестная команда: {cmd}")
    return 1


def _cmd_chat(orch: Orchestrator, args: Any, config: Config) -> int:
    message = args.message
    if not message:
        print("Интерактивный режим диалога. Введите 'exit' для выхода.")
        while True:
            try:
                message = input("> ")
            except EOFError:
                break
            if message.strip().lower() in ("exit", "quit", "выход"):
                break
            if not message.strip():
                continue
            result = orch.model.generate(prompt=message, max_tokens=512, temperature=0.7)
            print(f"Ответ: {result.data or result.error or 'без ответа'}")
        return 0
    result = orch.model.generate(prompt=message, max_tokens=512, temperature=0.7)
    print(f"Ответ: {result.data or result.error or 'без ответа'}")
    return 0


def _cmd_solve(orch: Orchestrator, args: Any) -> int:
    result = orch.solve(args.message)
    if result.status == "ok":
        print(f"Результат: {result.data}")
    else:
        print(f"Статус: {result.status}. Ошибка: {result.error}")
    return 0


def _cmd_status(orch: Orchestrator, config: Config) -> int:
    caps = orch.model.capabilities()
    print("=== Состояние системы Adaptive Intelligence ===")
    print(f"Модель: {caps.name} v{caps.version}")
    print(f"Бэкенд: {orch.model.name}")
    print(f"Загружена: {orch.model.is_loaded()}")
    if not orch.model.is_loaded():
        print("Режим: fallback (без модели — положите модель в ./models)")
    print(f"Поддерживает состояние: {caps.supports_state}")
    print(f"Потоковая генерация: {caps.supports_streaming}")
    print(f"Структурированный вывод: {caps.supports_structured_output}")
    print(f"Обучение: {caps.supports_training}")
    print(f"Макс. шагов: {orch.resources.max_steps}")
    print(f"Макс. повторов: {orch.resources.max_retries_per_action}")
    print(f"База памяти: {config.db_path}")
    print(f"Директория данных: {config.data_dir}")
    print(f"Директория моделей: {config.models_dir}")
    print(f"Сеть разрешена: {config.allow_network}")
    print(f"Обучение разрешено: {config.allow_training}")
    print(f"Инструменты: {[t.name for t in orch.tools.list()]}")
    return 0


def _cmd_history(orch: Orchestrator) -> int:
    tasks = orch.list_tasks()
    if not tasks:
        print("История задач пуста.")
        return 0
    for t in tasks:
        print(f"[{t.id}] {t.status.value} | {t.request[:60]}")
        if t.result:
            print(f"      результат: {t.result[:100]}")
        if t.error:
            print(f"      ошибка: {t.error}")
    return 0


def _cmd_mem_search(orch: Orchestrator, args: Any) -> int:
    items = orch.memory.search(args.query, limit=10)
    if not items:
        print("Воспоминания не найдены.")
        return 0
    for item in items:
        src = f" [источник: {item.source}]" if item.source else ""
        print(f"[{item.id}] {item.content[:120]}{src} | статус: {item.fact_status.value}")
    return 0


def _cmd_mem_add(orch: Orchestrator, args: Any) -> int:
    from adaptive_ai.core.contracts import MemoryItem

    try:
        fact_status = FactStatus(args.status)
    except ValueError:
        fact_status = FactStatus.USER_CLAIM

    item = MemoryItem(
        id="",  # будет сгенерирован в SqliteMemoryStore
        type=args.type,
        content=args.content,
        source=args.source,
        entity_id=args.entity,
        fact_status=fact_status,
    )
    # Генерируем id
    import uuid

    item.id = str(uuid.uuid4())
    res = orch.memory.save_fact(item)
    if res.status == "ok":
        print(f"Факт сохранён: id={res.data.get('id', item.id)}")
    else:
        print(f"Ошибка: {res.error}")
    return 0


def _cmd_mem_update(orch: Orchestrator, args: Any) -> int:
    res = orch.memory.update(args.id, content=args.content)
    if res.status == "ok":
        print(f"Запись {args.id} обновлена.")
    else:
        print(f"Ошибка: {res.error}")
    return 0


def _cmd_mem_delete(orch: Orchestrator, args: Any) -> int:
    res = orch.memory.delete(args.id)
    if res.status == "ok":
        print(f"Запись {args.id} удалена.")
    else:
        print(f"Ошибка: {res.error}")
    return 0


def _cmd_mem_view(orch: Orchestrator, args: Any) -> int:
    item = orch.memory.get(args.id)
    if not item:
        print(f"Запись с id {args.id} не найдена.")
        return 1
    print(f"ID: {item.id}")
    print(f"Тип: {item.type}")
    print(f"Содержимое: {item.content}")
    print(f"Источник: {item.source}")
    print(f"Статус достоверности: {item.fact_status.value}")
    print(f"Создано: {item.created_at}")
    print(f"Теги: {', '.join(item.tags)}")
    return 0


def _cmd_tools(orch: Orchestrator, args: Any) -> int:
    action = args.action or "list"
    if action == "list":
        print("=== Зарегистрированные инструменты ===")
        for tool in orch.tools.list():
            print(f"  {tool.name}: {tool.description}")
            print(f"    Вход: {tool.input_schema}")
            print(f"    Разрешения: {tool.required_permissions}")
            print(f"    Макс. время: {tool.max_time_seconds}с, Макс. RAM: {tool.max_memory_mb}MB")
        return 0
    if action == "test":
        tool_name = args.tool_name
        if not tool_name:
            print("Укажите имя инструмента: tools test <name> [args...]")
            return 1
        tool = orch.tools.get(tool_name)
        if not tool:
            print(f"Инструмент не найден: {tool_name}")
            return 1
        # Парсим аргументы key=value
        tool_args = {}
        for arg in (args.tool_args or []):
            if "=" in arg:
                k, v = arg.split("=", 1)
                tool_args[k] = v
        result = orch.tools.call(tool_name, tool_args)
        if result.status == "ok":
            print(f"Результат: {result.data}")
        else:
            print(f"Статус: {result.status}. Ошибка: {result.error}")
        return 0
    print(f"Неизвестное действие: {action}. Используйте 'list' или 'test'.")
    return 1


def _cmd_eval(orch: Orchestrator, args: Any, config: Config) -> int:
    from adaptive_ai.evaluation.datasets.suites import all_suites
    from adaptive_ai.evaluation.evaluators import Evaluator

    suites = all_suites()
    suite_name = args.suite or "all"

    if suite_name == "all":
        selected = []
        for s in suites.values():
            selected.extend(s)
    elif suite_name in suites:
        selected = suites[suite_name]
    else:
        print(f"Неизвестный набор: {suite_name}. Доступные: all, {', '.join(suites.keys())}")
        return 1

    print(f"=== Оценка качества: {suite_name} ({len(selected)} тестов) ===")

    evaluator = Evaluator(results_dir=str(config.data_dir / "eval_results"))

    def run_fn(inp):
        # Для инструментов — вызываем напрямую
        if isinstance(inp, str) and inp.startswith("calculator:"):
            expr = inp.split(":", 1)[1].strip()
            res = orch.tools.call("calculator", {"expression": expr})
            return res.data.get("result") if res.status == "ok" else None
        if isinstance(inp, str) and inp.startswith("python_calc:"):
            code = inp.split(":", 1)[1].strip()
            res = orch.tools.call("python_calc", {"code": code})
            return res.data.get("result") if res.status == "ok" else None
        # Для остальных — генерация модели
        result = orch.model.generate(prompt=str(inp), max_tokens=128, temperature=0.3)
        return result.data if result.status == "ok" else None

    results = evaluator.run(selected, run_fn)

    # Вывод результатов
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    errors = sum(1 for r in results if r.status == "error")
    print(f"\nИтог: {passed} passed, {failed} failed, {errors} errors из {len(results)}")

    for r in results:
        status_icon = "✅" if r.status == "passed" else "❌" if r.status == "failed" else "⚠️"
        print(f"  {status_icon} [{r.test_id}] {r.category}: {r.status} — {r.message[:80]}")

    # Сохранение результатов
    if args.save:
        path = evaluator.save(f"eval_{suite_name}")
        print(f"\nРезультаты сохранены: {path}")

    # Сохранение в память (для регрессионного анализа)
    try:
        orch.memory.save_episode(
            content=f"Оценка: {suite_name}. Passed: {passed}, Failed: {failed}, Errors: {errors}",
            source="evaluation",
            tags=["eval", suite_name, f"passed={passed}", f"failed={failed}"],
        )
    except Exception:
        pass

    return 0


def _cmd_diagnostics(orch: Orchestrator) -> int:
    print("=== Диагностика ===")
    try:
        caps = orch.model.capabilities()
        print(f"Модель загружена: {orch.model.is_loaded()}")
        print(f"Возможности: {caps}")
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка проверки модели: {exc}")
    try:
        items = orch.memory.search("тест", limit=1)
        print(f"Память доступна, записей найдено: {len(items)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка проверки памяти: {exc}")
    print(f"Инструменты зарегистрированы: {len(orch.tools.list())}")
    print("Диагностика завершена.")
    return 0


def _cmd_training_status(config: Config) -> int:
    print("=== Статус обучения ===")
    print(f"Обучение разрешено: {config.allow_training}")
    print(f"Директория обучения: {config.training_dir}")
    print("Полноценный Trainer не реализован (этап 9). Статус: not_supported.")
    return 0
