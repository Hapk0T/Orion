"""Реализация CLI (вынесена из main.py для чистоты)."""

from __future__ import annotations

import sys
import json
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

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        answer = f"[режим без модели] Получил ваше сообщение: {prompt[:200]}"
        yield TaskResult(status=Status.OK, data=answer)


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

        # Загрузка опыта из журнала
        try:
            loaded = orch.experience_store.load_from_journal(str(config.db_path))
            if loaded > 0:
                print(f"Загружено {loaded} записей опыта из журнала.")
        except Exception:  # noqa: BLE001
            pass
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
    if cmd == "models":
        return _cmd_models(orch, args, config)
    if cmd == "search-best":
        return _cmd_search_best(orch, args, config)
    if cmd == "skills":
        return _cmd_skills(orch, args)
    if cmd == "mcp":
        return _cmd_mcp(orch, args)
    if cmd == "experience":
        return _cmd_experience(orch, args)
    if cmd == "eval":
        return _cmd_eval(orch, args, config)
    if cmd == "tools":
        return _cmd_tools(orch, args)
    if cmd == "training-status":
        return _cmd_training_status(config)
    if cmd == "train":
        return _cmd_train(orch, args, config)

    print(f"Неизвестная команда: {cmd}")
    return 1


def _cmd_chat(orch: Orchestrator, args: Any, config: Config) -> int:
    message = args.message
    if not message:
        print("Интерактивный режим диалога. Введите 'exit' for выхода.")
        while True:
            try:
                message = input("> ")
            except EOFError:
                break
            if message.strip().lower() in ("exit", "quit", "выход"):
                break
            if not message.strip():
                continue
            print("Ответ: ", end="", flush=True)
            for chunk in orch.model.generate_stream(
                prompt=message,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
            ):
                if chunk.status == "ok" and chunk.data:
                    print(chunk.data, end="", flush=True)
            print()
        return 0
    print("Ответ: ", end="", flush=True)
    for chunk in orch.model.generate_stream(
        prompt=message,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
    ):
        if chunk.status == "ok" and chunk.data:
            print(chunk.data, end="", flush=True)
    print()
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


def _cmd_experience(orch: Orchestrator, args: Any) -> int:
    store = orch.experience_store
    action = args.action or "stats"
    limit = args.limit or 10

    if action == "stats":
        stats = store.stats()
        print("=== Статистика опыта ===")
        print(f"Всего записей: {stats['total']}")
        print(f"По качеству: {stats['by_quality']}")
        print(f"Переиспользуемых: {stats['reusable']}")
        print(f"С ошибками: {stats['with_errors']}")
        print(f"Обучающих примеров: {stats['training_examples']}")
        print(f"Повторяющихся задач: {stats['recurring_tasks']}")
        return 0

    if action == "list":
        exps = store._experiences[-limit:]
        if not exps:
            print("Опыт пуст.")
            return 0
        for e in exps:
            print(f"[{e.id[:8]}] {e.quality.value} | {str(e.input_conditions)[:60]}")
            if e.final_answer:
                print(f"      ответ: {str(e.final_answer)[:80]}")
            if e.errors:
                print(f"      ошибки: {e.errors}")
        return 0

    if action == "confirmed":
        exps = store.confirmed_only()
        print(f"Подтверждённые опыты: {len(exps)}")
        for e in exps[-limit:]:
            print(f"  [{e.id[:8]}] {str(e.input_conditions)[:60]}")
        return 0

    if action == "failed":
        exps = store.failed_only()
        print(f"Неудачные опыты: {len(exps)}")
        for e in exps[-limit:]:
            print(f"  [{e.id[:8]}] {str(e.input_conditions)[:60]} | ошибки: {e.errors}")
        return 0

    if action == "training":
        examples = store.training_examples()
        print(f"Обучающие примеры: {len(examples)}")
        for ex in examples[-limit:]:
            print(f"  input: {str(ex['input'])[:60]}")
            print(f"  output: {str(ex['output'])[:60]}")
            print(f"  quality: {ex['quality']}, source: {ex['source']}")
        return 0

    print(f"Неизвестное действие: {action}. Используйте stats, list, confirmed, failed, training.")
    return 1


def _cmd_skills(orch: Orchestrator, args: Any) -> int:
    registry = orch.skill_registry
    action = args.action or "list"

    if action == "list":
        skills = registry.list()
        if not skills:
            print("Навыки не найдены. Добавьте JSON-файлы в каталог skills/.")
            return 0
        print("=== Навыки ===")
        for s in skills:
            available = registry.check_availability(s.id, [t.name for t in orch.tools.list()])
            status = "✅" if available else "❌"
            print(f"  {status} {s.id} v{s.version}: {s.name}")
            print(f"      {s.purpose[:80]}")
            print(f"      Инструменты: {s.allowed_tools or 'все'}")
            issues = s.validate()
            if issues:
                print(f"      Проблемы: {issues}")
        return 0

    if action == "view":
        skill_id = args.skill_id
        if not skill_id:
            print("Укажите ID навыка: skills view <id>")
            return 1
        skill = registry.get(skill_id)
        if not skill:
            print(f"Навык не найден: {skill_id}")
            return 1
        print(f"=== Навык: {skill.name} v{skill.version} ===")
        print(f"ID: {skill.id}")
        print(f"Назначение: {skill.purpose}")
        print(f"Ограничения: {skill.constraints}")
        print(f"Требуемые входы: {skill.required_inputs}")
        print(f"Алгоритм: {skill.algorithm[:200]}")
        print(f"Разрешённые инструменты: {skill.allowed_tools or 'все'}")
        print(f"Примеры: {len(skill.examples)}")
        print(f"Тесты: {len(skill.test_cases)}")
        print(f"Версия: {skill.version}")
        print(f"Отпечаток: {skill.fingerprint()}")
        return 0

    if action == "check":
        skill_id = args.skill_id
        if not skill_id:
            print("Укажите ID навыка: skills check <id>")
            return 1
        skill = registry.get(skill_id)
        if not skill:
            print(f"Навык не найден: {skill_id}")
            return 1
        issues = skill.validate()
        available = registry.check_availability(skill_id, [t.name for t in orch.tools.list()])
        print(f"Навык: {skill.name} v{skill.version}")
        print(f"Доступен: {'да' if available else 'нет'}")
        print(f"Проблемы: {issues if issues else 'нет'}")
        return 0

    print(f"Неизвестное действие: {action}. Используйте list, view, check.")
    return 1


def _cmd_mcp(orch: Orchestrator, args: Any) -> int:
    from adaptive_ai.tools.mcp_bridge import McpBridge

    bridge = McpBridge()
    action = args.action or "status"

    if action == "status":
        print("=== MCP-статус ===")
        print(f"SDK доступен: {bridge.is_available()}")
        print(f"Подключённые серверы: {bridge.list_servers()}")
        if not bridge.is_available():
            print("Для подключения MCP: pip install mcp")
        return 0

    if action == "list":
        print("Подключённые MCP-серверы:")
        for s in bridge.list_servers():
            print(f"  {s}")
        if not bridge.list_servers():
            print("  (нет)")
        return 0

    print(f"Неизвестное действие: {action}. Используйте status, list.")
    return 1


def _cmd_search_best(orch: Orchestrator, args: Any, config: Config) -> int:
    from adaptive_ai.models.model_selector import (
        search_best_balanced_model,
        get_model_info_summary,
        select_model_with_cache,
    )

    print("=== Поиск наиболее сбалансированной модели ===")
    print(f"Бюджет ОЗУ: {config.max_ram_ratio * 100:.0f}%")

    # Показываем все модели с балансом
    summary = get_model_info_summary(config.models_dir)
    print(f"Найдено моделей: {summary['models_found']}")
    print()

    best = search_best_balanced_model(config.models_dir, ram_budget_ratio=config.max_ram_ratio)
    if best is None:
        print("Модели не найдены.")
        return 1

    print(f"Выбрана: {best.name}")
    print(f"  Путь: {best.path}")
    print(f"  Размер: {best.size_params}B | Квантование: {best.quantization}")
    print(f"  Файл: {best.file_size_bytes / 1024**3:.2f} GB | RAM: {best.estimated_ram_bytes / 1024**3:.2f} GB")

    # Кэшируем
    from adaptive_ai.models.model_selector import cache_model
    cache_model(best.path)
    print(f"\nРезультат кэширован в data/selected_model.txt")

    return 0


def _cmd_models(orch: Orchestrator, args: Any, config: Config) -> int:
    from adaptive_ai.models.model_selector import get_model_info_summary, select_best_model

    action = args.action or "list"

    if action == "list":
        summary = get_model_info_summary(config.models_dir)
        print("=== Модели ===")
        print(f"ОЗУ всего: {summary['total_ram_gb']} GB")
        print(f"ОЗУ доступно: {summary['available_ram_gb']} GB")
        print(f"Бюджет (80%): {summary['budget_gb']} GB")
        print(f"Найдено моделей: {summary['models_found']}")
        print(f"Помещаются в бюджет: {summary['models_fitting']}")
        print()
        for m in summary["models"]:
            fits = "✅" if m["fits"] else "❌"
            print(f"  {fits} {m['name']}")
            print(f"      Размер: {m['size_params']}B | Квантование: {m['quantization']} | Файл: {m['file_size_gb']} GB | RAM: {m['estimated_ram_gb']} GB")
        return 0

    if action == "select":
        best = select_best_model(config.models_dir, ram_budget_ratio=config.max_ram_ratio)
        if best is None:
            print("Модели не найдены в", config.models_dir)
            return 1
        print(f"Выбрана модель: {best.name}")
        print(f"  Путь: {best.path}")
        print(f"  Размер: {best.size_params}B | Квантование: {best.quantization}")
        print(f"  Файл: {best.file_size_bytes / 1024**3:.2f} GB | RAM: {best.estimated_ram_bytes / 1024**3:.2f} GB")
        print(f"\nДля использования: set ADAPTIVE_AI_MODEL_PATH={best.path}")
        return 0

    if action == "search-best":
        return _cmd_search_best(orch, args, config)

    if action == "info":
        summary = get_model_info_summary(config.models_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2) if False else str(summary))
        return 0

    print(f"Неизвестное действие: {action}. Используйте list, select, info.")
    return 1


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


def _cmd_train(orch: Orchestrator, args: Any, config: Config) -> int:
    from adaptive_ai.learning.trainer import get_trainer
    from adaptive_ai.learning.dataset_builder import DatasetBuilder
    from adaptive_ai.learning.experience import ExperienceStore
    from adaptive_ai.learning.promotion import Promotion

    trainer = get_trainer(str(config.training_dir))
    action = args.action or "status"

    if action == "status":
        print("=== Статус обучения ===")
        print(f"Обучение разрешено: {config.allow_training}")
        print(f"Директория обучения: {config.training_dir}")
        # Проверка совместимости
        compat = trainer.check_compatibility(orch.model)
        print(f"Совместимость: {compat.data if compat.status == 'ok' else compat.error}")
        res_check = trainer.check_resources()
        print(f"Ресурсы: {res_check.data if res_check.status == 'ok' else res_check.error}")
        return 0

    if action == "check":
        compat = trainer.check_compatibility(orch.model)
        print(f"Совместимость: {compat.status} — {compat.data or compat.error}")
        res_check = trainer.check_resources()
        print(f"Ресурсы: {res_check.status} — {res_check.data or res_check.error}")
        return 0

    if action == "prepare":
        # Подготовить набор из опыта
        store = orch.experience_store
        builder = DatasetBuilder()
        dataset = builder.build_from_experience(store)
        val_result = builder.validate(dataset)
        prep_result = trainer.prepare_dataset(dataset.to_dict() if hasattr(dataset, "to_dict") else dataset)
        print(f"Набор: {prep_result.data if prep_result.status == 'ok' else prep_result.error}")
        if val_result.data and val_result.data.get("warnings"):
            print(f"Предупреждения: {val_result.data['warnings']}")
        return 0

    if action == "run":
        if not config.allow_training:
            print("Обучение отключено. Установите ADAPTIVE_AI_ALLOW_TRAINING=1")
            return 1
        dataset_path = args.dataset
        if not dataset_path:
            print("Укажите --dataset <path>")
            return 1
        import json
        from pathlib import Path

        ds_data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        config_exp = {
            "base_model_id": orch.model.name,
            "base_model_checksum": "unknown",
            "dataset_version": ds_data.get("id", "unknown"),
            "params": {"epochs": 1, "lr": 1e-4},
        }
        result = trainer.run_experiment(config_exp)
        print(f"Эксперимент: {result.data if result.status == 'ok' else result.error}")
        return 0

    if action == "eval":
        model_path = args.model
        if not model_path:
            print("Укажите --model <path>")
            return 1
        result = trainer.evaluate(model_path, None)
        print(f"Оценка: {result.data if result.status == 'ok' else result.error}")
        return 0

    if action == "promote":
        artifact_path = args.artifact
        if not artifact_path:
            print("Укажите --artifact <path>")
            return 1
        promotion = Promotion()
        result = promotion.approve(artifact_path, version="candidate", config={})
        print(f"Одобрение: {result.data if result.status == 'ok' else result.error}")
        return 0

    if action == "status-exp":
        exp_id = args.exp_id
        if not exp_id:
            print("Укажите --exp-id <id>")
            return 1
        result = trainer.get_status(exp_id)
        print(f"Статус: {result.data if result.status == 'ok' else result.error}")
        return 0

    print(f"Неизвестное действие: {action}. Используйте status, check, prepare, run, eval, promote, status-exp.")
    return 1


def _cmd_training_status(config: Config) -> int:
    print("=== Статус обучения ===")
    print(f"Обучение разрешено: {config.allow_training}")
    print(f"Директория обучения: {config.training_dir}")
    print("Полноценный Trainer не реализован (этап 9). Статус: not_supported.")
    return 0
