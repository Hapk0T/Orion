"""Оркестратор — управляет циклом планирования, исполнения и проверки задач."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .contracts import (
    MemoryStore,
    ModelBackend,
    Status,
    TaskResult,
    TaskStatus,
    ToolRegistry,
)
from .error_monitor import ErrorMonitor
from .guard_engine import GuardEngine
from .policies import ResourcePolicy, SafetyPolicy, TaskOutcome
from .resource_monitor import ResourceMonitor


@dataclass
class Task:
    """Задача и её выполнение."""

    id: str
    request: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    plan: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    thinking_stages: list[str] = field(default_factory=list)
    outcome: TaskOutcome | None = None
    result: str | None = None
    error: str | None = None
    retrieved_memory: list[dict[str, Any]] = field(default_factory=list)
    retrieved_sources: list[dict[str, Any]] = field(default_factory=list)
    used_tools: list[dict[str, Any]] = field(default_factory=list)
    model_used: str | None = None
    config_version: str | None = None


class Orchestrator:
    """Оркестратор цикла решения задач.

    Не зависит от конкретной реализации SQLite, LLM или MCP — только от интерфейсов.
    """

    def __init__(
        self,
        model: ModelBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        safety: SafetyPolicy | None = None,
        resources: ResourcePolicy | None = None,
        error_monitor: ErrorMonitor | None = None,
        guard_engine: GuardEngine | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        self.model = model
        self.memory = memory
        self.tools = tools
        self.safety = safety or SafetyPolicy()
        self.resources = resources or ResourcePolicy()
        self.error_monitor = error_monitor or ErrorMonitor()
        self.guard_engine = guard_engine or GuardEngine()
        self.resource_monitor = resource_monitor or ResourceMonitor(max_ram_mb=resources.max_ram_mb if resources else 8192)
        self._tasks: dict[str, Task] = {}
        self._current: Task | None = None
        self._experience_store = None  # ленивая инициализация
        self._skill_registry = None  # ленивая инициализация
        self._selected_skill = None

    @property
    def skill_registry(self):
        if self._skill_registry is None:
            from adaptive_ai.skills.registry import SkillRegistry

            self._skill_registry = SkillRegistry()
        return self._skill_registry

    @property
    def experience_store(self):
        if self._experience_store is None:
            from adaptive_ai.learning.experience import ExperienceStore

            self._experience_store = ExperienceStore()
        return self._experience_store

    # ------------------------------------------------------------------ #
    # Задачи
    # ------------------------------------------------------------------ #
    def create_task(self, request: str) -> Task:
        task_id = str(uuid.uuid4())
        task = Task(id=task_id, request=request)
        self._tasks[task_id] = task
        self._current = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    # ------------------------------------------------------------------ #
    # Основной цикл
    # ------------------------------------------------------------------ #
    def solve(self, request: str) -> TaskResult:
        """Выполнить полный цикл решения задачи."""
        task = self.create_task(request)
        start = time.monotonic()

        try:
            # 1. Принять запрос — уже создано
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.now(timezone.utc)

            # 2. Определить тип задачи и ограничения (заглушка — расширяется навыками)
            task.plan = [{"step": 1, "action": "analyze", "description": "Анализ запроса"}]

            # 3. Извлечь релевантную память с источниками
            memories, sources = self._retrieve_with_sources(request, task)
            task.retrieved_memory = [m.__dict__ if hasattr(m, "__dict__") else m for m in memories]
            task.retrieved_sources = sources

            # 3b. Выбрать подходящий навык
            available_tools = [t.name for t in self.tools.list()]
            skill = self.skill_registry.select_best(request, available_tools)
            self._selected_skill = skill
            if skill:
                task.plan.append({"step": 0, "action": "skill", "description": f"Навык: {skill.name} v{skill.version}"})
                # Ограничить инструменты навыком
                if skill.allowed_tools:
                    context_skill = f"Навык: {skill.name}. Разрешённые инструменты: {skill.allowed_tools}"
                else:
                    context_skill = f"Навык: {skill.name}."
                context = context_skill

            # 4. Сформировать ограниченный контекст
            context = self._build_context(task, memories, sources) + (f"\n{context_skill}" if skill else "")

            # 5. Получить план или следующее действие модели
            step_count = 0
            while step_count < self.resources.max_steps:
                step_count += 1
                elapsed = time.monotonic() - start
                budget = self.resources.remaining_budget(step_count, elapsed)
                if budget["budget_exhausted"]:
                    task.outcome = TaskOutcome(
                        kind="budget", message="Исчерпан бюджет шагов или времени", details=budget
                    )
                    task.status = TaskStatus.FAILED
                    return TaskResult(status=Status.BUDGET_EXHAUSTED, error=task.outcome.message, task_id=task.id)

                # Проверка бюджета ОЗУ
                ram_check = self.resource_monitor.check_budget()
                if not ram_check["ok"]:
                    task.outcome = TaskOutcome(
                        kind="constraint",
                        message=f"Превышен бюджет ОЗУ: {ram_check['rss_mb']} MB > {ram_check['budget_mb']} MB",
                        details=ram_check,
                    )
                    task.status = TaskStatus.FAILED
                    return TaskResult(status=Status.CONSTRAINT_VIOLATED, error=task.outcome.message, task_id=task.id)

                # 5a. Спросить модель, какое действие выполнить
                action = self._ask_model_for_action(task, context)
                if action is None:
                    task.outcome = TaskOutcome(kind="model_error", message="Модель не вернула действие")
                    task.status = TaskStatus.FAILED
                    return TaskResult(status=Status.ERROR, error=task.outcome.message, task_id=task.id)

                # 6. Определяем инструмент
                tool_name = action.get("tool")

                # 5b. Проверка guards перед действием
                guard_context = {
                    "tool": tool_name,
                    "arguments": action.get("arguments", {}),
                    "task": task.request,
                    "step": step_count,
                }
                guard_hits = self.guard_engine.check_before_action(guard_context)
                if guard_hits:
                    # Guard сработал — записываем в контекст и продолжаем с предупреждением
                    for hit in guard_hits:
                        context += f"\nGuard '{hit['guard_id']}' сработал: {hit['reason']}. Требуется: {hit['required_check']}"
                    # Если guard критический (cost > 0), останавливаем
                    critical = [h for h in guard_hits if h.get("cost", 0) > 0]
                    if critical:
                        task.outcome = TaskOutcome(
                            kind="constraint",
                            message=f"Guard заблокировал действие: {critical[0]['reason']}",
                            details={"guards": critical},
                        )
                        task.status = TaskStatus.FAILED
                        return TaskResult(status=Status.CONSTRAINT_VIOLATED, error=task.outcome.message, task_id=task.id)

                # 7. При необходимости вызвать инструмент
                if tool_name:
                    tool_result = self._execute_tool(task, tool_name, action.get("arguments", {}))
                    task.used_tools.append(
                        {"tool": tool_name, "arguments": action.get("arguments", {}), "result": tool_result.__dict__}
                    )
                    if tool_result.status == Status.ERROR:
                        # Регистрируем ошибку в ErrorMonitor
                        self.error_monitor.check_external(
                            [{"tool": tool_name, "status": "error", "error": tool_result.error}],
                            [],
                        )
                        # 9. При ошибке скорректировать инструкцию или план (до лимита повторов)
                        retry_count = action.get("_retries", 0)
                        if retry_count < self.resources.max_retries_per_action:
                            action["_retries"] = retry_count + 1
                            context += f"\nОшибка инструмента: {tool_result.error}. Повторите с исправлением."
                            continue
                        task.outcome = TaskOutcome(kind="tool_error", message=tool_result.error or "Ошибка инструмента")
                        task.status = TaskStatus.FAILED
                        return TaskResult(status=Status.ERROR, error=task.outcome.message, task_id=task.id)

                    # 7. Получить и проверить результат
                    context += f"\nРезультат инструмента {tool_name}: {tool_result.data}"

                # 8. Решить, завершить или продолжить
                if action.get("finish"):
                    task.result = action.get("answer", "")
                    task.outcome = TaskOutcome(kind="completed", message="Задача завершена успешно")
                    task.status = TaskStatus.COMPLETED
                    self._record_task(task)
                    return TaskResult(status=Status.OK, data=task.result, task_id=task.id)

            # Цикл исчерпан без завершения
            task.outcome = TaskOutcome(kind="budget", message="Достигнут лимит шагов без завершения")
            task.status = TaskStatus.FAILED
            self._record_task(task)
            return TaskResult(status=Status.BUDGET_EXHAUSTED, error=task.outcome.message, task_id=task.id)

        except Exception as exc:  # noqa: BLE001 — верхний уровень, логируем и возвращаем
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            return TaskResult(status=Status.ERROR, error=str(exc), task_id=task.id)

    # ------------------------------------------------------------------ #
    # Вспомогательные методы цикла
    # ------------------------------------------------------------------ #
    def _retrieve_memory(self, query: str, task: Task) -> list[Any]:
        try:
            return self.memory.search(query, limit=5) or []
        except Exception:  # noqa: BLE001
            return []

    def _retrieve_with_sources(self, query: str, task: Task) -> tuple[list[Any], list[dict[str, Any]]]:
        """Извлечь воспоминания с сохранением источников."""
        memories = self._retrieve_memory(query, task)
        sources = []
        for m in memories:
            sources.append({
                "id": getattr(m, "id", None),
                "content": getattr(m, "content", str(m)),
                "source": getattr(m, "source", None),
                "fact_status": getattr(m, "fact_status", None),
                "fact_status_value": getattr(m, "fact_status", None).value if hasattr(m, "fact_status") and m.fact_status else "unknown",
            })
        return memories, sources

    def _build_context(self, task: Task, memories: list[Any], sources: list[dict[str, Any]] | None = None) -> str:
        parts = [f"Задача: {task.request}"]
        if memories:
            parts.append("Релевантные воспоминания:")
            for i, m in enumerate(memories):
                content = getattr(m, "content", str(m))
                source = getattr(m, "source", None)
                fact_status = getattr(m, "fact_status", None)
                status_str = ""
                if fact_status:
                    status_val = fact_status.value if hasattr(fact_status, "value") else str(fact_status)
                    status_str = f" [статус: {status_val}]"
                src_str = f" [источник: {source}]" if source else ""
                parts.append(f"- {content}{src_str}{status_str}")
        # Ограничение размера контекста
        context = "\n".join(parts)
        max_chars = self.resources.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (контекст обрезан)"
        return context

    def _ask_model_for_action(self, task: Task, context: str) -> dict[str, Any] | None:
        """Попросить модель определить следующее действие. Возвращает dict или None."""
        from .formats import PromptTemplate, StructuredResponse

        template = PromptTemplate(
            system_prompt=(
                "Ты помощник, который планирует действия и использует инструменты.\n"
                "Формат ответа:\n"
                "думаю: (твои размышления о задаче)\n"
                "думаю: (следующий этап размышлений, если нужен)\n"
                "ответ: (финальный ответ или JSON с действием)\n\n"
                "Для действий с инструментами используй JSON в ответе:\n"
                '{"tool": "имя_инструмента", "arguments": {...}, "finish": false}\n'
                "Если инструменты не нужны: {\"tool\": null, \"finish\": true, \"answer\": \"текст ответа\"}"
            ),
            context=context,
            user_message=(
                "Определи следующее действие.\n"
                "думаю: (размышления о задаче)\n"
                "ответ: (JSON с действием или финальный ответ)"
            ),
        )
        prompt = template.render()
        caps = self.model.capabilities()
        result = self.model.generate(
            prompt=prompt,
            system_prompt=None,
            max_tokens=256,
            temperature=0.3,
        )
        if result.status != Status.OK or not result.data:
            return None
        text = result.data if isinstance(result.data, str) else str(result.data)

        # Парсим через StructuredResponse
        structured = StructuredResponse.parse(text)
        # Сохраняем размышления в задачу
        if structured.thinking:
            task.thinking = structured.thinking.content
            task.thinking_stages = structured.thinking_stages
        if structured.tool_calls:
            tc = structured.tool_calls[0]
            return {
                "tool": tc.name if tc.name else None,
                "arguments": tc.arguments,
                "finish": False,
                "answer": structured.answer,
                "thinking": structured.thinking.content if structured.thinking else "",
            }
        if structured.answer:
            return {
                "tool": None,
                "arguments": {},
                "finish": True,
                "answer": structured.answer,
                "thinking": structured.thinking.content if structured.thinking else "",
            }

        # Fallback: пытаемся распарсить JSON
        import json as _json

        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return _json.loads(text[start_idx : end_idx + 1])
        except (_json.JSONDecodeError, ValueError):
            pass
        return {"tool": None, "arguments": {}, "finish": True, "answer": text.strip()}

    def _execute_tool(self, task: Task, name: str, arguments: dict[str, Any]) -> TaskResult:
        # Проверка безопасности
        if name in ("file_delete", "system_change", "network_send", "unverified_command"):
            if not self.safety.is_allowed(name):
                return TaskResult(
                    status=Status.CONSTRAINT_VIOLATED,
                    error=f"Действие {name} запрещено политикой безопасности",
                )
        return self.tools.call(name, arguments)

    def _record_task(self, task: Task) -> None:
        """Записать итог задачи в историю (память) и журнал опыта."""
        try:
            self.memory.save_episode(
                content=f"Задача: {task.request}\nРезультат: {task.result or task.error or ''}",
                source=f"orchestrator:{task.id}",
                tags=["task", task.status.value],
            )
        except Exception:  # noqa: BLE001
            pass

        # Запись в журнал опыта (если доступен)
        try:
            from adaptive_ai.memory.experience_journal import ExperienceJournal

            journal = ExperienceJournal(self.memory.db_path if hasattr(self.memory, "db_path") else None)
            journal.append(
                event_type="task_completed",
                payload={
                    "task_id": task.id,
                    "request": task.request,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                    "used_tools": task.used_tools,
                    "retrieved_memory": task.retrieved_memory,
                    "retrieved_sources": task.retrieved_sources,
                },
                source_id=f"orchestrator:{task.id}",
            )
            journal.close()
        except Exception:  # noqa: BLE001
            pass

        # Создание записи опыта
        try:
            from adaptive_ai.learning.experience import create_experience_from_task, ExperienceQuality

            # Определяем качество
            if task.status.value == "completed" and task.result:
                quality = ExperienceQuality.CONFIRMED
            elif task.status.value == "failed":
                quality = ExperienceQuality.FAILED
            elif task.result:
                quality = ExperienceQuality.PARTIAL
            else:
                quality = ExperienceQuality.UNKNOWN

            exp = create_experience_from_task(task, quality=quality)
            # Определяем уровень коррекции
            if task.error and "tool" in (task.error or "").lower():
                exp.correction_level = CorrectionLevel.SKILL
            elif task.error:
                exp.correction_level = CorrectionLevel.MEMORY

            # Сохраняем в ExperienceStore (если доступен)
            if self._experience_store is None:
                from adaptive_ai.learning.experience import ExperienceStore

                self._experience_store = ExperienceStore()
            self._experience_store.add(exp)
        except Exception:  # noqa: BLE001
            pass

        # Проверка ошибок после завершения
        try:
            errors = self.error_monitor.check_internal(
                claims=[task.result or ""],
                conditions=[task.request],
                result=task.result,
            )
            # Если обнаружена повторяющаяся ошибка — создаём guard
            for err in errors:
                if err.status == "recurring_pattern":
                    self.guard_engine.create_from_error(
                        error_case=err,
                        pattern=err.description[:100],
                        action_type="block",
                        applicability=[err.error_type.value],
                    )
        except Exception:  # noqa: BLE001
            pass
