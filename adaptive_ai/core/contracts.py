"""Контракты (интерфейсы) ключевых подсистем Adaptive Intelligence.

Все зависимости направляются к интерфейсам, а не к конкретным реализациям.
Если модуль ещё не реализован, он возвращает статус `not_supported`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Status(str, Enum):
    """Статусы операций."""

    OK = "ok"
    NOT_SUPPORTED = "not_supported"
    ERROR = "error"
    CANCELLED = "cancelled"
    INSUFFICIENT_DATA = "insufficient_data"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONSTRAINT_VIOLATED = "constraint_violated"


class TaskStatus(str, Enum):
    """Статусы задач."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FactStatus(str, Enum):
    """Статусы достоверности фактов."""

    CONFIRMED = "confirmed"      # подтверждённое утверждение
    USER_CLAIM = "user_claim"    # пользовательское утверждение
    HYPOTHESIS = "hypothesis"    # гипотеза
    MODEL_INFERENCE = "model_inference"  # вывод модели
    REFUTED = "refuted"          # опровергнутый факт


@dataclass
class TaskResult:
    """Стандартизированный результат действия/задачи."""

    status: Status
    data: Any = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None


@dataclass
class MemoryItem:
    """Единица воспоминания (факт/событие)."""

    id: str
    type: str
    content: str
    source: str | None = None
    entity_id: str | None = None
    fact_status: FactStatus = FactStatus.USER_CLAIM
    created_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    supersedes: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCapabilities:
    """Возможности бэкенда модели."""

    supports_state: bool = False
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_adapters: bool = False
    supports_training: bool = False
    supports_state_save: bool = False
    supports_state_restore: bool = False
    context_window: int = 4096
    name: str = "unknown"
    version: str = "unknown"


@runtime_checkable
class ModelBackend(Protocol):
    """Интерфейс адаптера языковой модели."""

    name: str

    def load(self, model_path: str | None = None) -> TaskResult:
        """Загрузить модель."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> TaskResult:
        """Сгенерировать текст."""

    def capabilities(self) -> ModelCapabilities:
        """Вернуть информацию о возможностях модели."""

    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель."""


@runtime_checkable
class MemoryStore(Protocol):
    """Интерфейс долговременной памяти."""

    def save_episode(self, content: str, source: str | None = None, **kwargs) -> TaskResult:
        ...

    def save_fact(self, item: MemoryItem) -> TaskResult:
        ...

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        ...

    def get(self, item_id: str) -> MemoryItem | None:
        ...

    def update(self, item_id: str, **kwargs) -> TaskResult:
        ...

    def delete(self, item_id: str) -> TaskResult:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class Tool(Protocol):
    """Интерфейс инструмента."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    errors: list[str]
    required_permissions: list[str]
    max_time_seconds: int
    max_memory_mb: int

    def execute(self, arguments: dict[str, Any]) -> TaskResult:
        """Выполнить инструмент."""


@runtime_checkable
class ToolRegistry(Protocol):
    """Реестр инструментов."""

    def register(self, tool: Tool) -> None:
        ...

    def get(self, name: str) -> Tool | None:
        ...

    def list(self) -> list[Tool]:
        ...

    def call(self, name: str, arguments: dict[str, Any]) -> TaskResult:
        ...


@runtime_checkable
class Trainer(Protocol):
    """Контракт обучения."""

    def check_compatibility(self, model_backend: ModelBackend) -> TaskResult:
        ...

    def check_resources(self) -> TaskResult:
        ...

    def prepare_dataset(self, data: Any) -> TaskResult:
        ...

    def run_experiment(self, config: dict[str, Any]) -> TaskResult:
        ...

    def get_status(self, experiment_id: str) -> TaskResult:
        ...

    def cancel_experiment(self, experiment_id: str) -> TaskResult:
        ...

    def evaluate(self, model_path: str, dataset: Any) -> TaskResult:
        ...

    def promote(self, artifact_path: str) -> TaskResult:
        ...
