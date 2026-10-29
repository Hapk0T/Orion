"""Экспериментальная пластичность архитектуры — исследовательский стенд.

Это отдельная исследовательская ветка. Она не должна задерживать работу
основного приложения.

На первом этапе — документация, интерфейсы и отдельный экспериментальный стенд.
Не создаём генератор произвольных нейросетей до появления проверяемой задачи.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ModuleType(str, Enum):
    """Типы обучаемых модулей."""

    ADAPTER = "adapter"          # LoRA-адаптер
    SPECIALIST = "specialist"    # специализированный модуль для типа задач
    ROUTER = "router"            # маршрутизатор
    MEMORY_MODULE = "memory_module"  # модуль памяти


class ExperimentStatus(str, Enum):
    """Статус эксперимента."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class LearnableModule:
    """Описание обучаемого модуля."""

    id: str
    module_type: ModuleType
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    input_dim: int = 0
    output_dim: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "experimental"


@dataclass
class RoutingDecision:
    """Решение маршрутизатора о выборе модуля."""

    task_type: str
    selected_module_id: str
    confidence: float = 0.0
    alternatives: list[str] = field(default_factory=list)
    reason: str = ""


class ModuleRouter:
    """Маршрутизатор модулей для конкретных типов задач.

    На первом этапе — простая эвристика по ключевым словам.
    В будущем — обучаемая маршрутизация.
    """

    def __init__(self) -> None:
        self._modules: dict[str, LearnableModule] = {}
        self._routing_rules: dict[str, str] = {}  # task_type -> module_id

    def register_module(self, module: LearnableModule) -> None:
        self._modules[module.id] = module

    def add_routing_rule(self, task_type: str, module_id: str) -> None:
        self._routing_rules[task_type] = module_id

    def route(self, task_description: str) -> RoutingDecision:
        """Выбрать модуль для задачи (эвристика)."""
        task_lower = task_description.lower()
        # Простая эвристика по ключевым словам
        task_type = "general"
        if any(w in task_lower for w in ["математ", "вычисл", "расчёт", "арифмет"]):
            task_type = "math"
        elif any(w in task_lower for w in ["код", "программ", "python", "функция"]):
            task_type = "code"
        elif any(w in task_lower for w in ["перевод", "язык", "english", "русск"]):
            task_type = "language"
        elif any(w in task_lower for w in ["память", "факт", "запомни"]):
            task_type = "memory"

        selected = self._routing_rules.get(task_type, "")
        if not selected and self._modules:
            selected = next(iter(self._modules.keys()))

        alternatives = [m for m in self._routing_rules.values() if m != selected]

        return RoutingDecision(
            task_type=task_type,
            selected_module_id=selected,
            confidence=0.5 if selected else 0.0,
            alternatives=alternatives,
            reason=f"Эвристическое определение типа: {task_type}",
        )

    def list_modules(self) -> list[LearnableModule]:
        return list(self._modules.values())


@dataclass
class ArchitectureExperiment:
    """Эксперимент с изменением архитектуры."""

    id: str
    name: str
    description: str
    modules_added: list[str] = field(default_factory=list)
    modules_removed: list[str] = field(default_factory=list)
    routing_changes: dict[str, str] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    experiment_metrics: dict[str, float] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PLANNED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    conclusion: str = ""

    def is_improvement(self) -> bool:
        """Проверить, улучшил ли эксперимент метрики."""
        if not self.baseline_metrics or not self.experiment_metrics:
            return False
        # Сравниваем по ключевым метрикам (больше = лучше)
        for key in self.baseline_metrics:
            if key in self.experiment_metrics:
                if self.experiment_metrics[key] < self.baseline_metrics[key]:
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "modules_added": self.modules_added,
            "modules_removed": self.modules_removed,
            "routing_changes": self.routing_changes,
            "baseline_metrics": self.baseline_metrics,
            "experiment_metrics": self.experiment_metrics,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "conclusion": self.conclusion,
            "is_improvement": self.is_improvement(),
        }


class ArchitectureExperimentRunner:
    """Запуск и управление экспериментами с архитектурой."""

    def __init__(self, experiments_dir: str | Path = "C:/Orion/experiments") -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, ArchitectureExperiment] = {}

    def create_experiment(
        self,
        name: str,
        description: str,
        modules_added: list[str] | None = None,
        modules_removed: list[str] | None = None,
        routing_changes: dict[str, str] | None = None,
    ) -> ArchitectureExperiment:
        import uuid

        exp = ArchitectureExperiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            modules_added=modules_added or [],
            modules_removed=modules_removed or [],
            routing_changes=routing_changes or {},
        )
        self._experiments[exp.id] = exp
        return exp

    def run_experiment(self, exp_id: str, baseline_metrics: dict[str, float]) -> ArchitectureExperiment | None:
        """Запустить эксперимент (на первом этапе — только запись метрик)."""
        exp = self._experiments.get(exp_id)
        if not exp:
            return None
        exp.status = ExperimentStatus.RUNNING
        exp.baseline_metrics = baseline_metrics
        # На первом этапе — только запись, без реального изменения архитектуры
        exp.experiment_metrics = baseline_metrics.copy()
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now(timezone.utc).isoformat()
        exp.conclusion = "Эксперимент завершён (первый этап — только запись метрик)."
        # Сохраняем
        exp_path = self.experiments_dir / f"{exp_id}.json"
        exp_path.write_text(json.dumps(exp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return exp

    def get_experiment(self, exp_id: str) -> ArchitectureExperiment | None:
        return self._experiments.get(exp_id)

    def list_experiments(self) -> list[ArchitectureExperiment]:
        return list(self._experiments.values())


def get_router() -> ModuleRouter:
    return ModuleRouter()


def get_experiment_runner() -> ArchitectureExperimentRunner:
    return ArchitectureExperimentRunner()
