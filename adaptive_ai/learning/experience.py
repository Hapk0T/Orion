"""Сохранение и накопление опыта выполнения задач.

Отделяет подтверждённые успехи от неизвестных результатов.
Формирует обучающие примеры из подтверждённого опыта.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExperienceQuality(str, Enum):
    """Качество обучающего сигнала."""

    CONFIRMED = "confirmed"          # подтверждённый успех
    LIKELY = "likely"                # вероятный успех
    UNKNOWN = "unknown"              # неизвестный результат
    FAILED = "failed"                # подтверждённая неудача
    PARTIAL = "partial"              # частичный успех


class CorrectionLevel(str, Enum):
    """Уровень коррекции (из ТЗ дополнения, раздел 8.1)."""

    MEMORY = "memory"       # Уровень A — память и доказательства
    SKILL = "skill"         # Уровень B — алгоритмы и навыки
    WEIGHTS = "weights"     # Уровень C — обучаемые параметры


@dataclass
class Experience:
    """Полная запись опыта выполнения задачи."""

    id: str
    task_id: str
    input_conditions: Any
    plan: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer: str | None = None
    quality: ExperienceQuality = ExperienceQuality.UNKNOWN
    confirmation_source: str | None = None
    reusable: bool = False
    correction_level: CorrectionLevel = CorrectionLevel.MEMORY
    provenance: list[str] = field(default_factory=list)  # ссылки на исходные записи
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def fingerprint(self) -> str:
        """Хэш для обнаружения повторяющихся задач."""
        data = f"{self.input_conditions}:{self.final_answer}:{self.quality.value}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]

    def to_training_example(self) -> dict[str, Any] | None:
        """Преобразовать в обучающий пример (только для подтверждённых)."""
        if self.quality not in (ExperienceQuality.CONFIRMED, ExperienceQuality.LIKELY):
            return None
        return {
            "input": self.input_conditions,
            "output": self.final_answer,
            "quality": self.quality.value,
            "source": self.confirmation_source,
            "provenance": self.provenance,
        }


class ExperienceStore:
    """Хранилище опыта с поиском, повторным использованием и формированием обучающих примеров."""

    def __init__(self) -> None:
        self._experiences: list[Experience] = []
        self._fingerprints: dict[str, int] = {}  # fingerprint -> count

    def load_from_journal(self, db_path: str) -> int:
        """Загрузить опыт из ExperienceJournal (SQLite)."""
        try:
            from adaptive_ai.memory.experience_journal import ExperienceJournal

            journal = ExperienceJournal(db_path)
            events = journal.get_events(event_type="task_completed", limit=1000)
            journal.close()
            count = 0
            for event in events:
                payload = event.payload
                exp = Experience(
                    id=event.id,
                    task_id=payload.get("task_id", ""),
                    input_conditions=payload.get("request", ""),
                    plan=payload.get("plan", []),
                    actions=payload.get("actions", []),
                    tool_results=payload.get("used_tools", []),
                    errors=[payload.get("error")] if payload.get("error") else [],
                    final_answer=payload.get("result"),
                    quality=_map_status_to_quality(payload.get("status", "unknown")),
                    confirmation_source=payload.get("status"),
                    tags=["journal"],
                    created_at=event.timestamp,
                )
                self.add(exp)
                count += 1
            return count
        except Exception:  # noqa: BLE001
            return 0

    def add(self, exp: Experience) -> None:
        self._experiences.append(exp)
        fp = exp.fingerprint()
        self._fingerprints[fp] = self._fingerprints.get(fp, 0) + 1

    def find_reusable(self, query: str) -> list[Experience]:
        """Найти переиспользуемые опыты по запросу."""
        return [e for e in self._experiences if e.reusable and query.lower() in str(e.input_conditions).lower()]

    def confirmed_only(self) -> list[Experience]:
        """Только подтверждённые опыты."""
        return [e for e in self._experiences if e.quality == ExperienceQuality.CONFIRMED]

    def failed_only(self) -> list[Experience]:
        """Только неудачные опыты."""
        return [e for e in self._experiences if e.quality == ExperienceQuality.FAILED]

    def recurring_tasks(self, min_count: int = 2) -> list[tuple[str, int]]:
        """Найти повторяющиеся задачи (по fingerprint)."""
        return [(fp, count) for fp, count in self._fingerprints.items() if count >= min_count]

    def recurring_errors(self) -> list[Experience]:
        """Найти повторяющиеся ошибки (опыты с ошибками, которые повторяются)."""
        error_exps = [e for e in self._experiences if e.errors]
        # Группируем по типу ошибки
        error_patterns: dict[str, list[Experience]] = {}
        for e in error_exps:
            for err in e.errors:
                error_patterns.setdefault(err[:50], []).append(e)
        return [e for exps in error_patterns.values() if len(exps) >= 2 for e in exps]

    def training_examples(self) -> list[dict[str, Any]]:
        """Сформировать обучающие примеры из подтверждённого опыта."""
        examples = []
        for exp in self._experiences:
            ex = exp.to_training_example()
            if ex is not None:
                examples.append(ex)
        return examples

    def assess_quality(self, exp: Experience, verification_result: bool | None = None) -> ExperienceQuality:
        """Оценить качество опыта на основе результатов."""
        if verification_result is True:
            return ExperienceQuality.CONFIRMED
        if verification_result is False:
            return ExperienceQuality.FAILED
        if exp.errors:
            return ExperienceQuality.FAILED
        if exp.final_answer and exp.confirmation_source:
            return ExperienceQuality.LIKELY
        if exp.final_answer:
            return ExperienceQuality.PARTIAL
        return ExperienceQuality.UNKNOWN

    def stats(self) -> dict[str, Any]:
        """Статистика опыта."""
        from collections import Counter

        quality_counts = Counter(e.quality.value for e in self._experiences)
        return {
            "total": len(self._experiences),
            "by_quality": dict(quality_counts),
            "reusable": sum(1 for e in self._experiences if e.reusable),
            "with_errors": sum(1 for e in self._experiences if e.errors),
            "training_examples": len(self.training_examples()),
            "recurring_tasks": len(self.recurring_tasks()),
        }


def _map_status_to_quality(status: str) -> ExperienceQuality:
    """Маппинг статуса задачи в качество опыта."""
    mapping = {
        "completed": ExperienceQuality.CONFIRMED,
        "failed": ExperienceQuality.FAILED,
        "cancelled": ExperienceQuality.FAILED,
    }
    return mapping.get(status, ExperienceQuality.UNKNOWN)


def create_experience_from_task(task: Any, quality: ExperienceQuality = ExperienceQuality.UNKNOWN) -> Experience:
    """Создать Experience из Task оркестратора."""
    import uuid

    return Experience(
        id=str(uuid.uuid4()),
        task_id=task.id,
        input_conditions=task.request,
        plan=task.plan,
        actions=task.steps,
        tool_results=task.used_tools,
        errors=[task.error] if task.error else [],
        final_answer=task.result,
        quality=quality,
        confirmation_source=task.outcome.kind if task.outcome else None,
        reusable=False,
        tags=["task", task.status.value if hasattr(task.status, "value") else str(task.status)],
        provenance=[f"orchestrator:{task.id}"],
    )
