"""Сохранение опыта выполнения задач."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Experience:
    task_id: str
    input_conditions: Any
    plan: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    errors: list[str]
    final_answer: str | None
    quality_score: float | None
    confirmation_source: str | None
    reusable: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExperienceStore:
    """Хранилище опыта. В v0.1 запись и чтение, без автоматического обучения."""

    def __init__(self) -> None:
        self._experiences: list[Experience] = []

    def add(self, exp: Experience) -> None:
        self._experiences.append(exp)

    def find_reusable(self, query: str) -> list[Experience]:
        return [e for e in self._experiences if e.reusable]

    def confirmed_only(self) -> list[Experience]:
        return [e for e in self._experiences if e.quality_score is not None]
