"""Движок guards — исполняемые ограничения, порождённые ошибками.

Guard — это не просто текстовое предупреждение, а проверяемое ограничение
с условиями применимости, статистикой и возможностью отката.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class GuardStatus(str, Enum):
    EXPERIMENTAL = "experimental"
    VERIFIED = "verified"
    DISABLED = "disabled"
    SUPERSEDED = "superseded"


@dataclass
class Guard:
    id: str
    version: int = 1
    pattern: str = ""
    action_type: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)
    applicability: list[str] = field(default_factory=list)
    failure_mechanism: str = ""
    cost: float = 0.0
    warning_signals: list[str] = field(default_factory=list)
    required_check: str = ""
    confirmation_level: float = 0.0
    examples_triggered: list[str] = field(default_factory=list)
    examples_not_applicable: list[str] = field(default_factory=list)
    false_positive_count: int = 0
    missed_risk_count: int = 0
    related_episode_ids: list[str] = field(default_factory=list)
    status: GuardStatus = GuardStatus.EXPERIMENTAL
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    superseded_by: str | None = None

    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.pattern}:{self.action_type}:{self.version}".encode()).hexdigest()[:16]

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Проверить, применим ли guard к ситуации. Возвращает (triggered, reason)."""
        # Проверка применимости
        for condition in self.applicability:
            if condition.lower() not in str(context).lower():
                return False, f"Условие не выполнено: {condition}"
        # Проверка паттерна
        if self.pattern and self.pattern.lower() in str(context).lower():
            return True, f"Guard '{self.id}' сработал: {self.failure_mechanism}"
        return False, ""


class GuardEngine:
    """Движок управления guards.

    Перед каждым значимым шагом проверяет применимые guards.
    Глобальные ограничения — небольшой набор.
    Остальные извлекаются из ограниченного индекса по ситуации.
    """

    def __init__(self, max_guards: int = 100) -> None:
        self._guards: list[Guard] = []
        self._max_guards = max_guards
        self._global_guards: list[str] = []  # id глобальных guards

    def register(self, guard: Guard) -> None:
        self._guards.append(guard)
        if len(self._guards) > self._max_guards:
            # Вытесняем наименее полезные experimental guards
            self._evict_least_useful()

    def _evict_least_useful(self) -> None:
        experimental = [g for g in self._guards if g.status == GuardStatus.EXPERIMENTAL]
        if experimental:
            # Вытесняем guard с наименьшим подтверждением
            least = min(experimental, key=lambda g: g.confirmation_level)
            self._guards.remove(least)

    def find_applicable(self, context: dict[str, Any]) -> list[Guard]:
        """Найти применимые guards для ситуации (ограниченный поиск)."""
        applicable = []
        for guard in self._guards:
            if guard.status == GuardStatus.DISABLED:
                continue
            if guard.status == GuardStatus.SUPERSEDED:
                continue
            triggered, _ = guard.check(context)
            if triggered:
                applicable.append(guard)
        return applicable

    def check_before_action(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Проверить guards перед действием. Возвращает список сработавших."""
        results = []
        for guard in self.find_applicable(context):
            triggered, reason = guard.check(context)
            if triggered:
                results.append({
                    "guard_id": guard.id,
                    "version": guard.version,
                    "reason": reason,
                    "status": guard.status.value,
                    "required_check": guard.required_check,
                    "cost": guard.cost,
                })
                guard.examples_triggered.append(str(context)[:200])
        return results

    def create_from_error(
        self,
        error_case: Any,
        pattern: str,
        action_type: str,
        applicability: list[str] | None = None,
    ) -> Guard:
        """Создать guard из обнаруженной ошибки."""
        guard = Guard(
            id=f"guard-{len(self._guards)}",
            pattern=pattern,
            action_type=action_type,
            applicability=applicability or [],
            failure_mechanism=error_case.description if hasattr(error_case, "description") else str(error_case),
            related_episode_ids=getattr(error_case, "related_episode_ids", []),
            status=GuardStatus.EXPERIMENTAL,
        )
        self.register(guard)
        return guard

    def get_guard(self, guard_id: str) -> Guard | None:
        for g in self._guards:
            if g.id == guard_id:
                return g
        return None

    def promote(self, guard_id: str) -> bool:
        """Продвинуть guard в статус VERIFIED (требует тестов)."""
        guard = self.get_guard(guard_id)
        if guard and guard.false_positive_count == 0 and len(guard.examples_triggered) >= 3:
            guard.status = GuardStatus.VERIFIED
            return True
        return False

    def disable(self, guard_id: str) -> bool:
        guard = self.get_guard(guard_id)
        if guard:
            guard.status = GuardStatus.DISABLED
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_guards": len(self._guards),
            "by_status": {
                "experimental": sum(1 for g in self._guards if g.status == GuardStatus.EXPERIMENTAL),
                "verified": sum(1 for g in self._guards if g.status == GuardStatus.VERIFIED),
                "disabled": sum(1 for g in self._guards if g.status == GuardStatus.DISABLED),
                "superseded": sum(1 for g in self._guards if g.status == GuardStatus.SUPERSEDED),
            },
            "total_triggers": sum(len(g.examples_triggered) for g in self._guards),
            "total_false_positives": sum(g.false_positive_count for g in self._guards),
        }
