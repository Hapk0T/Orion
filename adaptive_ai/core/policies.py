"""Политики и правила поведения системы (безопасность, бюджет, отмена)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyPolicy:
    """Политика безопасности: что запрещено без явного разрешения пользователя."""

    allow_destructive_actions: bool = False
    allow_file_deletion: bool = False
    allow_unverified_commands: bool = False
    allow_system_settings_changes: bool = False
    allow_external_network: bool = False
    allow_unverified_internet_code: bool = False
    require_confirmation_for: list[str] = field(
        default_factory=lambda: [
            "file_delete",
            "system_change",
            "network_send",
            "unverified_command",
        ]
    )

    def is_allowed(self, action: str) -> bool:
        mapping = {
            "file_delete": self.allow_file_deletion,
            "destructive": self.allow_destructive_actions,
            "unverified_command": self.allow_unverified_commands,
            "system_change": self.allow_system_settings_changes,
            "external_network": self.allow_external_network,
            "internet_code": self.allow_unverified_internet_code,
        }
        return mapping.get(action, False)


@dataclass
class ResourcePolicy:
    """Ограничения по ресурсам для цикла действий."""

    max_steps: int = 8
    max_retries_per_action: int = 2
    max_context_tokens: int = 4096
    max_ram_mb: int = 8192
    max_total_seconds: int = 300

    def remaining_budget(self, used_steps: int, used_seconds: float) -> dict[str, Any]:
        return {
            "steps_remaining": max(0, self.max_steps - used_steps),
            "seconds_remaining": max(0, self.max_total_seconds - used_seconds),
            "budget_exhausted": used_steps >= self.max_steps
            or used_seconds >= self.max_total_seconds,
        }


@dataclass
class TaskOutcome:
    """Исход задачи — различаем типы завершения."""

    kind: str  # completed | no_data | model_error | tool_error | constraint | budget | cancelled
    message: str
    details: dict[str, Any] = field(default_factory=dict)
