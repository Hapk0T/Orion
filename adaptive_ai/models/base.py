"""Базовый класс и общие утилиты для бэкендов моделей."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import ModelCapabilities, Status, TaskResult


class BaseBackend:
    """Базовый бэкенд. Реализует общую логику и дефолтные not_supported операции."""

    name = "base"

    def load(self, model_path: str | None = None) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="load не реализован")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="generate не реализован")

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.name)

    def is_loaded(self) -> bool:
        return False

    def save_state(self, path: str) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="save_state не реализован")

    def restore_state(self, path: str) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="restore_state не реализован")

    def load_adapter(self, path: str) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="load_adapter не реализован")

    def train(self, dataset: Any) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="train не реализован")
