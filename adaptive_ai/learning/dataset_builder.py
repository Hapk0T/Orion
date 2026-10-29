"""Построение обучающих наборов (заглушка — этап 9)."""

from __future__ import annotations

from adaptive_ai.core.contracts import Status, TaskResult


class DatasetBuilder:
    def build(self, experiences: list) -> TaskResult:
        return TaskResult(
            status=Status.NOT_SUPPORTED,
            error="Построение обучающих наборов не реализовано (этап 9).",
        )
