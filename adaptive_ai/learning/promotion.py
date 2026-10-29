"""Продвижение версии модели (заглушка — этап 9)."""

from __future__ import annotations

from adaptive_ai.core.contracts import Status, TaskResult


class Promotion:
    def promote_candidate(self, artifact_path: str) -> TaskResult:
        return TaskResult(
            status=Status.NOT_SUPPORTED,
            error="Продвижение версии не реализовано (этап 9).",
        )
