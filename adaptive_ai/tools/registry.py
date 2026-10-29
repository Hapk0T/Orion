"""Реестр инструментов."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult, Tool, ToolRegistry


class ToolRegistry(ToolRegistry):
    """Единый реестр инструментов."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def call(self, name: str, arguments: dict[str, Any]) -> TaskResult:
        tool = self.get(name)
        if tool is None:
            return TaskResult(status=Status.ERROR, error=f"Инструмент не найден: {name}")
        try:
            return tool.execute(arguments)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))
