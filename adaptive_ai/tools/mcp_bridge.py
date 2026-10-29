"""Мост к MCP (заглушка — подключается после этапа 8)."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult


class McpBridge:
    """Адаптер к MCP-серверам через официальный Python SDK (mcp).

    В v0.1 возвращает not_supported — MCP добавляется на этапе 8.
    """

    def __init__(self) -> None:
        self.available = False

    def check_available(self) -> bool:
        try:
            import mcp  # noqa: F401
            return True
        except Exception:
            return False

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> TaskResult:
        return TaskResult(
            status=Status.NOT_SUPPORTED,
            error="MCP не реализован в v0.1. Подключение через отдельный адаптер на этапе 8.",
        )
