"""Мост к MCP-серверам через официальный Python SDK.

Подключается через отдельный адаптер, не привязывает оркестратор
к одному MCP-серверу или поставщику моделей.
"""

from __future__ import annotations

import json
from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult


class McpBridge:
    """Адаптер к MCP-серверам.

    Если SDK mcp не установлен — возвращает not_supported.
    """

    def __init__(self) -> None:
        self._available = False
        self._sessions: dict[str, Any] = {}
        self._check_available()

    def _check_available(self) -> None:
        try:
            import mcp  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def list_servers(self) -> list[str]:
        return list(self._sessions.keys())

    def connect(self, server_name: str, command: str, args: list[str] | None = None) -> TaskResult:
        """Подключиться к MCP-серверу."""
        if not self._available:
            return TaskResult(
                status=Status.NOT_SUPPORTED,
                error="MCP SDK не установлен. Установите: pip install mcp",
            )
        # В v0.1 — заглушка. Реальное подключение через mcp.ClientSession
        # будет реализовано при наличии конкретного MCP-сервера.
        return TaskResult(
            status=Status.NOT_SUPPORTED,
            error="MCP-подключение будет реализовано при наличии конкретного сервера.",
        )

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> TaskResult:
        """Вызвать инструмент на MCP-сервере."""
        if not self._available:
            return TaskResult(
                status=Status.NOT_SUPPORTED,
                error="MCP SDK не установлен. Установите: pip install mcp",
            )
        if server not in self._sessions:
            return TaskResult(
                status=Status.ERROR,
                error=f"MCP-сервер не подключён: {server}",
            )
        return TaskResult(
            status=Status.NOT_SUPPORTED,
            error="MCP-вызов будет реализован при наличии подключённого сервера.",
        )

    def disconnect(self, server: str) -> TaskResult:
        if server in self._sessions:
            del self._sessions[server]
        return TaskResult(status=Status.OK)
