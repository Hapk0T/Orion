"""Поиск в памяти."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult, Tool


class SearchMemoryTool(Tool):
    name = "search_memory"
    description = "Поиск в долговременной памяти по полнотекстовому запросу."
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    output_schema = {"type": "object"}
    errors = ["memory_error"]
    required_permissions: list[str] = []
    max_time_seconds = 5
    max_memory_mb = 32

    def __init__(self, memory_store) -> None:
        self._store = memory_store

    def execute(self, arguments: dict[str, Any]) -> TaskResult:
        query = arguments.get("query", "")
        if not query:
            return TaskResult(status=Status.ERROR, error="query обязателен")
        try:
            items = self._store.search(query, limit=10) or []
            return TaskResult(
                status=Status.OK,
                data={"results": [{"id": i.id, "content": i.content, "source": i.source} for i in items]},
            )
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))
