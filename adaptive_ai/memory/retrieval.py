"""Интерфейсы и утилиты для извлечения контекста из памяти."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import MemoryItem, MemoryStore


class Retrieval:
    """Обёртка над MemoryStore с ограничением размера контекста."""

    def __init__(self, store: MemoryStore, max_context_tokens: int = 4096) -> None:
        self.store = store
        self.max_context_tokens = max_context_tokens

    def relevant(self, query: str, limit: int = 5) -> list[MemoryItem]:
        return self.store.search(query, limit=limit) or []

    def format_context(self, items: list[MemoryItem]) -> str:
        if not items:
            return ""
        lines = ["Релевантные воспоминания:"]
        for it in items:
            src = f" [источник: {it.source}]" if it.source else ""
            lines.append(f"- [{it.id}] {it.content}{src}")
        return "\n".join(lines)
