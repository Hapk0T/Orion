"""Извлечение контекста из памяти с ограничением размера и сохранением источников."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import MemoryItem, MemoryStore
from adaptive_ai.memory.sqlite_store import SqliteMemoryStore


class Retrieval:
    """Обёртка над MemoryStore с ограничением размера контекста и ранжированием."""

    def __init__(self, store: MemoryStore, max_context_tokens: int = 4096) -> None:
        self.store = store
        self.max_context_tokens = max_context_tokens

    def relevant(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Извлечь релевантные воспоминания с ограничением."""
        items = self.store.search(query, limit=limit) or []
        return items

    def format_context(self, items: list[MemoryItem]) -> str:
        """Сформировать контекст с источниками и статусами достоверности."""
        if not items:
            return ""
        lines = ["Релевантные воспоминания:"]
        total_chars = 0
        max_chars = self.max_context_tokens * 4  # приблизительно 4 символа на токен
        for it in items:
            src = f" [источник: {it.source}]" if it.source else ""
            status = f" [статус: {it.fact_status.value}]" if hasattr(it, "fact_status") else ""
            line = f"- [{it.id}] {it.content}{src}{status}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)
        return "\n".join(lines)

    def build_context_with_sources(
        self, query: str, limit: int = 5
    ) -> tuple[str, list[dict[str, Any]]]:
        """Извлечь воспоминания и вернуть контекст + список источников."""
        items = self.relevant(query, limit=limit)
        context = self.format_context(items)
        sources = [
            {
                "id": it.id,
                "content": it.content,
                "source": it.source,
                "fact_status": it.fact_status.value if hasattr(it, "fact_status") else "unknown",
                "type": it.type if hasattr(it, "type") else "unknown",
            }
            for it in items
        ]
        return context, sources
