"""Миграции схемы базы памяти (заглушка — схемы создаются в sqlite_store)."""

from __future__ import annotations

MIGRATIONS: list[str] = []


def apply_migrations() -> None:
    """Применить миграции. В v0.1 схема создаётся один раз в SqliteMemoryStore."""
    pass
