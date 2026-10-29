"""Долговременная память на основе SQLite."""

from __future__ import annotations

import json as _json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_ai.core.contracts import (
    FactStatus,
    MemoryItem,
    MemoryStore,
    Status,
    TaskResult,
)


class SqliteMemoryStore(MemoryStore):
    """Точная структурированная память на SQLite с транзакциями.

    Таблицы: episodes, facts, entities, relations, sources, decisions,
    feedback, memory_versions. Полнотекстовый поиск — через отдельные
    FTS5-таблицы (не external-content), которые заполняются явно.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                created_at TEXT NOT NULL,
                tags TEXT,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                value TEXT NOT NULL,
                entity_id TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                valid_from TEXT,
                valid_until TEXT,
                fact_status TEXT NOT NULL DEFAULT 'user_claim',
                supersedes TEXT,
                tags TEXT,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                uri TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_versions (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                previous_data TEXT
            );
            -- Отдельные FTS5-таблицы (не external-content), заполняются явно.
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(episode_id, content);
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_id, value);
            """
        )
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _uuid(self) -> str:
        import uuid

        return str(uuid.uuid4())

    def save_episode(self, content: str, source: str | None = None, **kwargs) -> TaskResult:
        try:
            item_id = self._uuid()
            now = self._now()
            tags = kwargs.get("tags", [])
            metadata = kwargs.get("metadata", {})

            self._conn.execute(
                "INSERT INTO episodes (id, content, source, created_at, tags, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, content, source, now, _json.dumps(tags, ensure_ascii=False), _json.dumps(metadata, ensure_ascii=False)),
            )
            self._conn.execute(
                "INSERT INTO episodes_fts (episode_id, content) VALUES (?, ?)",
                (item_id, content),
            )
            self._conn.commit()
            return TaskResult(status=Status.OK, data={"id": item_id})
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))

    def save_fact(self, item: MemoryItem) -> TaskResult:
        try:
            if not item.id:
                item.id = self._uuid()
            self._conn.execute(
                """INSERT INTO facts
                   (id, type, value, entity_id, source, created_at, valid_from, valid_until,
                    fact_status, supersedes, tags, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    item.type,
                    item.content,
                    item.entity_id,
                    item.source,
                    (item.created_at or datetime.now(timezone.utc)).isoformat(),
                    (item.valid_from.isoformat() if item.valid_from else None),
                    (item.valid_until.isoformat() if item.valid_until else None),
                    item.fact_status.value,
                    item.supersedes,
                    _json.dumps(item.tags, ensure_ascii=False),
                    _json.dumps(item.metadata, ensure_ascii=False),
                ),
            )
            self._conn.execute(
                "INSERT INTO facts_fts (fact_id, value) VALUES (?, ?)",
                (item.id, item.content),
            )
            self._conn.commit()
            return TaskResult(status=Status.OK, data={"id": item.id})
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """Полнотекстовый поиск по эпизодам и фактам."""
        results: list[MemoryItem] = []
        seen_ids: set[str] = set()
        try:
            cur = self._conn.cursor()
            # Поиск по эпизодам
            cur.execute(
                """SELECT e.id, e.content, e.source, e.created_at
                   FROM episodes_fts f
                   JOIN episodes e ON e.id = f.episode_id
                   WHERE episodes_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            )
            for row in cur.fetchall():
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    results.append(
                        MemoryItem(
                            id=row["id"],
                            type="episode",
                            content=row["content"],
                            source=row["source"],
                            created_at=row["created_at"],
                            tags=[],
                            fact_status=FactStatus.USER_CLAIM,
                        )
                    )
            # Поиск по фактам
            cur.execute(
                """SELECT fct.id, fct.value, fct.source, fct.created_at, fct.fact_status, fct.entity_id
                   FROM facts_fts f
                   JOIN facts fct ON fct.id = f.fact_id
                   WHERE facts_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            )
            for row in cur.fetchall():
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    results.append(
                        MemoryItem(
                            id=row["id"],
                            type="fact",
                            content=row["value"],
                            source=row["source"],
                            entity_id=row["entity_id"],
                            created_at=row["created_at"],
                            fact_status=FactStatus(row["fact_status"]),
                            tags=[],
                        )
                    )
        except Exception:  # noqa: BLE001
            pass
        return results[:limit]

    def get(self, item_id: str) -> MemoryItem | None:
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM episodes WHERE id = ?", (item_id,))
            row = cur.fetchone()
            if row:
                return MemoryItem(
                    id=row["id"],
                    type="episode",
                    content=row["content"],
                    source=row["source"],
                    created_at=row["created_at"],
                    tags=[],
                    fact_status=FactStatus.USER_CLAIM,
                )
            cur.execute("SELECT * FROM facts WHERE id = ?", (item_id,))
            row = cur.fetchone()
            if row:
                return MemoryItem(
                    id=row["id"],
                    type="fact",
                    content=row["value"],
                    source=row["source"],
                    entity_id=row["entity_id"],
                    created_at=row["created_at"],
                    valid_from=row["valid_from"],
                    valid_until=row["valid_until"],
                    fact_status=FactStatus(row["fact_status"]),
                    supersedes=row["supersedes"],
                    tags=[],
                )
        except Exception:  # noqa: BLE001
            return None
        return None

    def update(self, item_id: str, **kwargs) -> TaskResult:
        try:
            fields = []
            values = []
            for k, v in kwargs.items():
                if k in ("content", "value", "source", "tags", "metadata", "fact_status"):
                    fields.append(f"{k} = ?")
                    values.append(v)
            if not fields:
                return TaskResult(status=Status.OK)
            values.append(item_id)
            self._conn.execute(f"UPDATE facts SET {', '.join(fields)} WHERE id = ?", values)
            self._conn.commit()
            return TaskResult(status=Status.OK)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))

    def delete(self, item_id: str) -> TaskResult:
        try:
            self._conn.execute("DELETE FROM facts WHERE id = ?", (item_id,))
            self._conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (item_id,))
            self._conn.execute("DELETE FROM episodes WHERE id = ?", (item_id,))
            self._conn.execute("DELETE FROM episodes_fts WHERE episode_id = ?", (item_id,))
            self._conn.commit()
            return TaskResult(status=Status.OK)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
