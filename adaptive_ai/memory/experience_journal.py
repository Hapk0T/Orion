"""Неизменяемый журнал опыта (event sourcing).

Журнал является источником истины. Производные индексы перестраиваются из него.
Поддерживает контроль целостности, восстановление и версии схемы.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class JournalEvent:
    """Событие журнала."""

    id: str
    event_type: str
    timestamp: str
    schema_version: int
    payload: dict[str, Any]
    source_id: str | None = None
    checksum: str = ""

    def compute_checksum(self) -> str:
        data = json.dumps(
            {"id": self.id, "event_type": self.event_type, "timestamp": self.timestamp,
             "schema_version": self.schema_version, "payload": self.payload},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self.compute_checksum()


class ExperienceJournal:
    """Журнал опыта с контролем целостности.

    Поддерживает:
    - Последовательную запись событий
    - Контроль целостности (checksum)
    - Восстановление производных индексов
    - Версии схемы
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS journal_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                source_id TEXT,
                checksum TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS journal_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "INSERT OR IGNORE INTO journal_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        self._conn.commit()

    def append(self, event_type: str, payload: dict[str, Any], source_id: str | None = None) -> JournalEvent:
        """Добавить событие в журнал (неизменяемо)."""
        import uuid

        event = JournalEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            schema_version=self.SCHEMA_VERSION,
            payload=payload,
            source_id=source_id,
        )
        self._conn.execute(
            "INSERT INTO journal_events (id, event_type, timestamp, schema_version, payload, source_id, checksum) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event.id, event.event_type, event.timestamp, event.schema_version,
             json.dumps(event.payload, ensure_ascii=False), event.source_id, event.checksum),
        )
        self._conn.commit()
        return event

    def get_events(
        self,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JournalEvent]:
        """Получить события из журнала."""
        if event_type:
            cur = self._conn.execute(
                "SELECT * FROM journal_events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (event_type, limit, offset),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM journal_events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        events = []
        for row in cur.fetchall():
            event = JournalEvent(
                id=row["id"],
                event_type=row["event_type"],
                timestamp=row["timestamp"],
                schema_version=row["schema_version"],
                payload=json.loads(row["payload"]),
                source_id=row["source_id"],
                checksum=row["checksum"],
            )
            events.append(event)
        return events

    def verify_integrity(self) -> dict[str, Any]:
        """Проверить целостность журнала."""
        cur = self._conn.execute("SELECT * FROM journal_events")
        total = 0
        corrupted = 0
        for row in cur.fetchall():
            total += 1
            event = JournalEvent(
                id=row["id"],
                event_type=row["event_type"],
                timestamp=row["timestamp"],
                schema_version=row["schema_version"],
                payload=json.loads(row["payload"]),
                source_id=row["source_id"],
                checksum=row["checksum"],
            )
            if event.compute_checksum() != event.checksum:
                corrupted += 1
        return {
            "total_events": total,
            "corrupted": corrupted,
            "integrity_ok": corrupted == 0,
        }

    def count(self, event_type: str | None = None) -> int:
        if event_type:
            cur = self._conn.execute("SELECT COUNT(*) FROM journal_events WHERE event_type = ?", (event_type,))
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM journal_events")
        return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
