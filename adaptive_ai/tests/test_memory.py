"""Юнит-тесты памяти."""

from __future__ import annotations

import os
import tempfile

from adaptive_ai.memory.sqlite_store import SqliteMemoryStore
from adaptive_ai.core.contracts import FactStatus, MemoryItem


def test_save_and_search_episode():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        store = SqliteMemoryStore(path)
        res = store.save_episode("Тестовый эпизод о контроле качества", source="тест")
        assert res.status == "ok", res.error
        items = store.search("эпизод", limit=5)
        assert len(items) >= 1, "Должен быть найден хотя бы один эпизод"
        found = any("эпизод" in it.content for it in items)
        assert found
        store.close()
    finally:
        _cleanup_db(path)


def test_save_and_get_fact():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        store = SqliteMemoryStore(path)
        item = MemoryItem(
            id="fact-1",
            type="fact",
            content="Столица Франции — Париж",
            entity_id="entity-france",
            source="ручной ввод",
            fact_status=FactStatus.CONFIRMED,
            tags=["география"],
        )
        res = store.save_fact(item)
        assert res.status == "ok", res.error
        got = store.get("fact-1")
        assert got is not None
        assert got.content == "Столица Франции — Париж"
        assert got.fact_status == FactStatus.CONFIRMED
        store.close()
    finally:
        _cleanup_db(path)


def test_delete_fact():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        store = SqliteMemoryStore(path)
        item = MemoryItem(id="fact-del", type="fact", content="временный факт", fact_status=FactStatus.HYPOTHESIS)
        store.save_fact(item)
        assert store.get("fact-del") is not None
        res = store.delete("fact-del")
        assert res.status == "ok"
        assert store.get("fact-del") is None
        store.close()
    finally:
        _cleanup_db(path)


def test_persistence_after_reopen():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        store = SqliteMemoryStore(path)
        store.save_episode("Персистентная запись", source="тест")
        store.close()
        store2 = SqliteMemoryStore(path)
        items = store2.search("Персистентная", limit=5)
        assert len(items) >= 1
        store2.close()
    finally:
        _cleanup_db(path)


def _cleanup_db(path: str) -> None:
    import os

    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        try:
            if os.path.exists(p):
                os.unlink(p)
        except PermissionError:
            pass
