"""Построение обучающих наборов из опыта и данных памяти."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult
from adaptive_ai.learning.experience import Experience, ExperienceStore


@dataclass
class TrainingDataset:
    """Обучающий набор с версией и метаданными."""

    id: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "experience"
    checksum: str = ""

    def compute_checksum(self) -> str:
        import hashlib

        data = json.dumps(self.examples, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.checksum = self.compute_checksum()
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "checksum": self.checksum,
            "example_count": len(self.examples),
            "examples": self.examples,
        }

    @classmethod
    def load(cls, path: str | Path) -> "TrainingDataset":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        ds = cls(
            id=data["id"],
            examples=data["examples"],
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", ""),
            source=data.get("source", "experience"),
            checksum=data.get("checksum", ""),
        )
        return ds


class DatasetBuilder:
    """Построение обучающих наборов из опыта."""

    def __init__(self, min_quality: str = "confirmed") -> None:
        self.min_quality = min_quality

    def build_from_experience(self, store: ExperienceStore) -> TrainingDataset:
        """Построить набор из подтверждённого опыта."""
        import uuid

        examples = store.training_examples()
        # Фильтрация по качеству
        filtered = [e for e in examples if e.get("quality") in ("confirmed", "likely")]
        dataset = TrainingDataset(
            id=str(uuid.uuid4()),
            examples=filtered,
            source="experience",
        )
        dataset.checksum = dataset.compute_checksum()
        return dataset

    def build_from_memory(self, memory_store: Any, query: str = "") -> TrainingDataset:
        """Построить набор из фактов памяти."""
        import uuid

        examples = []
        try:
            items = memory_store.search(query, limit=100) if query else []
            for item in items:
                examples.append({
                    "input": getattr(item, "content", str(item)),
                    "output": getattr(item, "content", str(item)),
                    "quality": "confirmed",
                    "source": getattr(item, "source", "memory"),
                })
        except Exception:  # noqa: BLE001
            pass
        dataset = TrainingDataset(
            id=str(uuid.uuid4()),
            examples=examples,
            source="memory",
        )
        dataset.checksum = dataset.compute_checksum()
        return dataset

    def validate(self, dataset: TrainingDataset) -> TaskResult:
        """Проверить набор на корректность."""
        if not dataset.examples:
            return TaskResult(status=Status.ERROR, error="Набор пуст")
        # Проверка дубликатов
        seen = set()
        duplicates = 0
        for ex in dataset.examples:
            key = json.dumps(ex, sort_keys=True, ensure_ascii=False)
            if key in seen:
                duplicates += 1
            seen.add(key)
        if duplicates > 0:
            return TaskResult(
                status=Status.OK,
                data={"valid": True, "warnings": [f"Найдено {duplicates} дубликатов"]},
            )
        return TaskResult(status=Status.OK, data={"valid": True, "warnings": []})
