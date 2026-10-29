"""Модель навыка (skill)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """Версионируемый навык."""

    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    purpose: str = ""
    constraints: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    algorithm: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    changelog: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def fingerprint(self) -> str:
        data = f"{self.id}:{self.version}:{self.algorithm}:{self.allowed_tools}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_file(self, path: str | Path) -> None:
        """Сохранить навык в JSON-файл."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "Skill":
        """Загрузить навык из JSON-файла."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def has_tool_access(self, tool_name: str) -> bool:
        """Проверить, разрешён ли инструмент для этого навыка."""
        if not self.allowed_tools:
            return True  # если не указаны — разрешены все
        return tool_name in self.allowed_tools

    def validate(self) -> list[str]:
        """Проверить корректность навыка."""
        issues = []
        if not self.name:
            issues.append("Отсутствует имя навыка")
        if not self.purpose:
            issues.append("Отсутствует описание назначения")
        if not self.algorithm:
            issues.append("Отсутствует алгоритм")
        if not self.test_cases:
            issues.append("Отсутствуют тестовые случаи")
        return issues
