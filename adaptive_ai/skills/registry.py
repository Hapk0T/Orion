"""Реестр навыков с поиском и выбором подходящего."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_ai.skills.skill import Skill


class SkillRegistry:
    """Реестр навыков с поиском по назначению и проверкой доступности."""

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._skills_dir = Path(skills_dir) if skills_dir else None
        if self._skills_dir and self._skills_dir.exists():
            self._load_from_dir(self._skills_dir)

    def _load_from_dir(self, directory: Path) -> None:
        """Загрузить навыки из JSON-файлов в директории."""
        for path in directory.rglob("*.json"):
            try:
                skill = Skill.from_file(path)
                self._skills[skill.id] = skill
            except Exception:  # noqa: BLE001
                pass

    def register(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def find_by_purpose(self, query: str) -> list[Skill]:
        """Найти навыки по назначению (простой поиск по ключевым словам)."""
        query_lower = query.lower()
        results = []
        for skill in self._skills.values():
            if query_lower in skill.purpose.lower() or query_lower in skill.description.lower() or query_lower in skill.name.lower():
                results.append(skill)
        return results

    def find_by_tool(self, tool_name: str) -> list[Skill]:
        """Найти навыки, использующие указанный инструмент."""
        return [s for s in self._skills.values() if tool_name in s.allowed_tools or not s.allowed_tools]

    def check_availability(self, skill_id: str, available_tools: list[str]) -> bool:
        """Проверить, доступны ли все инструменты навыка."""
        skill = self.get(skill_id)
        if not skill:
            return False
        for tool in skill.allowed_tools:
            if tool not in available_tools:
                return False
        return True

    def select_best(self, query: str, available_tools: list[str]) -> Skill | None:
        """Выбрать наиболее подходящий навык для запроса."""
        candidates = self.find_by_purpose(query)
        # Фильтруем по доступности инструментов
        available = [s for s in candidates if self.check_availability(s.id, available_tools)]
        if not available:
            return None
        # Выбираем с наивысшей версией (простая эвристика)
        return max(available, key=lambda s: s.version)
