"""Форматы запросов, ответов, размышлений и памяти.

Стандартизированные форматы для взаимодействия между компонентами.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ResponseFormat(str, Enum):
    """Формат ответа модели."""

    TEXT = "text"           # Свободный текст
    JSON = "json"           # JSON (для инструментов)
    STRUCTURED = "structured"  # Структурированный (thinking + answer + tools)


@dataclass
class ThinkingBlock:
    """Блок размышлений модели."""

    content: str
    verified: bool = False
    evidence: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """Вызов инструмента."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "call_id": self.call_id}


@dataclass
class ToolResult:
    """Результат вызова инструмента."""

    call_id: str
    tool_name: str
    status: str  # ok, error
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class StructuredResponse:
    """Структурированный ответ модели."""

    thinking: ThinkingBlock | None = None
    thinking_stages: list[str] = field(default_factory=list)
    answer: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def is_finished(self) -> bool:
        return len(self.tool_calls) == 0 and len(self.answer) > 0

    @classmethod
    def parse(cls, text: str) -> "StructuredResponse":
        """Парсить структурированный ответ из текста модели."""
        resp = cls(raw_text=text)

        # Извлекаем блоки думаю: (многоэтапное размышление)
        thinking_matches = re.findall(r"думаю:\s*\n?(.*?)(?=думаю:|ответ:|$)", text, re.DOTALL)
        if thinking_matches:
            combined = "\n---\n".join(t.strip() for t in thinking_matches if t.strip())
            resp.thinking = ThinkingBlock(content=combined)
            # Последний этап — это финальное размышление перед ответом
            resp.thinking_stages = [t.strip() for t in thinking_matches if t.strip()]

        # Извлекаем ответ: блок
        answer_match = re.search(r"ответ:\s*\n?(.*)", text, re.DOTALL)
        if answer_match:
            resp.answer = answer_match.group(1).strip()
        # Иначе не устанавливаем answer — попробуем JSON парсинг ниже

        # Извлекаем tool_calls
        tool_match = re.search(r"<tool_calls>(.*?)</tool_calls>", text, re.DOTALL)
        if tool_match:
            try:
                calls_data = json.loads(tool_match.group(1).strip())
                for call in calls_data:
                    resp.tool_calls.append(ToolCall(
                        name=call.get("name", ""),
                        arguments=call.get("arguments", {}),
                        call_id=call.get("call_id", ""),
                    ))
            except (json.JSONDecodeError, ValueError):
                pass

        # Пытаемся парсить JSON (старый формат)
        if not resp.tool_calls and not resp.answer:
            try:
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    data = json.loads(text[start_idx:end_idx + 1])
                    if "tool" in data and data["tool"]:
                        resp.tool_calls.append(ToolCall(
                            name=data["tool"],
                            arguments=data.get("arguments", {}),
                        ))
                    if "answer" in data:
                        resp.answer = data["answer"]
                    if "finish" in data:
                        pass  # обрабатывается оркестратором
            except (json.JSONDecodeError, ValueError):
                pass

        # Если answer всё ещё пустой — берём весь текст
        if not resp.answer:
            resp.answer = text.strip()

        return resp

    def to_prompt_context(self) -> str:
        """Сформировать контекст для промпта."""
        parts = []
        if self.thinking:
            parts.append(f"Размышления: {self.thinking.content}")
        if self.answer:
            parts.append(f"Ответ: {self.answer}")
        if self.tool_calls:
            parts.append(f"Инструменты: {[c.name for c in self.tool_calls]}")
        return "\n".join(parts)


@dataclass
class PromptTemplate:
    """Шаблон промпта для RWKV World."""

    system_prompt: str = ""
    context: str = ""
    user_message: str = ""
    memory_context: str = ""
    skill_context: str = ""

    def render(self) -> str:
        """Сформировать промпт по формату RWKV World.

        Формат RWKV World (по рекомендациям разработчиков):
        - Разделители: \\n\\n между блоками
        - Нет пробелов после ':'
        - Структура: System/Context → User → Assistant:
        """
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        if self.memory_context:
            parts.append(f"Memory: {self.memory_context}")
        if self.skill_context:
            parts.append(f"Skill: {self.skill_context}")
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.user_message:
            parts.append(f"User: {self.user_message}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def render_chat(self) -> str:
        """Формат для чата (без инструкций JSON)."""
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        if self.memory_context:
            parts.append(f"Memory: {self.memory_context}")
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.user_message:
            parts.append(f"User: {self.user_message}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


@dataclass
class MemoryFact:
    """Факт в памяти с расширенными метаданными."""

    id: str
    content: str
    fact_status: str = "user_claim"
    confidence: float = 0.5
    verification_method: str = "none"  # none, source_check, tool_check, user_confirmed, model_inference
    source: str | None = None
    entity_id: str | None = None
    created_at: str = ""
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: str | None = None
    contradicts: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_memory_item(self) -> Any:
        """Конвертировать в MemoryItem."""
        from adaptive_ai.core.contracts import MemoryItem, FactStatus

        return MemoryItem(
            id=self.id,
            type="fact",
            content=self.content,
            source=self.source,
            entity_id=self.entity_id,
            fact_status=FactStatus(self.fact_status),
            tags=self.tags,
            metadata={
                "confidence": self.confidence,
                "verification_method": self.verification_method,
                "contradicts": self.contradicts,
                "derived_from": self.derived_from,
            },
        )

    @classmethod
    def from_memory_item(cls, item: Any) -> "MemoryFact":
        """Создать из MemoryItem."""
        meta = getattr(item, "metadata", {}) or {}
        return cls(
            id=item.id,
            content=item.content,
            fact_status=item.fact_status.value if hasattr(item.fact_status, "value") else str(item.fact_status),
            confidence=meta.get("confidence", 0.5),
            verification_method=meta.get("verification_method", "none"),
            source=item.source,
            entity_id=item.entity_id,
            created_at=str(item.created_at) if item.created_at else "",
            tags=item.tags or [],
            contradicts=meta.get("contradicts", []),
            derived_from=meta.get("derived_from", []),
        )


@dataclass
class TaskRecord:
    """Стандартизированная запись задачи."""

    id: str
    request: str
    status: str = "pending"
    created_at: str = ""
    model_used: str = ""
    config_version: str = ""
    thinking: str = ""
    answer: str = ""
    confidence: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    retrieved_memory: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    correction_level: str = "memory"  # memory, skill, weights
    outcome: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": self.request,
            "status": self.status,
            "created_at": self.created_at,
            "model_used": self.model_used,
            "config_version": self.config_version,
            "thinking": self.thinking,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "retrieved_memory": self.retrieved_memory,
            "error": self.error,
            "correction_level": self.correction_level,
            "outcome": self.outcome,
        }
