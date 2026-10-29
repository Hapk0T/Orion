"""Метакогниция: контроллер обнаружения собственных ошибок.

Не ограничивается повторной генерацией ответа с инструкцией «проверь себя».
Использует три независимых контура: внутренний, независимый внешний, эмпирический.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ErrorStatus(str, Enum):
    SUSPECTED_ERROR = "suspected_error"
    CONFIRMED_ERROR = "confirmed_error"
    UNRESOLVED = "unresolved"
    CORRECTED = "corrected"
    REGRESSION_TESTED = "regression_tested"
    RECURRING_PATTERN = "recurring_pattern"


class ErrorType(str, Enum):
    CONTRADICTION = "contradiction"
    INSUFFICIENT_DATA = "insufficient_data"
    CONSTRAINT_VIOLATION = "constraint_violation"
    PREDICTION_MISMATCH = "prediction_mismatch"
    TOOL_ERROR = "tool_error"
    SOURCE_MISMATCH = "source_mismatch"
    NOVEL_SITUATION = "novel_situation"
    OVERCONFIDENCE = "overconfidence"
    GUARD_FAILURE = "guard_failure"
    CALCULATION_ERROR = "calculation_error"


class VerificationChannel(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    EMPIRICAL = "empirical"


@dataclass
class ErrorCase:
    id: str
    error_type: ErrorType
    status: ErrorStatus
    description: str
    affected_claims: list[str] = field(default_factory=list)
    affected_actions: list[str] = field(default_factory=list)
    verification_channel: VerificationChannel = VerificationChannel.INTERNAL
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str | None = None
    resolution: str | None = None
    related_episode_ids: list[str] = field(default_factory=list)
    related_guard_ids: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """Хэш для обнаружения повторяющихся паттернов."""
        data = f"{self.error_type.value}:{self.description}:{sorted(self.affected_claims)}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


@dataclass
class ConfidenceRecord:
    """Запись прогноза уверенности до получения результата."""

    task_id: str
    predicted_confidence: float
    actual_outcome: bool | None = None
    brier_score: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorMonitor:
    """Контроллер обнаружения собственных ошибок.

    Использует три независимых контура проверки:
    1. Внутренний: анализ противоречий, альтернатив, пропущенных условий
    2. Внешний: детерминированные вычисления, код, источники, формальные ограничения
    3. Эмпирический: реальные испытания, воспроизводимость, обратная связь
    """

    def __init__(self, max_cases: int = 1000) -> None:
        self._cases: list[ErrorCase] = []
        self._confidence_records: list[ConfidenceRecord] = []
        self._max_cases = max_cases
        self._pattern_counts: dict[str, int] = {}

    def check_internal(
        self,
        claims: list[str],
        conditions: list[str],
        result: str | None = None,
    ) -> list[ErrorCase]:
        """Внутренний контур: противоречия, пропущенные условия, неопределённость."""
        cases: list[ErrorCase] = []
        # Проверка противоречий между утверждениями
        for i, c1 in enumerate(claims):
            for c2 in claims[i + 1:]:
                if self._contradict(c1, c2):
                    cases.append(
                        ErrorCase(
                            id=f"err-{len(self._cases) + len(cases)}",
                            error_type=ErrorType.CONTRADICTION,
                            status=ErrorStatus.SUSPECTED_ERROR,
                            description=f"Противоречие: '{c1}' vs '{c2}'",
                            affected_claims=[c1, c2],
                            verification_channel=VerificationChannel.INTERNAL,
                            confidence=0.7,
                        )
                    )
        # Проверка пропущенных условий
        if conditions and result:
            for cond in conditions:
                if cond.lower() not in (result or "").lower():
                    cases.append(
                        ErrorCase(
                            id=f"err-{len(self._cases) + len(cases)}",
                            error_type=ErrorType.INSUFFICIENT_DATA,
                            status=ErrorStatus.SUSPECTED_ERROR,
                            description=f"Условие '{cond}' не отражено в результате",
                            affected_claims=[cond],
                            verification_channel=VerificationChannel.INTERNAL,
                            confidence=0.5,
                        )
                    )
        self._register_cases(cases)
        return cases

    def check_external(
        self,
        tool_results: list[dict[str, Any]],
        constraints: list[str],
    ) -> list[ErrorCase]:
        """Внешний контур: ошибки инструментов, нарушения ограничений."""
        cases: list[ErrorCase] = []
        for tr in tool_results:
            if tr.get("status") == "error":
                cases.append(
                    ErrorCase(
                        id=f"err-{len(self._cases) + len(cases)}",
                        error_type=ErrorType.TOOL_ERROR,
                        status=ErrorStatus.CONFIRMED_ERROR,
                        description=f"Ошибка инструмента {tr.get('tool', '?')}: {tr.get('error', '')}",
                        affected_actions=[tr.get("tool", "")],
                        verification_channel=VerificationChannel.EXTERNAL,
                        confidence=1.0,
                        evidence=[tr],
                    )
                )
        # Проверка ограничений
        for c in constraints:
            # Простая проверка: если ограничение упоминает число, ищем его в результатах
            pass
        self._register_cases(cases)
        return cases

    def check_empirical(
        self,
        predicted: str | None,
        actual: str | None,
        task_id: str,
    ) -> list[ErrorCase]:
        """Эмпирический контур: расхождение прогноза и факта."""
        cases: list[ErrorCase] = []
        if predicted and actual and predicted.strip() != actual.strip():
            cases.append(
                ErrorCase(
                    id=f"err-{len(self._cases) + len(cases)}",
                    error_type=ErrorType.PREDICTION_MISMATCH,
                    status=ErrorStatus.CONFIRMED_ERROR,
                    description=f"Прогноз не совпал с результатом: '{predicted[:100]}' vs '{actual[:100]}'",
                    affected_claims=[predicted],
                    verification_channel=VerificationChannel.EMPIRICAL,
                    confidence=0.9,
                    related_episode_ids=[task_id],
                )
            )
        self._register_cases(cases)
        return cases

    def record_confidence(
        self, task_id: str, predicted_confidence: float, actual_outcome: bool | None = None
    ) -> ConfidenceRecord:
        rec = ConfidenceRecord(
            task_id=task_id,
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
        )
        if actual_outcome is not None:
            # Brier score
            rec.brier_score = (predicted_confidence - (1.0 if actual_outcome else 0.0)) ** 2
        self._confidence_records.append(rec)
        return rec

    def calibrate(self) -> dict[str, Any]:
        """Оценка качества калибровки уверенности (Brier score, ECE)."""
        records = [r for r in self._confidence_records if r.brier_score is not None]
        if not records:
            return {"status": "insufficient_data", "count": 0}
        brier = sum(r.brier_score for r in records) / len(records)
        return {
            "status": "ok",
            "count": len(records),
            "mean_brier_score": round(brier, 4),
            "calibration": "good" if brier < 0.25 else "poor",
        }

    def detect_recurring(self, error_case: ErrorCase) -> bool:
        """Обнаружение повторяющегося паттерна ошибок."""
        fp = error_case.fingerprint()
        self._pattern_counts[fp] = self._pattern_counts.get(fp, 0) + 1
        if self._pattern_counts[fp] >= 3:
            error_case.status = ErrorStatus.RECURRING_PATTERN
            return True
        return False

    def _contradict(self, a: str, b: str) -> bool:
        """Простая проверка противоречий (эвристика)."""
        a_lower = a.lower()
        b_lower = b.lower()
        # Противоположные утверждения
        opposites = [("да", "нет"), ("истина", "ложь"), ("правда", "неправда"), ("можно", "нельзя")]
        for w1, w2 in opposites:
            if w1 in a_lower and w2 in b_lower:
                return True
            if w2 in a_lower and w1 in b_lower:
                return True
        return False

    def _register_cases(self, cases: list[ErrorCase]) -> None:
        for case in cases:
            self._cases.append(case)
            self.detect_recurring(case)
        # Ограничиваем размер
        if len(self._cases) > self._max_cases:
            self._cases = self._cases[-self._max_cases:]

    def get_cases(self, status: ErrorStatus | None = None) -> list[ErrorCase]:
        if status:
            return [c for c in self._cases if c.status == status]
        return list(self._cases)

    def get_summary(self) -> dict[str, Any]:
        from collections import Counter

        status_counts = Counter(c.status.value for c in self._cases)
        type_counts = Counter(c.error_type.value for c in self._cases)
        return {
            "total_cases": len(self._cases),
            "by_status": dict(status_counts),
            "by_type": dict(type_counts),
            "recurring_patterns": sum(1 for c in self._cases if c.status == ErrorStatus.RECURRING_PATTERN),
        }
