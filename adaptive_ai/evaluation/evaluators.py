"""Проверяющие средства для повторяемого запуска тестовых наборов."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from adaptive_ai.core.contracts import Status, TaskResult


@dataclass
class TestCase:
    id: str
    category: str
    input: Any
    expected: Any | None = None
    check: Callable[[Any, Any], bool] | None = None  # функция проверки
    description: str = ""


@dataclass
class TestResult:
    test_id: str
    category: str
    status: str  # passed | failed | error
    input: Any
    expected: Any
    actual: Any
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Evaluator:
    """Запускает тестовые наборы и сохраняет результаты."""

    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[TestResult] = []

    def run(self, test_cases: list[TestCase], run_fn: Callable[[Any], Any]) -> list[TestResult]:
        results: list[TestResult] = []
        for tc in test_cases:
            try:
                actual = run_fn(tc.input)
                if tc.check is not None:
                    passed = tc.check(actual, tc.expected)
                else:
                    passed = actual == tc.expected
                status = "passed" if passed else "failed"
                result = TestResult(
                    test_id=tc.id,
                    category=tc.category,
                    status=status,
                    input=tc.input,
                    expected=tc.expected,
                    actual=actual,
                    message="" if passed else f"ожидалось {tc.expected}, получено {actual}",
                )
            except Exception as exc:  # noqa: BLE001
                result = TestResult(
                    test_id=tc.id,
                    category=tc.category,
                    status="error",
                    input=tc.input,
                    expected=tc.expected,
                    actual=None,
                    message=str(exc),
                )
            results.append(result)
            self.results.append(result)
        return results

    def save(self, name: str) -> Path:
        path = self.results_dir / f"{name}.json"
        data = [r.__dict__ for r in self.results]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for r in self.results:
            summary[r.status] = summary.get(r.status, 0) + 1
        return summary
