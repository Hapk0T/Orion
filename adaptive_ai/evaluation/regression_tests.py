"""Регрессионные тесты."""

from __future__ import annotations

from adaptive_ai.evaluation.evaluators import Evaluator, TestCase


def regression_suite() -> list[TestCase]:
    return [
        TestCase(
            id="regress_001",
            category="факты",
            input="Столица Франции",
            expected="Париж",
            description="Точное извлечение известного факта",
        ),
        TestCase(
            id="regress_002",
            category="калькулятор",
            input="2+2",
            expected=4,
            description="Арифметический калькулятор",
        ),
    ]


def run_regressions(evaluator: Evaluator, run_fn) -> None:
    cases = regression_suite()
    evaluator.run(cases, run_fn)
