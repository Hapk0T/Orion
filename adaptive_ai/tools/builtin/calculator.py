"""Калькулятор."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult, Tool


class CalculatorTool(Tool):
    name = "calculator"
    description = "Простой арифметический калькулятор (только безопасные операции)."
    input_schema = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    output_schema = {"type": "object"}
    errors = ["calc_error"]
    required_permissions: list[str] = []
    max_time_seconds = 5
    max_memory_mb = 16

    def execute(self, arguments: dict[str, Any]) -> TaskResult:
        expression = arguments.get("expression", "")
        if not expression:
            return TaskResult(status=Status.ERROR, error="expression обязателен")
        try:
            # Безопасный eval только для арифметики
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return TaskResult(status=Status.ERROR, error="Недопустимые символы в выражении")
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S107 — безопасная арифметика с фильтром
            return TaskResult(status=Status.OK, data={"result": result})
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))
