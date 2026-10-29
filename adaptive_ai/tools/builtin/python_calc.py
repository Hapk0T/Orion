"""Ограниченные Python-вычисления."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult, Tool


class PythonCalcTool(Tool):
    name = "python_calc"
    description = "Ограниченное выполнение Python-выражений (без импортов, без доступа к файловой системе)."
    input_schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
    output_schema = {"type": "object"}
    errors = ["calc_error"]
    required_permissions: list[str] = []
    max_time_seconds = 10
    max_memory_mb = 64

    def execute(self, arguments: dict[str, Any]) -> TaskResult:
        code = arguments.get("code", "")
        if not code:
            return TaskResult(status=Status.ERROR, error="code обязателен")
        banned = ["import", "open", "exec", "eval(", "__", "subprocess", "os.", "sys."]
        if any(b in code for b in banned):
            return TaskResult(status=Status.ERROR, error="Запрещённые конструкции в коде")
        try:
            result = eval(code, {"__builtins__": {}}, {})  # noqa: S107 — ограниченная среда
            return TaskResult(status=Status.OK, data={"result": result})
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))
