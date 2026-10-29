"""Запуск тестов в разрешённой рабочей области."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult, Tool


class RunTestsTool(Tool):
    name = "run_tests"
    description = "Запуск pytest в разрешённой рабочей области."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    output_schema = {"type": "object"}
    errors = ["test_error"]
    required_permissions: list[str] = []
    max_time_seconds = 60
    max_memory_mb = 256

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        self._allowed = [Path(d).resolve() for d in (allowed_dirs or [])]

    def execute(self, arguments: dict[str, Any]) -> TaskResult:
        target = Path(arguments.get("path", "")).resolve()
        if not any(str(target).startswith(str(d)) for d in self._allowed):
            return TaskResult(status=Status.ERROR, error="Путь не в разрешённой рабочей области")
        try:
            res = subprocess.run(
                ["pytest", str(target), "-q"],
                capture_output=True,
                text=True,
                timeout=self.max_time_seconds,
            )
            return TaskResult(
                status=Status.OK if res.returncode == 0 else Status.ERROR,
                data={"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode},
            )
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))
