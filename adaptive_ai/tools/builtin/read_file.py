"""Чтение файлов из разрешённых директорий."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult, Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Чтение текстового файла из разрешённой директории."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    output_schema = {"type": "object"}
    errors = ["file_not_found", "permission_denied"]
    required_permissions: list[str] = []
    max_time_seconds = 5
    max_memory_mb = 32

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        self._allowed = [Path(d).resolve() for d in (allowed_dirs or [])]

    def execute(self, arguments: dict[str, Any]) -> TaskResult:
        file_path = Path(arguments.get("path", "")).resolve()
        if not any(str(file_path).startswith(str(d)) for d in self._allowed):
            return TaskResult(status=Status.ERROR, error="Путь не в разрешённых директориях")
        if not file_path.is_file():
            return TaskResult(status=Status.ERROR, error="Файл не найден")
        try:
            text = file_path.read_text(encoding="utf-8")
            return TaskResult(status=Status.OK, data={"content": text, "path": str(file_path)})
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))
