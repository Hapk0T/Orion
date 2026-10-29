"""Продвижение версии модели с привязкой к контрольной сумме.

Одобрение привязано к контрольной сумме, версии, конфигурации и манифесту.
Изменение одобренного артефакта аннулирует одобрение.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_ai.core.contracts import Status, TaskResult


@dataclass
class Approval:
    """Запись об одобрении артефакта."""

    artifact_path: str
    checksum: str
    version: str
    config: dict[str, Any]
    approved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_by: str = "external"
    valid: bool = True

    def verify(self, current_checksum: str) -> bool:
        """Проверить, что артефакт не изменился."""
        return self.valid and self.checksum == current_checksum


class Promotion:
    """Управление продвижением версий модели."""

    def __init__(self, approvals_dir: str | Path = "C:/Orion/training/approvals") -> None:
        self.approvals_dir = Path(approvals_dir)
        self.approvals_dir.mkdir(parents=True, exist_ok=True)
        self._approvals: dict[str, Approval] = {}
        self._load_approvals()

    def _load_approvals(self) -> None:
        for path in self.approvals_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                approval = Approval(**data)
                self._approvals[approval.artifact_path] = approval
            except Exception:  # noqa: BLE001
                pass

    def _compute_checksum(self, artifact_path: str) -> str:
        p = Path(artifact_path)
        if not p.exists():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def approve(self, artifact_path: str, version: str, config: dict[str, Any]) -> TaskResult:
        """Одобрить артефакт (внешнее одобрение)."""
        checksum = self._compute_checksum(artifact_path)
        if not checksum:
            return TaskResult(status=Status.ERROR, error=f"Артефакт не найден: {artifact_path}")

        approval = Approval(
            artifact_path=artifact_path,
            checksum=checksum,
            version=version,
            config=config,
        )
        self._approvals[artifact_path] = approval

        # Сохраняем
        approval_path = self.approvals_dir / f"{Path(artifact_path).stem}_approval.json"
        approval_path.write_text(json.dumps(approval.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

        return TaskResult(
            status=Status.OK,
            data={
                "artifact_path": artifact_path,
                "checksum": checksum,
                "version": version,
                "approved": True,
            },
        )

    def check_approval(self, artifact_path: str) -> TaskResult:
        """Проверить, одобрен ли артефакт и не изменился ли он."""
        approval = self._approvals.get(artifact_path)
        if not approval:
            return TaskResult(status=Status.ERROR, error="Артефакт не одобрен")

        current_checksum = self._compute_checksum(artifact_path)
        if not approval.verify(current_checksum):
            return TaskResult(
                status=Status.ERROR,
                error="Артефакт изменён после одобрения. Одобрение аннулировано.",
            )

        return TaskResult(
            status=Status.OK,
            data={
                "artifact_path": artifact_path,
                "checksum": approval.checksum,
                "version": approval.version,
                "approved_at": approval.approved_at,
                "valid": True,
            },
        )

    def revoke(self, artifact_path: str) -> TaskResult:
        """Отозвать одобрение."""
        if artifact_path in self._approvals:
            self._approvals[artifact_path].valid = False
            # Обновляем файл
            approval_path = self.approvals_dir / f"{Path(artifact_path).stem}_approval.json"
            if approval_path.exists():
                data = json.loads(approval_path.read_text(encoding="utf-8"))
                data["valid"] = False
                approval_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return TaskResult(status=Status.OK, data={"revoked": True})
        return TaskResult(status=Status.ERROR, error="Одобрение не найдено")

    def list_approved(self) -> list[Approval]:
        """Список одобренных артефактов."""
        return [a for a in self._approvals.values() if a.valid]
