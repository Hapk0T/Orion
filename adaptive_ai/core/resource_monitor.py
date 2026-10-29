"""Мониторинг ресурсов — потребление ОЗУ, CPU, время.

Измеряет фактический RSS, пиковую память процесса и системное потребление.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceSnapshot:
    """Снимок потребления ресурсов."""

    timestamp: float = field(default_factory=time.time)
    rss_mb: float = 0.0          # RSS процесса (MB)
    vms_mb: float = 0.0          # Virtual memory (MB)
    system_total_mb: float = 0.0  # Всего ОЗУ (MB)
    system_available_mb: float = 0.0  # Доступно ОЗУ (MB)
    system_used_percent: float = 0.0  # Процент использования ОЗУ
    cpu_percent: float = 0.0     # Загрузка CPU (%)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rss_mb": round(self.rss_mb, 1),
            "vms_mb": round(self.vms_mb, 1),
            "system_total_mb": round(self.system_total_mb, 1),
            "system_available_mb": round(self.system_available_mb, 1),
            "system_used_percent": round(self.system_used_percent, 1),
            "cpu_percent": round(self.cpu_percent, 1),
        }


class ResourceMonitor:
    """Мониторинг потребления ресурсов."""

    def __init__(self, max_ram_mb: int = 8192) -> None:
        self.max_ram_mb = max_ram_mb
        self._snapshots: list[ResourceSnapshot] = []
        self._peak_rss_mb: float = 0.0

    def snapshot(self) -> ResourceSnapshot:
        """Сделать снимок текущего потребления."""
        snap = ResourceSnapshot()
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            snap.rss_mb = mem.rss / (1024**2)
            snap.vms_mb = mem.vms / (1024**2)
            sys_mem = psutil.virtual_memory()
            snap.system_total_mb = sys_mem.total / (1024**2)
            snap.system_available_mb = sys_mem.available / (1024**2)
            snap.system_used_percent = sys_mem.percent
            snap.cpu_percent = proc.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback без psutil
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                snap.rss_mb = usage.ru_maxrss / 1024  # KB -> MB на Linux
            except Exception:  # noqa: BLE001
                pass
        self._snapshots.append(snap)
        self._peak_rss_mb = max(self._peak_rss_mb, snap.rss_mb)
        return snap

    def check_budget(self) -> dict[str, Any]:
        """Проверить, не превышен ли бюджет ОЗУ."""
        snap = self.snapshot()
        exceeded = snap.rss_mb > self.max_ram_mb
        return {
            "ok": not exceeded,
            "rss_mb": round(snap.rss_mb, 1),
            "budget_mb": self.max_ram_mb,
            "exceeded": exceeded,
            "system_available_mb": round(snap.system_available_mb, 1),
        }

    def get_peak_rss(self) -> float:
        return self._peak_rss_mb

    def get_summary(self) -> dict[str, Any]:
        if not self._snapshots:
            return {"snapshots": 0}
        last = self._snapshots[-1]
        return {
            "snapshots": len(self._snapshots),
            "current_rss_mb": round(last.rss_mb, 1),
            "peak_rss_mb": round(self._peak_rss_mb, 1),
            "system_used_percent": round(last.system_used_percent, 1),
            "budget_mb": self.max_ram_mb,
            "budget_exceeded": last.rss_mb > self.max_ram_mb,
        }
