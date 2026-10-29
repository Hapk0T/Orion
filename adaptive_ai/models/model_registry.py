"""Реестр бэкендов моделей. Позволяет заменять модель без изменения ядра."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import ModelBackend, Status, TaskResult
from adaptive_ai.models.base import BaseBackend

# Регистрация бэкендов: имя -> класс
_BACKENDS: dict[str, type[BaseBackend]] = {}

# Бэкенды регистрируются при их импорте в main_impl.py
# (избегаем circular import)


def register_backend(name: str, cls: type[BaseBackend]) -> None:
    _BACKENDS[name] = cls


def list_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


def load_backend(name: str, model_path: str | None = None) -> ModelBackend:
    """Загрузить бэкенд по имени. Возвращает заглушку not_supported, если бэкенд Unknown."""
    if name not in _BACKENDS:
        return _UnsupportedBackend(name)
    instance = _BACKENDS[name]()
    result = instance.load(model_path)
    if result.status != Status.OK:
        # Возвращаем экземпляр, но отмечаем, что загрузка не удалась
        return instance
    return instance


class _UnsupportedBackend(BaseBackend):
    def __init__(self, name: str) -> None:
        self.name = name

    def capabilities(self):
        from adaptive_ai.core.contracts import ModelCapabilities

        return ModelCapabilities(name=self.name)

    def is_loaded(self) -> bool:
        return False
