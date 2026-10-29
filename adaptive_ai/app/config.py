"""Конфигурация системы. Пути к данным, моделям и БД настраиваются независимо от кода."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    """Глобальная конфигурация. Все пути — абсолютные, без хардкода."""

    # Директории пользовательских данных (не в Git, не в образ по умолчанию)
    data_dir: Path = field(default_factory=lambda: Path(_env("ADAPTIVE_AI_DATA_DIR", "C:/Orion/data")))
    models_dir: Path = field(default_factory=lambda: Path(_env("ADAPTIVE_AI_MODELS_DIR", "C:/Orion/models")))
    db_path: Path = field(default_factory=lambda: Path(_env("ADAPTIVE_AI_DB_PATH", "C:/Orion/data/adaptive_ai.db")))

    # Модель
    model_backend: str = _env("ADAPTIVE_AI_MODEL_BACKEND", "rwkv_cpp")
    model_path: str | None = _env("ADAPTIVE_AI_MODEL_PATH", None) or None

    # Оркестратор
    max_steps: int = int(_env("ADAPTIVE_AI_MAX_STEPS", "8"))
    max_retries: int = int(_env("ADAPTIVE_AI_MAX_RETRIES", "2"))

    # Ограничения ресурсов
    max_context_tokens: int = 4096
    max_ram_mb: int = 8192

    # Сетевые функции (отключены по умолчанию — offline-first)
    allow_network: bool = _env("ADAPTIVE_AI_ALLOW_NETWORK", "0").lower() in ("1", "true", "yes")

    # Обучение
    allow_training: bool = _env("ADAPTIVE_AI_ALLOW_TRAINING", "0").lower() in ("1", "true", "yes")
    training_dir: Path = field(default_factory=lambda: Path(_env("ADAPTIVE_AI_TRAINING_DIR", "C:/Orion/training")))

    def ensure_dirs(self) -> None:
        """Создаёт необходимые директории, если их нет."""
        for d in (self.data_dir, self.models_dir, self.training_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls) -> "Config":
        return cls()
