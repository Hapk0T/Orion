"""Контракт Trainer и заглушка реализации (этап 9)."""

from __future__ import annotations

from typing import Any

from adaptive_ai.core.contracts import ModelBackend, Status, TaskResult, Trainer


class TrainerStub(Trainer):
    """Заглушка Trainer. Явно обозначивает ограничение: обучение не реализовано."""

    def check_compatibility(self, model_backend: ModelBackend) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Тренировка не реализована (этап 9).")

    def check_resources(self) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Проверка ресурсов не реализована.")

    def prepare_dataset(self, data: Any) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Подготовка набора не реализована.")

    def run_experiment(self, config: dict[str, Any]) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Эксперимент обучения не реализован.")

    def get_status(self, experiment_id: str) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Статус эксперимента недоступен.")

    def cancel_experiment(self, experiment_id: str) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Отмена эксперимента недоступна.")

    def evaluate(self, model_path: str, dataset: Any) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Оценка модели не реализована.")

    def promote(self, artifact_path: str) -> TaskResult:
        return TaskResult(status=Status.NOT_SUPPORTED, error="Продвижение версии не реализовано.")


def get_trainer() -> Trainer:
    return TrainerStub()
