"""Автоматический выбор модели по размеру ОЗУ.

Сканирует папку models, определяет размер и квантование каждой модели,
выбирает наибольшую, которая помещается в 80% ОЗУ.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    """Информация о модели."""

    path: str
    name: str
    size_params: float  # в миллиардах параметров
    quantization: str  # FP16, Q8_0, Q5_1, Q4_0, etc.
    file_size_bytes: int
    estimated_ram_bytes: int  # оценка потребления RAM
    precision_rank: int  # чем выше — тем точнее (FP16=5, Q8=4, Q5=3, Q4=2, Q4_0=1)


QUANT_RANK = {
    "FP32": 6,
    "FP16": 5,
    "F16": 5,
    "Q8_0": 4,
    "Q8": 4,
    "Q5_1": 3,
    "Q5_0": 3,
    "Q5": 3,
    "Q4_1": 2,
    "Q4_0": 2,
    "Q4": 2,
}

# Коэффициент буферов поверх размера файла
BUFFER_FACTOR = 1.15


def _parse_model_name(filename: str) -> tuple[float, str]:
    """Извлечь размер модели (в млрд параметров) и квантование из имени файла."""
    name_lower = filename.lower()

    # Размер модели: 0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B
    size_match = re.search(r"(\d+\.?\d*)b", name_lower)
    size_params = float(size_match.group(1)) if size_match else 0.0

    # Квантование: Q4_0, Q5_1, Q8_0, FP16, F16, etc.
    quant = "FP16"  # по умолчанию
    for q in ["fp32", "fp16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]:
        if q in name_lower:
            quant = q.upper().replace("F16", "FP16").replace("FP32", "FP32")
            break
    # Нормализация
    quant_map = {
        "FP16": "FP16",
        "F16": "FP16",
        "FP32": "FP32",
        "Q8_0": "Q8_0",
        "Q8": "Q8_0",
        "Q5_1": "Q5_1",
        "Q5_0": "Q5_0",
        "Q5": "Q5_1",
        "Q4_1": "Q4_1",
        "Q4_0": "Q4_0",
        "Q4": "Q4_0",
    }
    quant = quant_map.get(quant, quant)

    return size_params, quant


def scan_models(models_dir: str | Path) -> list[ModelInfo]:
    """Сканировать папку models на .bin файлы и вернуть информацию."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    results = []
    for path in models_dir.glob("*.bin"):
        if path.name == "rwkv_vocab_v20230424.txt":
            continue
        size_params, quant = _parse_model_name(path.name)
        file_size = path.stat().st_size
        estimated_ram = int(file_size * BUFFER_FACTOR)
        precision_rank = QUANT_RANK.get(quant, 1)

        results.append(ModelInfo(
            path=str(path),
            name=path.stem,
            size_params=size_params,
            quantization=quant,
            file_size_bytes=file_size,
            estimated_ram_bytes=estimated_ram,
            precision_rank=precision_rank,
        ))
    return results


def select_best_model(models_dir: str | Path, ram_budget_ratio: float = 0.8) -> ModelInfo | None:
    """Выбрать наибольшую модель, которая помещается в ram_budget_ratio ОЗУ.

    Приоритет:
    1. Модель помещается в бюджет
    2. Наибольший размер (в параметрах)
    3. При равном размере — более высокая точность (FP16 > Q8 > Q5 > Q4)
    """
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
    except ImportError:
        total_ram = 16 * 1024**3  # предполагаем 16 GB

    budget = int(total_ram * ram_budget_ratio)

    models = scan_models(models_dir)
    if not models:
        return None

    # Фильтруем модели, которые помещаются в бюджет
    fitting = [m for m in models if m.estimated_ram_bytes <= budget]
    if not fitting:
        # Если ни одна не помещается — берём самую маленькую
        fitting = [min(models, key=lambda m: m.estimated_ram_bytes)]

    # Сортируем: сначала по размеру (убывание), потом по точности (убывание)
    fitting.sort(key=lambda m: (m.size_params, m.precision_rank), reverse=True)

    return fitting[0]


def search_best_balanced_model(
    models_dir: str | Path,
    ram_budget_ratio: float = 0.8,
    accuracy_weight: float = 0.6,
    speed_weight: float = 0.4,
) -> ModelInfo | None:
    """Найти наиболее сбалансированную по точности и скорости модель.

    Баланс = accuracy_weight * (size_params / max_size) + speed_weight * (1 - ram_usage_ratio)
    где ram_usage_ratio = estimated_ram / budget

    Предпочитает модели, которые:
    - Используют 50-80% бюджета (не слишком маленькие, не слишком большие)
    - Имеют высокую точность (FP16 > Q8 > Q5 > Q4)
    - Не слишком медленные (меньше модель = быстрее)

    Кэширует результат в data/selected_model.txt для ускорения.
    """
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
    except ImportError:
        total_ram = 16 * 1024**3

    budget = int(total_ram * ram_budget_ratio)

    models = scan_models(models_dir)
    if not models:
        return None

    # Фильтруем модели, которые помещаются в бюджет
    fitting = [m for m in models if m.estimated_ram_bytes <= budget]
    if not fitting:
        fitting = [min(models, key=lambda m: m.estimated_ram_bytes)]

    if len(fitting) == 1:
        return fitting[0]

    # Нормализация размера
    max_size = max(m.size_params for m in fitting) if fitting else 1.0
    if max_size == 0:
        max_size = 1.0

    # Вычисляем баланс для каждой модели
    scored = []
    for m in fitting:
        size_norm = m.size_params / max_size  # 0..1
        ram_usage_ratio = m.estimated_ram_bytes / budget  # 0..1
        # Предпочтение моделей, использующих 50-80% бюджета
        if ram_usage_ratio < 0.3:
            usage_score = ram_usage_ratio / 0.3  # штраф за слишком маленькую
        elif ram_usage_ratio <= 0.8:
            usage_score = 1.0  # идеальный диапазон
        else:
            usage_score = 1.0 - (ram_usage_ratio - 0.8) / 0.2  # штраф за слишком большую

        # Точность: нормализованная по точности
        accuracy_score = (m.precision_rank / 6.0) * size_norm

        # Скорость: меньше RAM = быстрее (обратная зависимость)
        speed_score = 1.0 - ram_usage_ratio

        # Итоговый баланс
        balance = (
            accuracy_weight * accuracy_score
            + speed_weight * speed_score
            + 0.2 * usage_score  # бонус за оптимальное использование
        )

        scored.append((m, balance))

    # Сортируем по балансу (убывание)
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[0][0]


def get_cached_model(models_dir: str | Path, cache_file: str | Path = "data/selected_model.txt") -> str | None:
    """Получить путь к кэшированной модели, если файл существует и модель есть."""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        return None
    try:
        cached_path = cache_path.read_text(encoding="utf-8").strip()
        if cached_path and Path(cached_path).exists():
            return cached_path
    except Exception:  # noqa: BLE001
        pass
    return None


def cache_model(model_path: str, cache_file: str | Path = "data/selected_model.txt") -> None:
    """Сохранить выбор модели в кэш."""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(model_path, encoding="utf-8")


def select_model_with_cache(
    models_dir: str | Path,
    ram_budget_ratio: float = 0.8,
    cache_file: str | Path = "data/selected_model.txt",
) -> ModelInfo | None:
    """Выбрать модель с использованием кэша.

    Если кэш существует и модель есть — используем кэш (не пересканируем).
    Иначе — сканируем и кэшируем результат.
    """
    cached = get_cached_model(models_dir, cache_file)
    if cached:
        # Проверяем, что кэшированная модель всё ещё существует
        if Path(cached).exists():
            # Находим ModelInfo для кэшированной модели
            models = scan_models(models_dir)
            for m in models:
                if m.path == cached:
                    return m
    # Кэш не найден или устарел — сканируем
    best = search_best_balanced_model(models_dir, ram_budget_ratio)
    if best is not None:
        cache_model(best.path, cache_file)
    return best


def get_model_info_summary(models_dir: str | Path) -> dict[str, Any]:
    """Получить сводку по всем моделям."""
    models = scan_models(models_dir)
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
        available_ram = psutil.virtual_memory().available
    except ImportError:
        total_ram = 16 * 1024**3
        available_ram = 8 * 1024**3

    budget = int(total_ram * 0.8)
    fitting = [m for m in models if m.estimated_ram_bytes <= budget]

    return {
        "total_ram_gb": round(total_ram / 1024**3, 1),
        "available_ram_gb": round(available_ram / 1024**3, 1),
        "budget_gb": round(budget / 1024**3, 1),
        "models_found": len(models),
        "models_fitting": len(fitting),
        "models": [
            {
                "name": m.name,
                "size_params": m.size_params,
                "quantization": m.quantization,
                "file_size_gb": round(m.file_size_bytes / 1024**3, 2),
                "estimated_ram_gb": round(m.estimated_ram_bytes / 1024**3, 2),
                "fits": m.estimated_ram_bytes <= budget,
            }
            for m in sorted(models, key=lambda m: m.size_params, reverse=True)
        ],
    }
