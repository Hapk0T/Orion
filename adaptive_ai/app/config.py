"""Конфигурация системы. Пути к данным, моделям и БД настраиваются независимо от кода.

Поддерживает чтение из config.txt (INI-подобный формат) и переменных окружения.
Приоритет: переменные окружения > config.txt > значения по умолчанию.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _load_config_file(path: str | Path) -> dict[str, dict[str, str]]:
    """Загрузить конфиг из INI-подобного файла."""
    result: dict[str, dict[str, str]] = {}
    current_section = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    result.setdefault(current_section, {})
                    continue
                if "=" in line and current_section:
                    key, value = line.split("=", 1)
                    result[current_section][key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return result


def _cfg(config_data: dict[str, dict[str, str]], section: str, key: str, default: str) -> str:
    """Получить значение из конфига с fallback на default."""
    env_name = f"ADAPTIVE_AI_{section.upper()}_{key.upper()}"
    env_val = os.environ.get(env_name)
    if env_val is not None:
        return env_val
    return config_data.get(section, {}).get(key, default)


@dataclass
class Config:
    """Глобальная конфигурация. Все пути — абсолютные, без хардкода."""

    # Внутренние данные конфига
    _config_data: dict = field(default_factory=dict, repr=False)
    _config_path: str = ""

    # Директории пользовательских данных
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    db_path: Path = field(default_factory=lambda: Path("data/adaptive_ai.db"))

    # Модель
    model_backend: str = "rwkv_cpp"
    model_path: str | None = None

    # Оркестратор
    max_steps: int = 8
    max_retries: int = 2
    max_context_tokens: int = 4096
    max_total_seconds: int = 300

    # Генерация
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_tokens: int = 512
    system_prompt: str = ""

    # Инструменты
    allowed_read_dirs: str = "data,models"
    allowed_test_dirs: str = "data"
    max_file_size_mb: int = 10
    python_calc_timeout: int = 10
    test_timeout: int = 60

    # Оценка
    eval_results_dir: str = "data/eval_results"
    regression_enabled: bool = True

    # Обучение
    allow_training: bool = False
    training_dir: Path = field(default_factory=lambda: Path("training"))
    auto_prepare_datasets: bool = True
    min_experience_quality: str = "confirmed"

    # Ресурсы
    max_ram_mb: int = 8192
    max_ram_ratio: float = 0.8
    training_ram_mb: int = 4096

    # Метакогниция
    error_monitor_enabled: bool = True
    guard_engine_enabled: bool = True
    confidence_calibration: bool = True
    max_error_cases: int = 1000
    max_guards: int = 100

    # Память
    max_search_results: int = 10
    fts_enabled: bool = True
    memory_max_context_tokens: int = 2048
    relevance_threshold: float = 0.3

    # Пути
    skills_dir: str = "skills"

    # Сеть
    allow_network: bool = False

    def __post_init__(self) -> None:
        # Загружаем конфиг-файл
        config_path = os.environ.get("ADAPTIVE_AI_CONFIG", "config.txt")
        # Ищем config.txt в текущей директории и в родительской
        for candidate in [config_path, "config.txt", "C:/Orion/config.txt"]:
            if os.path.exists(candidate):
                self._config_path = candidate
                break
        self._config_data = _load_config_file(self._config_path) if self._config_path else {}

        cd = self._config_data

        # Применяем значения из конфига (если не переопределены env)
        def get(section, key, default):
            env_name = f"ADAPTIVE_AI_{section.upper()}_{key.upper()}"
            env_val = os.environ.get(env_name)
            if env_val is not None:
                return env_val
            return cd.get(section, {}).get(key, default)

        # Пути
        self.data_dir = Path(get("paths", "data_dir", str(self.data_dir)))
        self.models_dir = Path(get("paths", "models_dir", str(self.models_dir)))
        self.skills_dir = get("paths", "skills_dir", self.skills_dir)

        # Модель
        self.model_backend = get("model", "backend", self.model_backend)
        mp = get("model", "path", "auto")
        if mp and mp != "auto" and mp != "None":
            self.model_path = mp
        elif self.model_path is None:
            # Автовыбор с кэшированием
            try:
                from adaptive_ai.models.model_selector import select_model_with_cache

                best = select_model_with_cache(self.models_dir, ram_budget_ratio=self.max_ram_ratio)
                if best is not None:
                    self.model_path = best.path
            except Exception:  # noqa: BLE001
                pass

        lib_path = get("model", "library_path", "")
        if lib_path:
            os.environ.setdefault("RWKV_CPP_LIB", lib_path)
        vocab_path = get("model", "vocab_path", "")
        if vocab_path:
            os.environ.setdefault("RWKV_VOCAB_PATH", vocab_path)

        # БД
        db = get("memory", "db_path", str(self.db_path))
        self.db_path = Path(db)

        # Оркестратор
        self.max_steps = int(get("orchestrator", "max_steps", str(self.max_steps)))
        self.max_retries = int(get("orchestrator", "max_retries", str(self.max_retries)))
        self.max_context_tokens = int(get("orchestrator", "max_context_tokens", str(self.max_context_tokens)))
        self.max_total_seconds = int(get("orchestrator", "max_total_seconds", str(self.max_total_seconds)))

        # Генерация
        self.temperature = float(get("generation", "temperature", str(self.temperature)))
        self.top_p = float(get("generation", "top_p", str(self.top_p)))
        self.top_k = int(get("generation", "top_k", str(self.top_k)))
        self.repetition_penalty = float(get("generation", "repetition_penalty", str(self.repetition_penalty)))
        self.max_tokens = int(get("generation", "max_tokens", str(self.max_tokens)))
        self.system_prompt = get("generation", "system_prompt", self.system_prompt)

        # Инструменты
        self.allowed_read_dirs = get("tools", "allowed_read_dirs", self.allowed_read_dirs)
        self.allowed_test_dirs = get("tools", "allowed_test_dirs", self.allowed_test_dirs)
        self.max_file_size_mb = int(get("tools", "max_file_size_mb", str(self.max_file_size_mb)))
        self.python_calc_timeout = int(get("tools", "python_calc_timeout", str(self.python_calc_timeout)))
        self.test_timeout = int(get("tools", "test_timeout", str(self.test_timeout)))

        # Оценка
        self.eval_results_dir = get("evaluation", "results_dir", self.eval_results_dir)
        self.regression_enabled = get("evaluation", "regression_enabled", "true").lower() == "true"

        # Обучение
        self.allow_training = get("learning", "allow_training", "false").lower() == "true"
        self.training_dir = Path(get("learning", "training_dir", str(self.training_dir)))
        self.auto_prepare_datasets = get("learning", "auto_prepare_datasets", "true").lower() == "true"
        self.min_experience_quality = get("learning", "min_experience_quality", self.min_experience_quality)

        # Ресурсы
        self.max_ram_mb = int(get("resources", "max_ram_mb", str(self.max_ram_mb)))
        self.max_ram_ratio = float(get("resources", "max_ram_ratio", str(self.max_ram_ratio)))
        self.training_ram_mb = int(get("resources", "training_ram_mb", str(self.training_ram_mb)))

        # Метакогниция
        self.error_monitor_enabled = get("metacognition", "error_monitor_enabled", "true").lower() == "true"
        self.guard_engine_enabled = get("metacognition", "guard_engine_enabled", "true").lower() == "true"
        self.confidence_calibration = get("metacognition", "confidence_calibration", "true").lower() == "true"
        self.max_error_cases = int(get("metacognition", "max_error_cases", str(self.max_error_cases)))
        self.max_guards = int(get("metacognition", "max_guards", str(self.max_guards)))

        # Память
        self.max_search_results = int(get("memory", "max_search_results", str(self.max_search_results)))
        self.fts_enabled = get("memory", "fts_enabled", "true").lower() == "true"
        self.memory_max_context_tokens = int(get("memory", "max_context_tokens", str(self.memory_max_context_tokens)))
        self.relevance_threshold = float(get("memory", "relevance_threshold", str(self.relevance_threshold)))

        # Сеть
        self.allow_network = get("model", "allow_network", "false").lower() == "true"

    def ensure_dirs(self) -> None:
        """Создаёт необходимые директории, если их нет."""
        for d in (self.data_dir, self.models_dir, self.training_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls) -> "Config":
        return cls()
