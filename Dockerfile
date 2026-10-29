# Adaptive Intelligence — Docker-контейнер (CPU-инференс, offline-first)
# Собирается без AVX2/AVX/FMA — работает на старых CPU (i7-3612QM) и новых.
# Большие модели и пользовательские данные НЕ входят в образ.
# Монтируйте их через volumes при запуске.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ADAPTIVE_AI_DATA_DIR=/data \
    ADAPTIVE_AI_MODELS_DIR=/models \
    ADAPTIVE_AI_DB_PATH=/data/adaptive_ai.db \
    RWKV_CPP_BUILD_FLAGS="-DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF"

# Системные зависимости для сборки rwkv.cpp (CMake + компилятор)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-зависимости проекта
COPY pyproject.toml ./
RUN pip install --no-cache-dir -r pyproject.toml 2>/dev/null || true

# Исходный код проекта
COPY . /app

# Сборка rwkv.cpp (без AVX2 — совместимость со старыми CPU)
RUN if [ -d rwkv.cpp ]; then \
        cd rwkv.cpp && mkdir -p build && cd build \
        && cmake .. ${RWKV_CPP_BUILD_FLAGS} \
        && cmake --build . --config Release -j2 \
        && cmake --install . ; \
    fi

# Каталоги для пользовательских данных и моделей (몬тируются как volumes)
RUN mkdir -p /data /models \
    && chmod 777 /data /models

VOLUME ["/data", "/models"]

ENTRYPOINT ["python", "app/main.py"]
CMD ["--help"]
