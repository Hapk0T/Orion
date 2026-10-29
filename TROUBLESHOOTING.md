# TROUBLESHOOTING.md

## Известные проблемы и решения

### 1. "Illegal instruction" при запуске rwkv.cpp
**Причина:** CPU не поддерживает AVX2 (i7-3612QM — Ivy Bridge).
**Решение:** Собрать rwkv.cpp с флагами `-DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF`.
В Dockerfile уже задан `RWKV_CPP_BUILD_FLAGS`.

### 2. Python / git / docker / cmake не найдены
**Решение:** Установите Python 3.12, Git, Docker Desktop, CMake перед началом разработки.

### 3. Модель не найдена
**Причина:** Большие модели не входят в репозиторий/образ.
**Решение:** Положите модель в `./models` и укажите `ADAPTIVE_AI_MODEL_PATH`.

### 4. rwkv.cpp библиотека не найдена
**Решение:** Соберите библиотеку (см. Dockerfile) или укажите `RWKV_CPP_LIB`.

### 5. generate() возвращает not_supported
**Причина:** Генерация через rwkv.cpp не реализована в v0.1 (требуется токенизатор и C API).
**Статус:** ожидает этапа 1 (совместимость модели).
