# TROUBLESHOOTING.md

## Известные проблемы и решения

### 1. "Illegal instruction" при запуске rwkv.cpp
**Причина:** CPU не поддерживает AVX2 (i7-3612QM — Ivy Bridge).
**Решение:** Собрать rwkv.cpp с флагами `-DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF`.
В Dockerfile уже задан `RWKV_CPP_BUILD_FLAGS`.

### 2. Нет компилятора C/C++ для сборки rwkv.cpp
**Решение:** Установить LLVM-MinGW через winget:
```
winget install MartinStorsjo.LLVM-MinGW.UCRT
```
Затем собрать:
```
cd rwkv.cpp
cmake -S . -B build -G "MinGW Makefiles" -DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release -j2
```

### 3. Python / git / docker / cmake не найдены
**Решение:** Установите Python 3.12, Git, Docker Desktop, CMake перед началом разработки.

### 4. Модель не найдена
**Причина:** Большие модели не входят в репозиторий/образ.
**Решение:** Положите модель в `./models` и укажите `ADAPTIVE_AI_MODEL_PATH`.

### 5. rwkv.cpp библиотека не найдена
**Решение:** Соберите библиотеку (см. пункт 2 или Dockerfile) или укажите `RWKV_CPP_LIB`.

### 6. generate() возвращает not_supported
**Причина:** Модель не загружена (путь не указан или файл не существует).
**Статус:** Этап 1 — бэкенд реализован, требуется файл модели RWKV в формате rwkv.cpp (.bin).

### 7. Токенизация некорректна
**Причина:** Используется простой байтовый токенизатор (каждый байт UTF-8 = токен).
**Решение:** Для качественной генерации подключить полноценный BPE-токенизатор RWKV.
