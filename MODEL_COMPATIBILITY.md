# MODEL_COMPATIBILITY.md

## Проверенные модели и рантаймы

| Модель | Размер | Формат | RAM | CPU | Статус |
|--------|--------|--------|-----|-----|--------|
| RWKV-4-Pile-169M | 169M | rwkv.cpp Q4_0 | ~1 GB | i7-3612QM (нет AVX2) | Не проверено (нет модели) |
| RWKV-4-Pile-1.5B | 1.5B | rwkv.cpp Q4_0 | ~2 GB | i7-3612QM (нет AVX2) | Не проверено |
| RWKV-7-World-0.1B | ~100M | rwkv.cpp Q4_0 | ~0.5 GB | i7-3612QM (нет AVX2) | Не проверено |
| RWKV-7-World-0.4B | ~400M | rwkv.cpp Q4_0 | ~1 GB | i7-3612QM (нет AVX2) | Не проверено |
| RWKV-7-World-1.5B | ~1.5B | rwkv.cpp Q4_0 | ~2 GB | i7-3612QM (нет AVX2) | Не проверено |

## Требования к CPU

- AVX2 не требуется (сборка с `RWKV_AVX2=OFF`).
- Работает на старых CPU (SSE4.2 minimum).
- Компилятор для сборки: LLVM-MinGW (clang) или MSVC.

## Ограничения

- GT 620M не подходит для ускорения LLM.
- Инференс преимущественно CPU.
- Большие модели не загружаются автоматически.
- Токенизатор: текущая реализация использует простой байтовый токенизатор (каждый байт UTF-8 = токен). Для RWKV-7 World требуется полноценный RWKV World Tokenizer (vocab 65536, файл `rwkv_vocab_v20230424.txt`). Без него генерация будет работать, но качество текста будет низким.

## Сборка rwkv.cpp (без AVX2)

```cmd
cd rwkv.cpp
cmake -S . -B build -G "MinGW Makefiles" -DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release -j2
# Библиотека: build/bin/librwkv.dll
```

## Загрузка и конвертация модели RWKV-7 World

По ТЗ автоматические сетевые загрузки запрещены. Модель скачивается и конвертируется вручную.

### 1. Скачать модель с Hugging Face

```
https://huggingface.co/BlinkDL/rwkv-7-world
```

Рекомендуемые варианты для целевого железа (i7-3612QM, 16 GB RAM, HDD):
- `RWKV-7-World-0.1B-v2.8-20241210-ctx4096.pth` — smallest, ~200 MB
- `RWKV-7-World-0.4B-v2.8-20241210-ctx4096.pth` — ~400 MB

Положить `.pth` файл в `C:\Orion\models\`.

### 2. Установить зависимости для конвертации

```cmd
pip install torch numpy
```

### 3. Конвертировать .pth в rwkv.cpp .bin

```cmd
cd C:\Orion\rwkv.cpp
python python\convert_pytorch_to_ggml.py C:\Orion\models\RWKV-7-World-0.1B-v2.8-20241210-ctx4096.pth C:\Orion\models\rwkv-7-world-0.1B.bin FP16
```

### 4. Квантовать (опционально, но рекомендуется для слабого железа)

```cmd
python python\quantize.py C:\Orion\models\rwkv-7-world-0.1B.bin C:\Orion\models\rwkv-7-world-0.1B-Q4_0.bin Q4_0
```

### 5. Указать путь к модели

```cmd
set ADAPTIVE_AI_MODEL_PATH=C:\Orion\models\rwkv-7-world-0.1B-Q4_0.bin
set RWKV_CPP_LIB=C:\Orion\rwkv.cpp\build\bin\librwkv.dll
```

### 6. Проверить

```cmd
python -m adaptive_ai.app.main status
python -m adaptive_ai.app.main chat "Привет"
```

## Совместимость форматов rwkv.cpp

- Поддерживаемые форматы квантования: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- FP16 / FP32 — без квантования
- Версия файла: RWKV_FILE_VERSION_1 (101)
