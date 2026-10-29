# MODEL_COMPATIBILITY.md

## Проверенные модели и рантаймы

| Модель | Размер | Формат | RAM | CPU | Статус |
|--------|--------|--------|-----|-----|--------|
| RWKV-4-Pile-169M | 169M | rwkv.cpp Q4_0 | ~1 GB | i7-3612QM (нет AVX2) | Не проверено (нет модели) |
| RWKV-4-Pile-1.5B | 1.5B | rwkv.cpp Q4_0 | ~2 GB | i7-3612QM (нет AVX2) | Не проверено |
| RWKV-7-World-0.1B | ~100M | rwkv.cpp Q4_0 | ~0.5 GB | i7-3612QM (нет AVX2) | Не проверено |
| RWKV-7-World-0.4B | ~400M | rwkv.cpp Q4_0 | ~1 GB | i7-3612QM (нет AVX2) | ✅ Проверено — генерирует текст на русском языке |
| RWKV-7-World-1.5B | ~1.5B | rwkv.cpp Q4_0 | ~2 GB | i7-3612QM (нет AVX2) | Не проверено |
| RWKV7-G1k-7.2B | 7.2B | rwkv.cpp Q4_0 | ~3.6 GB | i7-3612QM (нет AVX2) | ⚠️ Тесно для 16 GB RAM, но возможно |
| RWKV7-G1k-13.3B | 13.3B | rwkv.cpp Q4_0 | ~6.7 GB | i7-3612QM (нет AVX2) | ❌ Не реалистично для целевого железа |

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

## Загрузка и конвертация модели RWKV-7 World / RWKV7-G1k

По ТЗ автоматические сетевые загрузки запрещены. Модель скачивается и конвертируется вручную.

### 1. Скачать модель с Hugging Face

Выберите вариант в зависимости от железа:

| Модель | Размер .pth | RAM (Q4_0) | Для какого железа |
|--------|-------------|------------|-------------------|
| `RWKV7-G1k-0.1B-20260930` | ~200 MB | ~0.5 GB | i7-3612QM, 16 GB RAM ✅ |
| `RWKV7-G1k-0.4B-20260930` | ~400 MB | ~1 GB | i7-3612QM, 16 GB RAM ✅ |
| `RWKV7-G1k-1.5B-20260930` | ~1.5 GB | ~2 GB | i7-3612QM, 16 GB RAM ⚠️ |
| `RWKV7-G1k-7.2B-20260930` | ~14.4 GB | ~3.6 GB | i7-3612QM, 16 GB RAM ⚠️ Тесно |
| `RWKV7-G1k-13.3B-20260930` | ~26 GB | ~6.7 GB | ❌ i7-3612QM — нереалистично |

Ссылки:
- https://huggingface.co/RWKV/RWKV7-G1k-7.2B-20260930
- https://huggingface.co/RWKV/RWKV7-G1k-13.3B-20260930
- https://huggingface.co/BlinkDL/rwkv-7-world

Положите `.pth` файл в `C:\Orion\models\`.

### Ограничение 7.2B на целевом железе

Модель `RWKV7-G1k-7.2B` (7,199,932,416 параметров, 32 слоев, hidden 4096, context 25600) технически поддерживается rwkv.cpp, но на i7-3612QM с 16 GB DDR3 и HDD:

- Q4_0 квантование (~3.6 GB) оставляет ~12 GB на ОС, Python и буферы
- HDD делает загрузку модели (14.4 GB .pth) и генерацию очень медленными
- Конвертация .pth -> .bin требует ~14 GB RAM во время обработки
- Рекомендуется использовать на более мощном оборудовании (32+ GB RAM, SSD)

Для проверки работоспособности системы начните с 0.1B или 0.4B.

### 2. Установить зависимости для конвертации

```cmd
pip install torch numpy
```

### 3. Конвертировать .pth в rwkv.cpp .bin

```cmd
cd C:\Orion\rwkv.cpp
python python\convert_pytorch_to_ggml.py C:\Orion\models\<имя_модели>.pth C:\Orion\models\<имя_модели>.bin FP16
```

### 4. Квантовать (обязательно для слабого железа)

```cmd
python python\quantize.py C:\Orion\models\<имя_модели>.bin C:\Orion\models\<имя_модели>-Q4_0.bin Q4_0
```

### 5. Указать путь к модели

```cmd
set ADAPTIVE_AI_MODEL_PATH=C:\Orion\models\<имя_модели>-Q4_0.bin
set RWKV_CPP_LIB=C:\Orion\rwkv.cpp\build\bin\librwkv.dll
```

### 6. Проверить

```cmd
python -m adaptive_ai.app.main status
python -m adaptive_ai.app.main chat "Привет"
```

### Ограничение 13.3B на целевом железе

Модель `RWKV7-G1k-13.3B` (13,270,298,624 параметров, 61 слоев, hidden 4096) технически поддерживается rwkv.cpp, но на i7-3612QM с 16 GB DDR3 и HDD:

- Даже Q4_0 квантование (~6.7 GB) оставляет менее 9 GB на ОС, Python и буферы
- HDD делает загрузку модели и генерацию крайне медленными
- Рекомендуется использовать на более мощном оборудовании (32+ GB RAM, SSD)

Для проверки работоспособности системы начните с 0.1B или 0.4B.

## Совместимость форматов rwkv.cpp

- Поддерживаемые форматы квантования: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- FP16 / FP32 — без квантования
- Версия файла: RWKV_FILE_VERSION_1 (101)
