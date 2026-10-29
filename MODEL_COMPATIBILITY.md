# MODEL_COMPATIBILITY.md

## Проверенные модели и рантаймы

| Модель | Размер | Формат | RAM | CPU | Статус |
|--------|--------|--------|-----|-----|--------|
| RWKV-4-Pile-169M | 169M | rwkv.cpp Q4_0 | ~1 GB | i7-3612QM (нет AVX2) | Не проверено (нет модели) |
| RWKV-4-Pile-1.5B | 1.5B | rwkv.cpp Q4_0 | ~2 GB | i7-3612QM (нет AVX2) | Не проверено |

## Требования к CPU

- AVX2 не требуется (сборка с `RWKV_AVX2=OFF`).
- Работает на старых CPU (SSE4.2 minimum).
- Компилятор для сборки: LLVM-MinGW (clang) или MSVC.

## Ограничения

- GT 620M не подходит для ускорения LLM.
- Инференс преимущественно CPU.
- Большие модели не загружаются автоматически.
- Токенизатор: простой байтовый (каждый байт UTF-8 = токен). Для полноценной работы нужен BPE-токенизатор RWKV.

## Сборка rwkv.cpp (без AVX2)

```cmd
cd rwkv.cpp
cmake -S . -B build -G "MinGW Makefiles" -DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release -j2
# Библиотека: build/bin/librwkv.dll
```
