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

### 6. generate() возвращает некорректный текст или ошибку токенизации
**Причина:** Используется простой байтовый токенизатор (каждый байт UTF-8 = токен).
RWKV-7 World использует RWKV World Tokenizer (vocab 65536). Без полноценного токенизатора генерация будет работать, но качество текста будет низким.
**Решение:** Для качественной генерации подключить RWKV World Tokenizer (`rwkv_vocab_v20230424.txt`) и токенизацию на основе BPE.

### 7. Модель RWKV-7 World не загружается (ошибка формата)
**Причина:** rwkv.cpp требует модель в собственном формате `.bin` (ggml), а не `.pth`.
**Решение:** Конвертировать через `python rwkv\convert_pytorch_to_ggml.py <model.pth> <output.bin> FP16`, затем опционально квантовать через `quantize.py`.

### 8. "Illegal instruction" при запуске rwkv.cpp
**Причина:** CPU не поддерживает AVX2 (i7-3612QM — Ivy Bridge).
**Решение:** Собрать rwkv.cpp с флагами `-DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF`.
В Dockerfile уже задан `RWKV_CPP_BUILD_FLAGS`.

### 10. Модель RWKV7-G1k-13.3B не загружается или вызывает OOM
**Причина:** 13.3B параметров требует ~26.6 GB в FP16 или ~6.7 GB в Q4_0. На i7-3612QM с 16 GB DDR3 и HDD это нереалистично.
**Решение:** Используйте kleinere модели (0.1B, 0.4B, 1.5B) для целевого железа. 13.3B требует 32+ GB RAM и SSD.

### 11. Конвертация RWKV7-G1k через convert_pytorch_to_ggml.py не работает
**Причина:** Скрипт может не поддерживать новейшие форматы RWKV7-G1k напрямую.
**Решение:** Используйте официальный конвертер из репозитория RWKV/RWKV-LM (`scripts/convert_rwkv7_to_hf.py`) или попробуйте предварительно конвертировать через rwkv.cpp скрипт. Проверьте issues в репозитории rwkv.cpp для последних обновлений поддержки v7.
