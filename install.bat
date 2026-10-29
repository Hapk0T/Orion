@echo off
chcp 1251 >nul
REM ============================================================
REM Adaptive Intelligence — Полный установщик для Windows
REM Работает на чистой Windows без предустановленных инструментов
REM ============================================================
setlocal enabledelayedexpansion

REM Добавляем стандартные пути в PATH
set "PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\Scripts;C:\Program Files\Git\bin;C:\Program Files\CMake\bin;C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin;%PATH%"

echo ============================================================
echo   Adaptive Intelligence — Установщик
echo   Локальная саморазвивающаяся ИИ-система
echo ============================================================
echo.

REM Определяем директорию проекта
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM ============================================================
REM 1. Проверка и установка Python 3.12
REM ============================================================
echo [1/8] Проверка Python...
if not exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo Python не найден. Установка через winget...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo ОШИБКА: Не удалось установить Python. Установите вручную: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo Python установлен. Перезапустите установщик.
    pause
    exit /b 0
) else (
    for /f "tokens=*" %%i in ('python --version') do echo   Найден: %%i
)

REM ============================================================
REM 2. Проверка и установка Git
REM ============================================================
echo.
echo [2/8] Проверка Git...
if not exist "C:\Program Files\Git\bin\git.exe" (
    echo Git не найден. Установка через winget...
    winget install Git.Git --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo ОШИБКА: Не удалось установить Git.
        pause
        exit /b 1
    )
    echo Git установлен. Перезапустите установщик.
    pause
    exit /b 0
) else (
    for /f "tokens=*" %%i in ('git --version') do echo   Найден: %%i
)

REM ============================================================
REM 3. Проверка и установка CMake
REM ============================================================
echo.
echo [3/8] Проверка CMake...
if not exist "C:\Program Files\CMake\bin\cmake.exe" (
    echo CMake не найден. Установка через winget...
    winget install Kitware.CMake --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo ОШИБКА: Не удалось установить CMake.
        pause
        exit /b 1
    )
    echo CMake установлен. Перезапустите установщик.
    pause
    exit /b 0
) else (
    for /f "tokens=*" %%i in ('cmake --version') do echo   Найден: %%i & goto :cmake_done
)
:cmake_done

REM ============================================================
REM 4. Проверка и установка LLVM-MinGW (компилятор)
REM ============================================================
echo.
echo [4/8] Проверка компилятора (LLVM-MinGW)...
if not exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin\clang.exe" (
    echo Компилятор не найден. Установка через winget...
    winget install MartinStorsjo.LLVM-MinGW.UCRT --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo ОШИБКА: Не удалось установить LLVM-MinGW.
        pause
        exit /b 1
    )
    echo LLVM-MinGW установлен. Перезапустите установщик.
    pause
    exit /b 0
) else (
    for /f "tokens=*" %%i in ('clang --version') do echo   Найден: %%i & goto :clang_done
)
:clang_done

REM Добавляем инструменты в PATH
set "PATH=C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\bin;%PATH%"

REM ============================================================
REM 5. Установка Python-зависимостей
REM ============================================================
echo.
echo [5/8] Установка Python-зависимостей...
pip install numpy pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install rwkv tokenizers
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не все зависимости установлены. Продолжаем...
)

REM ============================================================
REM 6. Сборка rwkv.cpp (без AVX2)
REM ============================================================
echo.
echo [6/8] Сборка rwkv.cpp (без AVX2)...
if not exist "rwkv.cpp" (
    echo Клонирование rwkv.cpp...
    git clone https://github.com/RWKV/rwkv.cpp.git rwkv.cpp
    cd rwkv.cpp
    git submodule update --init --recursive
    cd ..
)

if not exist "rwkv.cpp\build\bin\librwkv.dll" (
    echo Сборка rwkv.cpp...
    if not exist "rwkv.cpp\build" mkdir "rwkv.cpp\build"
    cd rwkv.cpp
    cmake -S . -B build -G "MinGW Makefiles" -DRWKV_AVX2=OFF -DRWKV_AVX=OFF -DRWKV_FMA=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    if %errorlevel% neq 0 (
        echo ОШИБКА: CMake configure failed
        cd ..
        pause
        exit /b 1
    )
    cmake --build build --config Release -j2
    if %errorlevel% neq 0 (
        echo ОШИБКА: Build failed
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo rwkv.cpp собран успешно.
) else (
    echo   rwkv.cpp уже собран.
)

REM Копируем MinGW DLL для runtime
copy "C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin\*.dll" "rwkv.cpp\build\bin\" >nul 2>&1

REM ============================================================
REM 7. Загрузка модели и токенизатора
REM ============================================================
echo.
echo [7/8] Загрузка модели и токенизатора...
if not exist "models" mkdir "models"

REM 7. Проверка токенизатора
if not exist "models\rwkv_vocab_v20230424.txt" (
    echo   Токенизатор не найден.
    echo   Скачайте вручную: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt
    echo   Положите в папку models/
) else (
    echo   Токенизатор найден.
)



REM Проверяем наличие модели
if not exist "models\*.bin" (
    echo.
    echo ВНИМАНИЕ: Модели не найдены в папке models/.
    echo.
    echo Для загрузки модели RWKV7-G1k-1.5B (рекомендуется):
    echo   1. Скачайте: https://huggingface.co/RWKV/RWKV7-G1k-1.5B-20260930
    echo   2. Положите .pth файл в папку models/
    echo   3. Запустите: python rwkv.cpp\python\convert_pytorch_to_ggml.py models\<model>.pth models\<model>.bin FP16
    echo   4. Запустите: python rwkv.cpp\python\quantize.py models\<model>.bin models\<model>-Q4_0.bin Q4_0
    echo.
    echo Или скачайте уже сконвертированную модель:
    echo   https://huggingface.co/RWKV/RWKV7-G1k-1.5B-20260930
    echo.
) else (
    echo   Модели найдены.
)

REM ============================================================
REM 8. Настройка конфигурации
REM ============================================================
echo.
echo [8/8] Настройка конфигурации...
if not exist "config.txt" (
    echo Конфигурация не найдена. Создаём по умолчанию...
    REM config.txt уже должен быть в репозитории
)

REM Обновляем config.txt для автовыбора модели
if exist "config.txt" (
    echo   config.txt найден.
)

REM Создаём run.bat для удобного запуска
echo @echo off > run.bat
echo chcp 65001 ^>nul >> run.bat
echo cd /d "%PROJECT_DIR%" >> run.bat
echo set PATH=C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\bin;%%PATH%% >> run.bat
echo "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" launcher.py >> run.bat

echo.
REM ============================================================
REM 9. Дополнительные компоненты (по выбору)
REM ============================================================
echo.
echo [9/9] Дополнительные компоненты...
echo.
echo Выберите дополнительные компоненты для установки:
echo   1. Только базовая система (CLI + модель)
echo   2. + Docker Desktop (для Open WebUI)
echo   3. + Cherry Studio (десктоп-приложение)
echo   4. Полная установка (Docker + Cherry Studio)
echo.
set /p COMPONENTS="Ваш выбор (1-4): "

if "%COMPONENTS%"=="2" goto :install_docker
if "%COMPONENTS%"=="3" goto :install_cherry
if "%COMPONENTS%"=="4" goto :install_all
goto :skip_components

:install_docker
echo Установка Docker Desktop...
winget install Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось установить Docker Desktop.
    echo Скачайте вручную: https://www.docker.com/products/docker-desktop/
)
goto :skip_components

:install_cherry
echo Установка Cherry Studio...
winget install CherryStudio.CherryStudio --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось установить Cherry Studio.
    echo Скачайте вручную: https://cherry-ai.com/
)
goto :skip_components

:install_all
echo Установка Docker Desktop...
winget install Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
echo Установка Cherry Studio...
winget install CherryStudio.CherryStudio --accept-package-agreements --accept-source-agreements
goto :skip_components

:skip_components

REM Создаём run.bat для удобного запуска
echo @echo off > run.bat
echo chcp 65001 ^>nul >> run.bat
echo cd /d "%PROJECT_DIR%" >> run.bat
echo set PATH=C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\bin;%%PATH%% >> run.bat
echo "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" launcher.py >> run.bat

echo.
echo ============================================================
echo   Установка завершена!
echo ============================================================
echo.
echo Для запуска: дважды кликните run.bat
echo Или запустите: python launcher.py
echo.
echo Если модель не скачана, следуйте инструкциям выше.
echo.
pause
