"""Launcher — интерактивный запуск Adaptive Intelligence.

Спрашивает пользователя, как запустить систему:
  1. CLI (командная строка)
  2. Open WebUI (веб-интерфейс)
  3. Cherry Studio (десктоп-приложение)
  4. API Server (только сервер)

Запуск: python launcher.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import time

# Устанавливаем UTF-8 для консоли
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Путь к проекту
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# Добавляем проект в PATH
sys.path.insert(0, PROJECT_DIR)


def print_banner():
    print("=" * 60)
    print("  Adaptive Intelligence — Launcher")
    print("  Локальная саморазвивающаяся ИИ-система")
    print("=" * 60)
    print()


def print_menu():
    print("Выберите режим запуска:")
    print()
    print("  1. CLI — командная строка (диалог, задачи, память)")
    print("  2. Open WebUI — веб-интерфейс (браузер)")
    print("  3. Cherry Studio — десктоп-приложение")
    print("  4. API Server — только сервер (для подключения клиентов)")
    print("  5. Выход")
    print()


def launch_cli():
    """Запуск CLI."""
    print("Запуск CLI...")
    print("Команды: chat, solve, status, mem-search, mem-add, tools, skills, models, eval, experience, train")
    print("Для выхода из чата введите 'exit'")
    print()
    subprocess.run([PYTHON, "-m", "adaptive_ai.app.main", "chat"], cwd=PROJECT_DIR)


def launch_api_server(port: int = 8080):
    """Запуск API-сервера."""
    print(f"Запуск API-сервера на порту {port}...")
    print(f"OpenAI-совместимый endpoint: http://localhost:{port}/v1")
    print("Нажмите Ctrl+C для остановки.")
    print()
    subprocess.run([PYTHON, "-m", "adaptive_ai.app.api_server", "--port", str(port)], cwd=PROJECT_DIR)


def launch_open_webui():
    """Запуск Open WebUI."""
    print("=== Open WebUI ===")
    print()
    print("Open WebUI — веб-интерфейс для LLM.")
    print()

    # Проверяем Docker
    docker_available = False
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            docker_available = True
            print(f"  Docker: {result.stdout.strip()}")
    except FileNotFoundError:
        pass

    if not docker_available:
        print("  Docker не найден.")
        print()
        print("  Варианты:")
        print("  1. Установить Docker Desktop: https://www.docker.com/products/docker-desktop/")
        print("  2. Использовать Cherry Studio (вариант 3) — не требует Docker")
        print("  3. Использовать pip-версию Open WebUI (экспериментально)")
        print()
        choice = input("  Выберите вариант (1/2/3): ").strip()
        if choice == "1":
            print("  Установите Docker Desktop и перезапустите лаунчер.")
            return
        elif choice == "2":
            launch_cherry_studio()
            return
        elif choice == "3":
            print("  Установка Open WebUI через pip...")
            subprocess.run([PYTHON, "-m", "pip", "install", "open-webui"], cwd=PROJECT_DIR)
            print("  Запуск Open WebUI...")
            api_process = subprocess.Popen(
                [PYTHON, "-m", "adaptive_ai.app.api_server", "--port", "8080"],
                cwd=PROJECT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)
            try:
                subprocess.run([PYTHON, "-m", "open_webui", "serve", "--port", "3000"], cwd=PROJECT_DIR)
            except KeyboardInterrupt:
                pass
            finally:
                api_process.terminate()
            return
        else:
            return

    # Запускаем API-сервер в фоне
    print("\nЗапуск API-сервера...")
    api_process = subprocess.Popen(
        [PYTHON, "-m", "adaptive_ai.app.api_server", "--port", "8080"],
        cwd=PROJECT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    # Запускаем Open WebUI
    print("Запуск Open WebUI через Docker...")
    print("  (первый запуск может занять несколько минут)")
    print()
    try:
        subprocess.run([
            "docker", "run", "-d",
            "-p", "3000:8080",
            "--add-host=host.docker.internal:host-gateway",
            "-e", "OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1",
            "-e", "OPENAI_API_KEY=adaptive-ai",
            "--name", "open-webui",
            "ghcr.io/open-webui/open-webui:main",
        ], check=False)
        print()
        print("Open WebUI запущен!")
        print("  Откройте в браузере: http://localhost:3000")
        print("  API Adaptive Intelligence: http://localhost:8080/v1")
        print()
        print("Для остановки:")
        print("  docker stop open-webui && docker rm open-webui")
        print("  (API-сервер остановится при закрытии этого окна)")
        print()
        input("Нажмите Enter для остановки API-сервера...")
    except Exception as e:
        print(f"Ошибка запуска Open WebUI: {e}")
    finally:
        api_process.terminate()
        api_process.wait()


def launch_cherry_studio():
    """Запуск Cherry Studio."""
    print("=== Cherry Studio ===")
    print()
    print("Cherry Studio — десктоп-приложение для работы с LLM.")
    print()
    print("Шаги:")
    print("  1. Скачайте Cherry Studio: https://cherry-ai.com/")
    print("  2. Установите и запустите приложение")
    print("  3. В настройках добавьте провайдера:")
    print("     - Тип: OpenAI")
    print("     - Base URL: http://localhost:8080/v1")
    print("     - API Key: adaptive-ai (любой)")
    print("     - Модель: rwkv_cpp (или имя вашей модели)")
    print()
    print("Запуск API-сервера Adaptive Intelligence...")
    print("Нажмите Ctrl+C для остановки.")
    print()
    subprocess.run([PYTHON, "-m", "adaptive_ai.app.api_server", "--port", "8080"], cwd=PROJECT_DIR)


def main():
    print_banner()

    while True:
        print_menu()
        try:
            choice = input("Введите номер (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if choice == "1":
            launch_cli()
        elif choice == "2":
            launch_open_webui()
        elif choice == "3":
            launch_cherry_studio()
        elif choice == "4":
            port = input("Порт (по умолчанию 8080): ").strip()
            port = int(port) if port else 8080
            launch_api_server(port)
        elif choice == "5":
            print("Выход.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
