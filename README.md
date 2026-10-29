# Adaptive Intelligence

Локальная саморазвивающаяся интеллектуальная система. Модульная, офлайн-первая, работает на CPU.

## Статус

Проект в разработке (v0.1). Смотрите [ROADMAP.md](ROADMAP.md).

## Быстрый старт

### Через Docker (рекомендуется)

```bash
docker compose up --build
```

Модели и данные класть в `./models` и `./data` (они не входят в репозиторий).

### Локально

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m adaptive_ai.app.main --help
```

## Архитектура

Смотрите [ARCHITECTURE.md](ARCHITECTURE.md).

## Ограничения

Смотрите [TROUBLESHOOTING.md](TROUBLESHOOTING.md) и [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md).
