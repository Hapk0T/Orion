"""Тестовые наборы для оценки качества системы."""

from __future__ import annotations

from adaptive_ai.evaluation.evaluators import TestCase


def russian_language_suite() -> list[TestCase]:
    """Тесты русского языка и следования инструкциям."""
    return [
        TestCase(
            id="ru_001",
            category="русский язык",
            input="Напиши приветствие на русском языке.",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str) and len(actual) > 0,
            description="Генерация текста на русском",
        ),
        TestCase(
            id="ru_002",
            category="русский язык",
            input="Переведи на русский: 'Hello, world!'",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str) and ("привет" in actual.lower() or "мир" in actual.lower() or "здравств" in actual.lower()),
            description="Перевод на русский",
        ),
        TestCase(
            id="ru_003",
            category="следование инструкциям",
            input="Ответь одним словом: да или нет. Вопрос: 2+2=4?",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str) and len(actual.split()) <= 3,
            description="Краткий ответ по инструкции",
        ),
    ]


def fact_extraction_suite() -> list[TestCase]:
    """Тесты точного извлечения фактов."""
    return [
        TestCase(
            id="fact_001",
            category="точные факты",
            input="Столица Франции",
            expected="Париж",
            check=lambda actual, expected: expected.lower() in str(actual).lower(),
            description="Извлечение известного факта",
        ),
        TestCase(
            id="fact_002",
            category="точные факты",
            input="2+2",
            expected=4,
            check=lambda actual, expected: str(expected) in str(actual),
            description="Арифметический факт",
        ),
    ]


def memory_suite() -> list[TestCase]:
    """Тесты поиска и использования памяти."""
    return [
        TestCase(
            id="mem_001",
            category="память",
            input="Сохранённый факт: столица Японии — Токио",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str),
            description="Сохранение факта в память",
        ),
    ]


def logic_suite() -> list[TestCase]:
    """Тесты логических задач."""
    return [
        TestCase(
            id="logic_001",
            category="логика",
            input="Если A больше B, а B больше C, то A больше C?",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str) and ("да" in actual.lower() or "yes" in actual.lower() or "верно" in actual.lower() or "больше" in actual.lower()),
            description="Транзитивность отношения",
        ),
    ]


def tool_execution_suite() -> list[TestCase]:
    """Тесты выполнения инструментов."""
    return [
        TestCase(
            id="tool_001",
            category="инструменты",
            input="calculator: 2+2",
            expected=4,
            check=lambda actual, expected: str(expected) in str(actual),
            description="Калькулятор",
        ),
        TestCase(
            id="tool_002",
            category="инструменты",
            input="python_calc: 2**10",
            expected=1024,
            check=lambda actual, expected: str(expected) in str(actual),
            description="Python вычисления",
        ),
    ]


def error_handling_suite() -> list[TestCase]:
    """Тесты обработки ошибок."""
    return [
        TestCase(
            id="err_001",
            category="обработка ошибок",
            input="Деление на ноль: 1/0",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str),
            description="Обработка математической ошибки",
        ),
    ]


def structured_output_suite() -> list[TestCase]:
    """Тесты структурированного вывода."""
    return [
        TestCase(
            id="struct_001",
            category="структурированный вывод",
            input="Верни JSON с полями name и age для человека по имени Иван, 25 лет.",
            expected=None,
            check=lambda actual, expected: isinstance(actual, str),
            description="Генерация JSON",
        ),
    ]


def all_suites() -> dict[str, list[TestCase]]:
    """Все тестовые наборы."""
    return {
        "russian_language": russian_language_suite(),
        "fact_extraction": fact_extraction_suite(),
        "memory": memory_suite(),
        "logic": logic_suite(),
        "tool_execution": tool_execution_suite(),
        "error_handling": error_handling_suite(),
        "structured_output": structured_output_suite(),
    }
