"""Юнит-тесты встроенных инструментов."""

from adaptive_ai.core.contracts import Status
from adaptive_ai.tools.builtin.calculator import CalculatorTool
from adaptive_ai.tools.builtin.python_calc import PythonCalcTool
from adaptive_ai.tools.builtin.read_file import ReadFileTool
from adaptive_ai.tools.builtin.run_tests import RunTestsTool
from adaptive_ai.tools.builtin.search_memory import SearchMemoryTool
from adaptive_ai.tools.registry import ToolRegistry


def test_calculator_basic():
    tool = CalculatorTool()
    res = tool.execute({"expression": "2+2"})
    assert res.status == Status.OK
    assert res.data["result"] == 4


def test_calculator_invalid_chars():
    tool = CalculatorTool()
    res = tool.execute({"expression": "import os"})
    assert res.status == Status.ERROR


def test_python_calc_basic():
    tool = PythonCalcTool()
    res = tool.execute({"code": "2**10"})
    assert res.status == Status.OK
    assert res.data["result"] == 1024


def test_python_calc_banned():
    tool = PythonCalcTool()
    res = tool.execute({"code": "import os"})
    assert res.status == Status.ERROR


def test_read_file_allowed():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        tool = ReadFileTool(allowed_dirs=[os.path.dirname(path)])
        res = tool.execute({"path": path})
        assert res.status == Status.OK
        assert res.data["content"] == "hello world"
    finally:
        os.unlink(path)


def test_read_file_denied():
    tool = ReadFileTool(allowed_dirs=["/tmp/allowed"])
    res = tool.execute({"path": "C:/Windows/system32/drivers/etc/hosts"})
    assert res.status == Status.ERROR


def test_registry_call():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    res = registry.call("calculator", {"expression": "3*3"})
    assert res.status == Status.OK
    assert res.data["result"] == 9


def test_registry_not_found():
    registry = ToolRegistry()
    res = registry.call("nonexistent", {})
    assert res.status == Status.ERROR
    assert "не найден" in res.error


def test_search_memory_tool():
    class FakeStore:
        def search(self, query, limit=10):
            return [type("Item", (), {"id": "1", "content": "test fact", "source": "src"})()]

    tool = SearchMemoryTool(FakeStore())
    res = tool.execute({"query": "test"})
    assert res.status == Status.OK
    assert len(res.data["results"]) == 1
