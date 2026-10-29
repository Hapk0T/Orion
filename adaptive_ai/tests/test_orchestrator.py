"""Юнит-тесты оркестратора."""

from __future__ import annotations

from adaptive_ai.core.contracts import ModelBackend, ModelCapabilities, Status, TaskResult
from adaptive_ai.core.orchestrator import Orchestrator
from adaptive_ai.core.policies import SafetyPolicy, ResourcePolicy


class FakeBackend(ModelBackend):
    name = "fake"

    def __init__(self, answer: str = "ответ") -> None:
        self._answer = answer
        self._loaded = True

    def load(self, model_path: str | None = None) -> TaskResult:
        self._loaded = True
        return TaskResult(status=Status.OK)

    def generate(self, prompt, system_prompt=None, context=None, tools=None, max_tokens=512, temperature=0.7) -> TaskResult:
        return TaskResult(status=Status.OK, data='{"tool": null, "finish": true, "answer": "' + self._answer + '"}')

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(name=self.name)

    def is_loaded(self) -> bool:
        return self._loaded


class FakeMemory:
    def __init__(self) -> None:
        self.saved = []

    def search(self, query, limit=10):
        return []

    def save_episode(self, content, source=None, **kwargs):
        self.saved.append((content, source))
        from adaptive_ai.core.contracts import TaskResult

        return TaskResult(status=Status.OK, data={"id": "x"})

    def get(self, item_id):
        return None

    def update(self, item_id, **kwargs):
        return TaskResult(status=Status.OK)

    def delete(self, item_id):
        return TaskResult(status=Status.OK)

    def close(self):
        pass


class FakeRegistry:
    def __init__(self) -> None:
        self.calls = []

    def register(self, tool) -> None:
        pass

    def get(self, name):
        return None

    def list(self):
        return []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        from adaptive_ai.core.contracts import TaskResult

        return TaskResult(status=Status.OK, data={"ok": True})


def test_solve_simple():
    orch = Orchestrator(
        model=FakeBackend(answer="готово"),
        memory=FakeMemory(),
        tools=FakeRegistry(),
        safety=SafetyPolicy(),
        resources=ResourcePolicy(max_steps=8, max_retries_per_action=2),
    )
    result = orch.solve("напиши привет")
    assert result.status == "ok", result.error
    assert result.data == "готово"


def test_budget_exhausted():
    orch = Orchestrator(
        model=FakeBackend(answer='{"tool": null, "finish": false}'),
        memory=FakeMemory(),
        tools=FakeRegistry(),
        safety=SafetyPolicy(),
        resources=ResourcePolicy(max_steps=1, max_retries_per_action=0),
    )
    result = orch.solve("должен исчерпать бюджет")
    assert result.status == "budget_exhausted", result.status
