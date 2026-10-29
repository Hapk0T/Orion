"""Типизированный граф рассуждения и действий.

Реализует рассуждение как исполняемый граф узлов, а не только как сохраняемый текст.
Каждое существенное утверждение ссылается на источник, наблюдение или явно
помеченную гипотезу.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    GOAL = "goal"
    DECOMPOSE = "decompose"
    RETRIEVE = "retrieve"
    TRANSFORM = "transform"
    RULE_CHECK = "rule_check"
    TOOL_CALL = "tool_call"
    VERIFICATION = "verification"
    MODEL_JUDGMENT = "model_judgment"
    OBSERVATION = "observation"
    CLAIM = "claim"
    STOP = "stop"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Evidence:
    """Доказательство для узла или утверждения."""

    source_type: str  # source, observation, computation, hypothesis
    source_id: str | None = None
    content: str = ""
    verified: bool = False


@dataclass
class GraphNode:
    """Узел графа рассуждения."""

    id: str
    node_type: NodeType
    handler_version: str = "1.0"
    inputs: list[str] = field(default_factory=list)  # id родительских узлов
    outputs: list[str] = field(default_factory=list)  # id дочерних узлов
    evidence: list[Evidence] = field(default_factory=list)
    result: Any = None
    status: NodeStatus = NodeStatus.PENDING
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_claim(self) -> bool:
        return self.node_type == NodeType.CLAIM

    def has_evidence(self) -> bool:
        return len(self.evidence) > 0

    def is_verified(self) -> bool:
        return any(e.verified for e in self.evidence)


@dataclass
class DecisionGraph:
    """Граф рассуждения и действий."""

    id: str
    task_id: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    root_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def set_root(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.root_id = node_id

    def add_edge(self, parent_id: str, child_id: str) -> None:
        if parent_id in self.nodes and child_id in self.nodes:
            if child_id not in self.nodes[parent_id].outputs:
                self.nodes[parent_id].outputs.append(child_id)
            if parent_id not in self.nodes[child_id].inputs:
                self.nodes[child_id].inputs.append(parent_id)

    def get_pending_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.PENDING]

    def get_completed_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.COMPLETED]

    def get_failed_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

    def topological_order(self) -> list[GraphNode]:
        """Топологическая сортировка узлов."""
        visited: set[str] = set()
        result: list[GraphNode] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                for child_id in node.outputs:
                    visit(child_id)
                result.append(node)

        if self.root_id:
            visit(self.root_id)
        for nid in self.nodes:
            if nid not in visited:
                visit(nid)
        result.reverse()
        return result

    def validate(self) -> list[str]:
        """Проверить корректность графа. Возвращает список проблем."""
        issues: list[str] = []
        for nid, node in self.nodes.items():
            for inp in node.inputs:
                if inp not in self.nodes:
                    issues.append(f"Узел {nid}: несуществующий вход {inp}")
            for out in node.outputs:
                if out not in self.nodes:
                    issues.append(f"Узел {nid}: несуществующий выход {out}")
        # Проверка: CLAIM узлы должны иметь evidence
        for nid, node in self.nodes.items():
            if node.is_claim() and not node.has_evidence():
                issues.append(f"Узел {nid}: утверждение без доказательств")
        # Проверка: STOP узлы должны быть листьями
        for nid, node in self.nodes.items():
            if node.node_type == NodeType.STOP and node.outputs:
                issues.append(f"Узел {nid}: STOP-узел имеет дочерние узлы")
        return issues

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "node_count": len(self.nodes),
            "completed": len(self.get_completed_nodes()),
            "failed": len(self.get_failed_nodes()),
            "pending": len(self.get_pending_nodes()),
            "issues": self.validate(),
        }
