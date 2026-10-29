"""Юнит-тесты для ErrorMonitor, GuardEngine, DecisionGraph, ExperienceJournal."""

from adaptive_ai.core.error_monitor import ErrorMonitor, ErrorStatus, ErrorType
from adaptive_ai.core.guard_engine import GuardEngine, GuardStatus
from adaptive_ai.core.decision_graph import DecisionGraph, GraphNode, NodeType, NodeStatus
from adaptive_ai.memory.experience_journal import ExperienceJournal


def test_error_monitor_internal_contradiction():
    monitor = ErrorMonitor()
    cases = monitor.check_internal(
        claims=["Ответ: да", "Ответ: нет"],
        conditions=[],
        result=None,
    )
    assert len(cases) >= 1
    assert cases[0].error_type == ErrorType.CONTRADICTION


def test_error_monitor_external_tool_error():
    monitor = ErrorMonitor()
    cases = monitor.check_external(
        tool_results=[{"tool": "calc", "status": "error", "error": "division by zero"}],
        constraints=[],
    )
    assert len(cases) >= 1
    assert cases[0].status == ErrorStatus.CONFIRMED_ERROR


def test_error_monitor_empirical_mismatch():
    monitor = ErrorMonitor()
    cases = monitor.check_empirical(
        predicted="42",
        actual="43",
        task_id="t1",
    )
    assert len(cases) >= 1
    assert cases[0].error_type == ErrorType.PREDICTION_MISMATCH


def test_error_monitor_confidence_calibration():
    monitor = ErrorMonitor()
    monitor.record_confidence("t1", 0.9, True)
    monitor.record_confidence("t2", 0.3, False)
    result = monitor.calibrate()
    assert result["status"] == "ok"
    assert result["count"] == 2


def test_error_monitor_recurring_pattern():
    monitor = ErrorMonitor()
    for _ in range(3):
        cases = monitor.check_internal(
            claims=["Ответ: да", "Ответ: нет"],
            conditions=[],
            result=None,
        )
    recurring = [c for c in monitor.get_cases() if c.status == ErrorStatus.RECURRING_PATTERN]
    assert len(recurring) >= 1


def test_guard_engine_register_and_check():
    engine = GuardEngine()
    from adaptive_ai.core.error_monitor import ErrorCase, ErrorType, ErrorStatus, VerificationChannel
    err = ErrorCase(
        id="err-1",
        error_type=ErrorType.TOOL_ERROR,
        status=ErrorStatus.CONFIRMED_ERROR,
        description="Тестовая ошибка",
    )
    guard = engine.create_from_error(err, pattern="test", action_type="block", applicability=[])
    assert guard.id == "guard-0"
    assert guard.status == GuardStatus.EXPERIMENTAL


def test_guard_engine_find_applicable():
    engine = GuardEngine()
    from adaptive_ai.core.error_monitor import ErrorCase, ErrorType, ErrorStatus
    err = ErrorCase(
        id="err-1",
        error_type=ErrorType.TOOL_ERROR,
        status=ErrorStatus.CONFIRMED_ERROR,
        description="Тест",
    )
    guard = engine.create_from_error(err, pattern="danger", action_type="block", applicability=[])
    # Без паттерна в контексте — не сработает
    hits = engine.check_before_action({"context": "safe operation"})
    assert len(hits) == 0


def test_decision_graph_basic():
    graph = DecisionGraph(id="g1", task_id="t1")
    node1 = GraphNode(id="n1", node_type=NodeType.GOAL, result="цель")
    node2 = GraphNode(id="n2", node_type=NodeType.CLAIM, result="утверждение", evidence=[])
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge("n1", "n2")
    assert len(graph.nodes) == 2
    assert graph.get_node("n2").inputs == ["n1"]


def test_decision_graph_validate_claim_without_evidence():
    graph = DecisionGraph(id="g1", task_id="t1")
    node = GraphNode(id="n1", node_type=NodeType.CLAIM, result="утверждение без доказательств")
    graph.add_node(node)
    issues = graph.validate()
    assert any("утверждение без доказательств" in i for i in issues)


def test_experience_journal_append_and_verify():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        journal = ExperienceJournal(path)
        event = journal.append("task_completed", {"task_id": "t1", "result": "ok"})
        assert event.id is not None
        assert journal.count() == 1
        integrity = journal.verify_integrity()
        assert integrity["integrity_ok"] is True
        journal.close()
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = path + suffix
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


def test_experience_journal_multiple_events():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        journal = ExperienceJournal(path)
        journal.append("task_completed", {"task_id": "t1"})
        journal.append("error_detected", {"error": "test"})
        assert journal.count() == 2
        assert journal.count("task_completed") == 1
        assert journal.count("error_detected") == 1
        journal.close()
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = path + suffix
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass
