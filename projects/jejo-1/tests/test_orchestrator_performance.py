"""
Micro performance tests for core orchestrator node operations.

These tests avoid external network/API dependencies.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from backend.agents.orchestrator import nodes as orchestrator_nodes


def make_state(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "session_id": "perf-session",
        "user_input": "선적 지연 리스크를 검토해줘",
        "context": {},
        "conversation_history": [],
        "active_agent": None,
        "agent_specific_state": {},
        "orchestrator_response": None,
        "llm_intent_classification": None,
        "selected_agent_name": None,
    }
    base.update(overrides)
    return base


def test_keyword_routing_speed():
    state = make_state()

    start = time.perf_counter()
    for _ in range(2000):
        updated = orchestrator_nodes.detect_intent_and_route_node(dict(state))
        assert updated["selected_agent_name"] == "riskmanaging"
    elapsed = time.perf_counter() - start

    # Very generous threshold to avoid flaky CI failures.
    assert elapsed < 2.0


def test_response_normalization_speed():
    report_payload = {
        "response": {
            "analysis_id": "a1",
            "risk_scoring": {
                "overall_risk_level": "high",
                "overall_risk_score": 12.0,
                "risk_factors": [],
            },
            "response_summary": "요약",
            "suggested_actions": ["조치 1"],
        },
        "agent_type": "riskmanaging",
        "metadata": {},
    }

    state = make_state(orchestrator_response=report_payload)

    start = time.perf_counter()
    for _ in range(2000):
        normalized = orchestrator_nodes.normalize_response_node(dict(state))
        assert normalized["type"] == "report"
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0
