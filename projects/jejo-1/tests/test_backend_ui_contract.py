from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, List

from fastapi.testclient import TestClient

import backend.main as main_module
from backend.agents.orchestrator import nodes as orchestrator_nodes
from backend.agents.orchestrator.session_store import InMemoryConversationStore
from backend.api import routes


class _StubCompiledGraph:
    def __init__(self, response: Dict[str, Any]):
        self.response = response

    async def ainvoke(self, _state: Dict[str, Any]) -> Dict[str, Any]:
        return self.response


class _DummyRiskAgent:
    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        analysis_in_progress: bool,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        updated_history = list(conversation_history)
        updated_history.append({"role": "User", "content": user_input})

        if not analysis_in_progress:
            follow_up = (
                "리스크 평가를 위한 핵심 정보가 부족합니다. "
                "계약 규모, 페널티 조항, 예상 지연 기간을 알려주세요."
            )
            updated_history.append({"role": "Agent", "content": follow_up})
            return {
                "response": {
                    "response": follow_up,
                    "agent_type": "riskmanaging",
                    "metadata": {"status": "insufficient_info"},
                },
                "conversation_history": updated_history,
                "analysis_in_progress": True,
            }

        report = {
            "analysis_id": "abf220d2",
            "input_summary": "10만달러 계약, 일당 0.5% 페널티, 약 2주 지연 상황",
            "risk_scoring": {
                "overall_risk_level": "medium",
                "risk_factors": [
                    {
                        "name": "재정적 손실",
                        "impact": 3,
                        "likelihood": 4,
                        "risk_score": 12,
                        "reasoning": "지연 페널티가 누적되어 손실 가능성이 높습니다.",
                        "mitigation_suggestions": ["선적 분할 또는 납기 재협상"],
                    }
                ],
            },
            "prevention_strategy": {"short_term": ["고객사에 즉시 ETA 변경 통보"]},
            "similar_cases": [{"content": "delay penalty", "source": "cases.csv"}],
            "evidence_sources": ["cases.csv"],
        }
        updated_history.append({"role": "Agent", "content": "risk report generated"})
        return {
            "response": {
                "response": json.dumps(report, ensure_ascii=False),
                "agent_type": "riskmanaging",
                "metadata": {"status": "success"},
            },
            "conversation_history": updated_history,
            "analysis_in_progress": False,
        }


class _DummyDefaultAgent:
    def __init__(self, name: str):
        self.name = name

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        analysis_in_progress: bool,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        updated_history = list(conversation_history)
        updated_history.append({"role": "User", "content": user_input})
        updated_history.append({"role": "Agent", "content": f"{self.name} handled"})
        return {
            "response": {
                "response": f"{self.name} handled",
                "agent_type": self.name,
                "metadata": {},
            },
            "conversation_history": updated_history,
            "analysis_in_progress": False,
        }


def test_chat_route_fallback_normalizes_new_report_schema(monkeypatch):
    raw_report = {
        "analysis_id": "abf220d2",
        "input_summary": "선적 지연으로 페널티 리스크가 있습니다.",
        "risk_scoring": {
            "overall_risk_level": "medium",
            "risk_factors": [
                {"name": "재정적 손실", "impact": 3, "likelihood": 4, "risk_score": 12}
            ],
        },
        "prevention_strategy": {"short_term": ["고객사 선제 통보"]},
    }
    stub = _StubCompiledGraph(
        {
            "orchestrator_response": {
                "response": json.dumps(raw_report, ensure_ascii=False),
                "agent_type": "riskmanaging",
                "metadata": {"status": "success"},
            }
        }
    )
    monkeypatch.setattr(routes, "compiled_orchestrator_app", stub)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/chat",
            json={"session_id": "session-contract-1", "message": "리스크 분석", "context": {}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "report"
    assert body["report"]["analysis_id"] == "abf220d2"
    assert body["report"]["risk_scoring"]["overall_risk_score"] == 12.0
    assert body["report"]["response_summary"] == "선적 지연으로 페널티 리스크가 있습니다."
    assert "고객사 선제 통보" in body["report"]["suggested_actions"]


def test_backend_risk_flow_transitions_from_followup_to_report(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(orchestrator_nodes.ORCHESTRATOR_COMPONENTS, "conversation_store", store)
    monkeypatch.setattr(
        orchestrator_nodes.ORCHESTRATOR_COMPONENTS,
        "agents_instances",
        {
            "riskmanaging": _DummyRiskAgent(),
            "default_chat": _DummyDefaultAgent("default_chat"),
            "quiz": _DummyDefaultAgent("quiz"),
            "email": _DummyDefaultAgent("email"),
        },
    )

    session_id = "session-risk-flow-1"
    with TestClient(main_module.app, raise_server_exceptions=False) as client:
        first = client.post(
            "/api/chat",
            json={
                "session_id": session_id,
                "message": "선적이 늦어져서 패널티가 걱정돼요",
                "context": {"mode": "riskmanaging"},
            },
        )
        state_after_first = copy.deepcopy(store.get_state(session_id))
        second = client.post(
            "/api/chat",
            json={
                "session_id": session_id,
                "message": "10만달러에 일당 0.5퍼센트, 지연은 약 2주입니다",
                "context": {},
            },
        )

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["type"] == "chat"
    assert "핵심 정보가 부족" in first_body["message"]

    assert state_after_first is not None
    assert state_after_first["active_agent"] == "riskmanaging"
    assert state_after_first["agent_specific_state"].get("analysis_in_progress") is True

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["type"] == "report"
    assert second_body["report"]["analysis_id"] == "abf220d2"
    assert second_body["report"]["risk_scoring"]["overall_risk_score"] == 12.0
    assert second_body["report"]["response_summary"] == "10만달러 계약, 일당 0.5% 페널티, 약 2주 지연 상황"

    state_after_second = copy.deepcopy(store.get_state(session_id))
    assert state_after_second is not None
    assert state_after_second["active_agent"] is None
    assert state_after_second["agent_specific_state"].get("analysis_in_progress") is False
    assert state_after_second["last_interaction_timestamp"] <= time.time()
