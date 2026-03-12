"""
API-level tests for /api/chat with deterministic orchestrator stubs.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.agents.orchestrator import nodes as orchestrator_nodes
from backend.agents.orchestrator.session_store import InMemoryConversationStore
from backend.api import routes


class StubCompiledGraph:
    def __init__(self, response: Dict[str, Any]):
        self.response = response
        self.last_state: Dict[str, Any] | None = None

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.last_state = state
        return self.response


class DummyAgent:
    def __init__(self, name: str):
        self.name = name

    def run(
        self,
        user_input: str,
        conversation_history: list[Dict[str, str]],
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


@pytest.fixture
def client():
    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def real_orchestrator_client(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(orchestrator_nodes.ORCHESTRATOR_COMPONENTS, "conversation_store", store)
    monkeypatch.setattr(
        orchestrator_nodes.ORCHESTRATOR_COMPONENTS,
        "agents_instances",
        {
            "quiz": DummyAgent("quiz"),
            "email": DummyAgent("email"),
            "riskmanaging": DummyAgent("riskmanaging"),
            "default_chat": DummyAgent("default_chat"),
        },
    )
    with TestClient(main_module.app, raise_server_exceptions=False) as test_client:
        yield test_client, store


def test_chat_endpoint_returns_chat_response(monkeypatch, client):
    stub = StubCompiledGraph(
        {
            "type": "chat",
            "message": "테스트 응답",
            "report": None,
            "meta": {"source": "stub"},
        }
    )
    monkeypatch.setattr(routes, "compiled_orchestrator_app", stub)

    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-abc",
            "message": "안녕하세요",
            "context": {"mode": "quiz"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "chat"
    assert body["message"] == "테스트 응답"
    assert body["report"] is None
    assert body["meta"]["source"] == "stub"

    assert stub.last_state is not None
    assert stub.last_state["session_id"] == "session-abc"
    assert stub.last_state["user_input"] == "안녕하세요"
    assert stub.last_state["context"] == {"mode": "quiz"}


def test_chat_endpoint_returns_report_response(monkeypatch, client):
    stub = StubCompiledGraph(
        {
            "type": "report",
            "message": "요약",
            "report": {
                "analysis_id": "r1",
                "risk_scoring": {
                    "overall_risk_level": "high",
                    "overall_risk_score": 12.0,
                    "risk_factors": {},
                },
                "response_summary": "요약",
                "suggested_actions": ["조치 1"],
                "similar_cases": [],
                "evidence_sources": [],
            },
            "meta": {},
        }
    )
    monkeypatch.setattr(routes, "compiled_orchestrator_app", stub)

    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-r1",
            "message": "리스크 분석해줘",
            "context": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "report"
    assert body["report"]["analysis_id"] == "r1"
    assert body["report"]["risk_scoring"]["overall_risk_level"] == "high"


def test_chat_endpoint_requires_session_id(client):
    response = client.post(
        "/api/chat",
        json={
            "message": "세션 없는 요청",
            "context": {},
        },
    )
    assert response.status_code == 422


def test_chat_endpoint_real_orchestrator_returns_normalized_chat(real_orchestrator_client):
    client, _store = real_orchestrator_client
    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-real-1",
            "message": "안녕",
            "context": {"mode": "default_chat"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "chat"
    assert body["message"] == "default_chat handled"


def test_chat_endpoint_mode_overrides_existing_active_agent(real_orchestrator_client):
    client, store = real_orchestrator_client
    store.save_state(
        "session-real-2",
        {
            "active_agent": "email",
            "conversation_history": [],
            "agent_specific_state": {},
            "last_interaction_timestamp": time.time(),
        },
    )

    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-real-2",
            "message": "퀴즈로 바꿔",
            "context": {"mode": "quiz"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "chat"
    assert body["message"] == "quiz handled"
