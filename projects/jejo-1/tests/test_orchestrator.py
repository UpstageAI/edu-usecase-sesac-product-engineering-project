"""
Orchestrator node-level tests for current LangGraph architecture.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from backend.agents.orchestrator import nodes as orchestrator_nodes
from backend.agents.orchestrator.session_store import InMemoryConversationStore


def make_state(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "session_id": "session-1",
        "user_input": "안녕하세요",
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


class DummyAgent:
    def __init__(self, name: str, analysis_in_progress: bool = False):
        self.name = name
        self.analysis_in_progress = analysis_in_progress

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
            "analysis_in_progress": self.analysis_in_progress,
        }


@pytest.fixture(autouse=True)
def reset_orchestrator_components(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(orchestrator_nodes.ORCHESTRATOR_COMPONENTS, "conversation_store", store)
    monkeypatch.setattr(
        orchestrator_nodes.ORCHESTRATOR_COMPONENTS,
        "agents_instances",
        {
            "quiz": DummyAgent("quiz"),
            "email": DummyAgent("email"),
            "riskmanaging": DummyAgent("riskmanaging", analysis_in_progress=False),
            "default_chat": DummyAgent("default_chat"),
        },
    )
    return store


def test_load_session_state_initializes_new_session(reset_orchestrator_components):
    state = make_state(session_id="new-session")
    updated = orchestrator_nodes.load_session_state_node(state)

    assert updated["conversation_history"] == []
    assert updated["active_agent"] is None
    assert updated["agent_specific_state"] == {}


def test_detect_intent_routes_to_risk_by_keyword():
    state = make_state(user_input="선적 지연 리스크를 분석해줘")
    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "riskmanaging"


def test_detect_intent_routes_to_quiz_by_keyword_even_with_active_agent():
    state = make_state(
        user_input="인코텀즈 퀴즈 내줘",
        active_agent="email",
    )
    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "quiz"


def test_detect_intent_routes_to_email_by_keyword():
    state = make_state(user_input="고객사에 보낼 메일 초안 작성해줘")
    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "email"


def test_detect_intent_routes_to_email_by_review_keyword():
    state = make_state(user_input="이거 리뷰해줘")
    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "email"


def test_detect_intent_keeps_active_email_for_followup_clarification():
    state = make_state(
        user_input="어떤정보가 필요한데",
        active_agent="email",
        agent_specific_state={"awaiting_follow_up": True},
    )

    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "email"


def test_detect_intent_keeps_active_email_for_short_edit_request():
    state = make_state(
        user_input="한국어로 만들어줄래?",
        active_agent="email",
        agent_specific_state={"awaiting_follow_up": False},
    )

    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "email"


def test_detect_intent_keeps_active_quiz_for_numeric_answer():
    state = make_state(
        user_input="4",
        active_agent="quiz",
        agent_specific_state={"pending_quiz": {"question": "q", "answer": 0}},
    )

    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "quiz"


def test_detect_intent_respects_explicit_mode_context():
    state = make_state(user_input="아무거나", context={"mode": "quiz"})
    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "quiz"


def test_detect_intent_mode_overrides_active_agent():
    state = make_state(
        user_input="리스크 분석 요청",
        context={"mode": "quiz"},
        active_agent="email",
    )
    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "quiz"


def test_detect_intent_default_chat_is_not_sticky(monkeypatch):
    monkeypatch.setattr(orchestrator_nodes, "_classify_intent_with_llm", lambda _: "quiz")
    state = make_state(
        user_input="다음 단계 알려줘",
        active_agent="default_chat",
    )

    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "quiz"
    assert updated["llm_intent_classification"]["predicted_type"] == "quiz"


def test_detect_intent_uses_llm_when_no_fast_path(monkeypatch):
    monkeypatch.setattr(orchestrator_nodes, "_classify_intent_with_llm", lambda _: "email")
    state = make_state(user_input="오늘 뭐부터 하면 좋을까")

    updated = orchestrator_nodes.detect_intent_and_route_node(state)

    assert updated["selected_agent_name"] == "email"
    assert updated["llm_intent_classification"]["predicted_type"] == "email"


async def test_call_agent_node_sets_active_agent_for_normal_agent():
    state = make_state(
        selected_agent_name="quiz",
        user_input="퀴즈",
        conversation_history=[],
        agent_specific_state={},
    )

    updated = await orchestrator_nodes.call_agent_node(state)

    assert updated["active_agent"] == "quiz"
    assert updated["orchestrator_response"]["agent_type"] == "quiz"
    assert len(updated["conversation_history"]) == 2


async def test_call_agent_node_clears_active_agent_when_risk_analysis_complete(monkeypatch):
    monkeypatch.setitem(
        orchestrator_nodes.ORCHESTRATOR_COMPONENTS.agents_instances,
        "riskmanaging",
        DummyAgent("riskmanaging", analysis_in_progress=False),
    )

    state = make_state(
        selected_agent_name="riskmanaging",
        user_input="리스크 분석",
        conversation_history=[],
        agent_specific_state={"analysis_in_progress": True},
    )

    updated = await orchestrator_nodes.call_agent_node(state)

    assert updated["active_agent"] is None
    assert updated["agent_specific_state"]["analysis_in_progress"] is False


async def test_call_agent_node_merges_agent_specific_state(monkeypatch):
    class FollowUpAgent(DummyAgent):
        def run(self, user_input, conversation_history, analysis_in_progress, context=None):
            base = super().run(user_input, conversation_history, analysis_in_progress, context)
            base["agent_specific_state"] = {"awaiting_follow_up": True}
            return base

    monkeypatch.setitem(
        orchestrator_nodes.ORCHESTRATOR_COMPONENTS.agents_instances,
        "email",
        FollowUpAgent("email"),
    )

    state = make_state(
        selected_agent_name="email",
        user_input="리뷰해줘",
        conversation_history=[],
        agent_specific_state={},
    )

    updated = await orchestrator_nodes.call_agent_node(state)

    assert updated["agent_specific_state"]["awaiting_follow_up"] is True


def test_finalize_and_normalize_roundtrip(reset_orchestrator_components):
    state = make_state(
        session_id="save-session",
        selected_agent_name="default_chat",
        active_agent="default_chat",
        conversation_history=[{"role": "Agent", "content": "done"}],
        agent_specific_state={"analysis_in_progress": False},
        orchestrator_response={
            "response": "완료",
            "agent_type": "default_chat",
            "metadata": {},
        },
    )

    saved = orchestrator_nodes.finalize_and_save_state_node(state)
    persisted = reset_orchestrator_components.get_state("save-session")
    normalized = orchestrator_nodes.normalize_response_node(saved)

    assert persisted is not None
    assert persisted["active_agent"] == "default_chat"
    assert normalized["type"] == "chat"
    assert normalized["message"] == "완료"
