from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.agents.default_chat.default_chat_agent import DefaultChatAgent
from backend.agents.email_agent import nodes as email_nodes
from backend.agents.email_agent.email_agent import EmailAgent
from backend.agents.orchestrator import nodes as orchestrator_nodes
from backend.agents.orchestrator.session_store import InMemoryConversationStore
from backend.agents.quiz_agent import nodes as quiz_nodes
from backend.agents.quiz_agent.quiz_agent import QuizAgent


class _QueuedLLM:
    def __init__(self, responses: List[str]):
        self._responses = list(responses)

        class _Completions:
            def __init__(self, parent: "_QueuedLLM"):
                self._parent = parent

            def create(self, *args, **kwargs):
                if not self._parent._responses:
                    raise RuntimeError("No queued LLM response")
                content = self._parent._responses.pop(0)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        self.chat = SimpleNamespace(completions=_Completions(self))


class _DummyRiskAgent:
    def run(self, user_input, conversation_history, analysis_in_progress, context=None):
        updated = list(conversation_history)
        updated.append({"role": "User", "content": user_input})
        updated.append({"role": "Agent", "content": "risk stub"})
        return {
            "response": {"response": "risk stub", "agent_type": "riskmanaging", "metadata": {}},
            "conversation_history": updated,
            "analysis_in_progress": False,
        }


@pytest.fixture
def regression_client(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(orchestrator_nodes.ORCHESTRATOR_COMPONENTS, "conversation_store", store)
    monkeypatch.setattr(
        orchestrator_nodes.ORCHESTRATOR_COMPONENTS,
        "agents_instances",
        {
            "quiz": QuizAgent(),
            "email": EmailAgent(),
            "riskmanaging": _DummyRiskAgent(),
            "default_chat": DefaultChatAgent(),
        },
    )
    with TestClient(main_module.app, raise_server_exceptions=False) as client:
        yield client


def test_quiz_followup_question_does_not_fall_back_to_default_chat(monkeypatch, regression_client):
    quiz_llm = _QueuedLLM(
        [
            json.dumps(
                {
                    "error": {
                        "message": "입력 데이터가 부족합니다. 무역 용어 퀴즈 생성을 위해 주제가 필요합니다.",
                        "required_fields": ["topic"],
                    }
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "quiz_response": "필요한 정보는 퀴즈 주제입니다. 예: 인코텀즈, 결제조건, 선적서류",
                },
                ensure_ascii=False,
            ),
        ]
    )
    monkeypatch.setattr(quiz_nodes.QUIZ_AGENT_COMPONENTS, "llm", quiz_llm)

    session_id = "flow-quiz-1"
    first = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "무역용어 퀴즈내줘", "context": {}},
    )
    second = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "어떤정보가 필요한데", "context": {}},
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_msg = first.json()["message"]
    second_msg = second.json()["message"]
    assert "입력 데이터가 부족합니다" in first_msg
    assert "일반적인 답변" not in second_msg
    assert "필요한 정보는 퀴즈 주제" in second_msg


def test_email_draft_then_review_uses_history_and_returns_review(monkeypatch, regression_client):
    draft_email = (
        "Dear Mr. Johnson,\n\n"
        "Please review the attached BL draft and share your feedback.\n\n"
        "Best regards,\n[Your Name]"
    )
    email_llm = _QueuedLLM(
        [
            draft_email,
            json.dumps({"email_content": draft_email}, ensure_ascii=False),
        ]
    )
    monkeypatch.setattr(email_nodes.EMAIL_AGENT_COMPONENTS, "llm", email_llm)

    session_id = "flow-email-1"
    first = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "이메일 초안 영어로만들어줘", "context": {}},
    )
    second = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "이거 이메일 리뷰해줘", "context": {}},
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_msg = first.json()["message"]
    second_msg = second.json()["message"]
    assert "Dear Mr. Johnson" in first_msg
    assert "이메일 리뷰 요약" in second_msg
    assert second_msg.strip() != draft_email.strip()


def test_email_followup_language_request_keeps_email_context(monkeypatch, regression_client):
    draft_en = (
        "Dear Mr. Johnson,\n\n"
        "Please review the attached BL draft and share your feedback.\n\n"
        "Best regards,\n[Your Name]"
    )
    draft_ko = (
        "안녕하세요,\n\n"
        "첨부된 BL 초안을 검토해 주시면 감사하겠습니다.\n\n"
        "감사합니다."
    )
    email_llm = _QueuedLLM([draft_en, draft_ko])
    monkeypatch.setattr(email_nodes.EMAIL_AGENT_COMPONENTS, "llm", email_llm)

    session_id = "flow-email-lang-1"
    first = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "이메일 초안 영어로 만들어줘", "context": {}},
    )
    second = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "한국어로 만들어줄래?", "context": {}},
    )

    assert first.status_code == 200
    assert second.status_code == 200

    second_msg = second.json()["message"]
    assert "일반적인 답변" not in second_msg
    assert "안녕하세요" in second_msg


def test_quiz_answer_turn_is_evaluated_instead_of_falling_back(monkeypatch, regression_client):
    quiz_payload = [
        {
            "question": "FOB는 무엇의 약자인가요?",
            "choices": [
                "Free On Board",
                "Freight On Bill",
                "Final Order Base",
                "Forwarder On Booking",
            ],
            "answer": 0,
            "explanation": "FOB는 Free On Board의 약자입니다.",
            "quiz_type": "용어→설명",
            "difficulty": "medium",
        }
    ]
    # Single queued response: second turn(답안 입력)은 LLM 없이 채점 경로로 처리되어야 함.
    monkeypatch.setattr(
        quiz_nodes.QUIZ_AGENT_COMPONENTS,
        "llm",
        _QueuedLLM([json.dumps(quiz_payload, ensure_ascii=False)]),
    )

    session_id = "flow-quiz-answer-1"
    first = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "무역용어 퀴즈내줘", "context": {}},
    )
    second = regression_client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "4", "context": {}},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert "[퀴즈 1]" in first.json()["message"]
    assert "일반적인 답변" not in second.json()["message"]
    assert ("정답입니다" in second.json()["message"]) or ("오답입니다" in second.json()["message"])
