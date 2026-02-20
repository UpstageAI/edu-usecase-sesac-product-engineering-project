from __future__ import annotations

from typing import Any, Dict

from backend.agents.email_agent import email_agent as email_agent_module
from backend.agents.quiz_agent import quiz_agent as quiz_agent_module


class _StubCompiledGraph:
    def __init__(self, final_state: Dict[str, Any]):
        self.final_state = final_state

    async def ainvoke(self, _state: Dict[str, Any]) -> Dict[str, Any]:
        return self.final_state


async def test_email_agent_run_returns_payload_and_sets_followup_state(monkeypatch):
    monkeypatch.setattr(
        email_agent_module,
        "compiled_email_agent_app",
        _StubCompiledGraph(
            {
                "agent_output_for_orchestrator": {
                    "response": "리뷰를 위해 이메일 본문이 필요합니다.",
                    "agent_type": "email",
                    "metadata": {
                        "llm_output_details": {
                            "required_fields": ["email_content"],
                        }
                    },
                },
                "conversation_history": [],
                "analysis_in_progress": False,
            }
        ),
    )

    agent = email_agent_module.EmailAgent()
    result = await agent.run(
        user_input="이거 리뷰해줘",
        conversation_history=[],
        analysis_in_progress=False,
        context={},
    )

    assert result["response"]["agent_type"] == "email"
    assert result["agent_specific_state"]["awaiting_follow_up"] is True
    assert len(result["conversation_history"]) == 2
    assert result["conversation_history"][0]["role"] == "User"
    assert result["conversation_history"][1]["role"] == "Agent"


async def test_quiz_agent_run_returns_payload_and_updates_history(monkeypatch):
    monkeypatch.setattr(
        quiz_agent_module,
        "compiled_quiz_agent_app",
        _StubCompiledGraph(
            {
                "agent_output_for_orchestrator": {
                    "response": "[퀴즈 1]\nFOB는 무엇의 약자인가요?",
                    "agent_type": "quiz",
                    "metadata": {"llm_output_details": {"questions": [{"question": "FOB"}]}},
                },
                "conversation_history": [],
                "analysis_in_progress": False,
            }
        ),
    )

    agent = quiz_agent_module.QuizAgent()
    result = await agent.run(
        user_input="퀴즈줘",
        conversation_history=[],
        analysis_in_progress=False,
        context={},
    )

    assert result["response"]["agent_type"] == "quiz"
    assert result["agent_specific_state"]["awaiting_follow_up"] is False
    assert len(result["conversation_history"]) == 2
    assert result["conversation_history"][0]["content"] == "퀴즈줘"
    assert "[퀴즈 1]" in result["conversation_history"][1]["content"]


async def test_quiz_agent_run_sets_pending_quiz_when_answer_exists(monkeypatch):
    monkeypatch.setattr(
        quiz_agent_module,
        "compiled_quiz_agent_app",
        _StubCompiledGraph(
            {
                "agent_output_for_orchestrator": {
                    "response": "[퀴즈 1]\nFOB는 무엇의 약자인가요?",
                    "agent_type": "quiz",
                    "metadata": {
                        "llm_output_details": {
                            "questions": [
                                {
                                    "question": "FOB는 무엇의 약자인가요?",
                                    "choices": ["Free On Board", "Freight On Bill", "Final Order Base", "Forwarder On Booking"],
                                    "answer": 0,
                                    "explanation": "FOB는 Free On Board입니다.",
                                }
                            ]
                        }
                    },
                },
                "conversation_history": [],
                "analysis_in_progress": False,
            }
        ),
    )

    agent = quiz_agent_module.QuizAgent()
    result = await agent.run(
        user_input="퀴즈줘",
        conversation_history=[],
        analysis_in_progress=False,
        context={},
    )

    assert result["agent_specific_state"]["awaiting_follow_up"] is True
    assert result["agent_specific_state"]["pending_quiz"]["answer"] == 0


async def test_quiz_agent_run_evaluates_numeric_answer_from_pending_quiz():
    agent = quiz_agent_module.QuizAgent()
    result = await agent.run(
        user_input="4",
        conversation_history=[{"role": "Agent", "content": "[퀴즈 1]"}],
        analysis_in_progress=False,
        context={
            "_agent_specific_state": {
                "pending_quiz": {
                    "question": "FOB는 무엇의 약자인가요?",
                    "choices": ["Free On Board", "Freight On Bill", "Final Order Base", "Forwarder On Booking"],
                    "answer": 0,
                    "explanation": "FOB는 Free On Board입니다.",
                }
            }
        },
    )

    assert result["response"]["agent_type"] == "quiz"
    assert ("정답입니다" in result["response"]["response"]) or ("오답입니다" in result["response"]["response"])
    assert result["agent_specific_state"]["pending_quiz"] is None
