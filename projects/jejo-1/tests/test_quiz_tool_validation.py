from __future__ import annotations

import backend.agents.eval_agent as eval_agent
from backend.agents.quiz_agent.tools import validate_quiz_quality


def _sample_quiz_data():
    return {
        "questions": [
            {
                "quiz_id": "q1",
                "question": "FOB란 무엇인가요?",
                "choices": ["Free On Board", "Final Order Base", "Freight On Bill", "None"],
                "answer": 0,
                "explanation": "FOB는 Free On Board의 약자입니다.",
            }
        ]
    }


def test_validate_quiz_quality_runs_async_eval_from_sync_context(monkeypatch):
    async def fake_evaluate_quiz_list(_questions):
        return [{"quiz_id": "q1", "is_valid": True, "issues": []}]

    monkeypatch.setattr(eval_agent, "evaluate_quiz_list", fake_evaluate_quiz_list)

    result = validate_quiz_quality.invoke({"quiz_data": _sample_quiz_data()})

    assert result["is_valid"] is True
    assert result["total_questions"] == 1
    assert result["valid_questions"] == 1
    assert result["issues"] == []


async def test_validate_quiz_quality_runs_async_eval_inside_running_loop(monkeypatch):
    async def fake_evaluate_quiz_list(_questions):
        return [{"quiz_id": "q1", "is_valid": False, "issues": ["grounding 부족"]}]

    monkeypatch.setattr(eval_agent, "evaluate_quiz_list", fake_evaluate_quiz_list)

    result = validate_quiz_quality.invoke({"quiz_data": _sample_quiz_data()})

    assert result["is_valid"] is False
    assert result["total_questions"] == 1
    assert result["valid_questions"] == 0
    assert result["issues"] == ["Question 1: grounding 부족"]
