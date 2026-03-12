import json

import pytest

from backend.core.response_converter import normalize_response


def test_normalize_response_maps_new_risk_report_schema():
    raw_report = {
        "analysis_id": "abf220d2",
        "input_summary": "선적 지연으로 패널티 발생 가능성이 높습니다.",
        "risk_factors": {
            "재정적 손실": {
                "name_kr": "재정적 손실",
                "impact": 3,
                "likelihood": 4,
                "score": 12,
            }
        },
        "risk_scoring": {
            "overall_risk_level": "medium",
            "overall_assessment": "지연 페널티 발생 가능성이 있습니다.",
            "risk_factors": [
                {
                    "name": "재정적 손실",
                    "impact": 3,
                    "likelihood": 4,
                    "risk_score": 12,
                    "reasoning": "일당 0.5% 조항이 있어 손실이 누적됩니다.",
                    "mitigation_suggestions": ["납기 재협상", "부분 선적 검토"],
                },
                {
                    "name": "일정 지연",
                    "impact": 4,
                    "likelihood": 3,
                    "risk_score": 12,
                    "reasoning": "2주 지연이 확정적입니다.",
                },
            ],
        },
        "prevention_strategy": {
            "short_term": ["선복 우선 배정 요청"],
            "long_term": ["버퍼 리드타임 7일 반영"],
        },
        "control_gap_analysis": {
            "recommendations": ["지연 알림 SLA를 계약서에 명시"],
        },
        "similar_cases": [
            {"content": "shipment delay discount", "source": "casebook.csv"}
        ],
        "evidence_sources": ["casebook.csv"],
    }

    result = normalize_response({"response": json.dumps(raw_report, ensure_ascii=False)})

    assert result["type"] == "report"
    assert result["report"]["analysis_id"] == "abf220d2"
    assert result["message"] == "선적 지연으로 패널티 발생 가능성이 높습니다."
    assert result["report"]["response_summary"] == "선적 지연으로 패널티 발생 가능성이 높습니다."

    scoring = result["report"]["risk_scoring"]
    assert scoring["overall_risk_level"] == "medium"
    assert scoring["overall_risk_score"] == pytest.approx(12.0)
    assert "재정적 손실" in scoring["risk_factors"]
    assert scoring["risk_factors"]["재정적 손실"]["score"] == pytest.approx(12.0)
    assert scoring["risk_factors"]["재정적 손실"]["reason"] == "일당 0.5% 조항이 있어 손실이 누적됩니다."

    suggested_actions = result["report"]["suggested_actions"]
    assert "선복 우선 배정 요청" in suggested_actions
    assert "버퍼 리드타임 7일 반영" in suggested_actions
    assert "지연 알림 SLA를 계약서에 명시" in suggested_actions
    assert "납기 재협상" in suggested_actions


def test_normalize_response_keeps_chat_message_when_not_report():
    result = normalize_response({"response": "일반적인 답변입니다."})
    assert result["type"] == "chat"
    assert result["message"] == "일반적인 답변입니다."
    assert result["report"] is None


def test_normalize_response_formats_quiz_json_array_string():
    raw = {
        "response": json.dumps(
            [
                {
                    "question": "FOB는 무엇의 약자인가요?",
                    "choices": ["Free On Board", "Freight On Bill", "Final Order Base", "Forwarder On Booking"],
                    "answer": 0,
                }
            ],
            ensure_ascii=False,
        )
    }

    result = normalize_response(raw)

    assert result["type"] == "chat"
    assert "[퀴즈 1]" in result["message"]
    assert "FOB는 무엇의 약자인가요?" in result["message"]
    assert "1. Free On Board" in result["message"]
    assert len(result["meta"]["quiz_questions"]) == 1
    assert result["meta"]["quiz_questions"][0]["question"] == "FOB는 무엇의 약자인가요?"


def test_normalize_response_formats_quiz_fenced_json_array_string():
    payload = [
        {
            "question": "B/L의 기능은 무엇인가요?",
            "choices": ["운송계약 증빙", "세율 결정", "환율 고정", "보험료 계산"],
            "answer": 0,
        }
    ]
    raw = {"response": f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"}

    result = normalize_response(raw)

    assert result["type"] == "chat"
    assert "[퀴즈 1]" in result["message"]
    assert "B/L의 기능은 무엇인가요?" in result["message"]


def test_normalize_response_extracts_quiz_questions_from_agent_metadata():
    raw = {
        "response": "[퀴즈 1]\nFOB는 무엇의 약자인가요?\n1. Free On Board\n2. Freight On Bill",
        "agent_type": "quiz",
        "metadata": {
            "llm_output_details": {
                "questions": [
                    {
                        "question": "FOB는 무엇의 약자인가요?",
                        "choices": ["Free On Board", "Freight On Bill", "Final Order Base", "Forwarder On Booking"],
                        "answer": 0,
                        "explanation": "FOB는 Free On Board입니다.",
                    },
                    {
                        "question": "CIF에서 운임 부담 주체는 누구인가요?",
                        "choices": ["매수인", "매도인", "보험사", "은행"],
                        "answer": 1,
                        "explanation": "CIF는 매도인이 운임과 보험료를 부담합니다.",
                    },
                ]
            }
        },
    }

    result = normalize_response(raw)

    assert result["type"] == "chat"
    assert "[퀴즈 1]" in result["message"]
    assert len(result["meta"]["quiz_questions"]) == 2
