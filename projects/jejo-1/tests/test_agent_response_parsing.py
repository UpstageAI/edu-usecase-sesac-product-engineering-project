import json
from types import SimpleNamespace

from backend.agents.email_agent import nodes as email_nodes
from backend.agents.quiz_agent import nodes as quiz_nodes


def _mock_llm(content: str) -> SimpleNamespace:
    class _MockCompletions:
        def __init__(self, response_content: str):
            self._response_content = response_content

        def create(self, *args, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=self._response_content)
                    )
                ]
            )

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=_MockCompletions(content),
        )
    )


def test_email_node_uses_email_content_field(monkeypatch):
    payload = {
        "email_content": "안녕하세요. BL 초안 검토 부탁드립니다.",
        "recipient_country": "사우디아라비아",
    }
    monkeypatch.setattr(
        email_nodes.EMAIL_AGENT_COMPONENTS,
        "llm",
        _mock_llm(json.dumps(payload, ensure_ascii=False)),
    )

    updated = email_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "이메일 초안만들어"}]}
    )

    assert updated["final_response_content"] == payload["email_content"]
    assert updated["llm_output_details"] == payload


def test_email_node_uses_error_message_from_nested_error(monkeypatch):
    payload = {
        "error": {
            "message": "입력 정보가 부족합니다.",
            "required_fields": ["email_content", "recipient_country", "purpose"],
        }
    }
    monkeypatch.setattr(
        email_nodes.EMAIL_AGENT_COMPONENTS,
        "llm",
        _mock_llm(json.dumps(payload, ensure_ascii=False)),
    )

    updated = email_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "내일 서울날씨"}]}
    )

    assert updated["final_response_content"] == "입력 정보가 부족합니다."


def test_email_node_builds_missing_fields_message(monkeypatch):
    payload = {
        "status": "error",
        "required_fields": ["email_content", "recipient_country", "purpose"],
    }
    monkeypatch.setattr(
        email_nodes.EMAIL_AGENT_COMPONENTS,
        "llm",
        _mock_llm(json.dumps(payload, ensure_ascii=False)),
    )

    updated = email_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "안녕"}]}
    )

    assert updated["final_response_content"] == (
        "필수 입력 정보가 부족합니다: email_content, recipient_country, purpose"
    )


def test_email_node_parses_fenced_json_payload(monkeypatch):
    payload = {"email_content": "안녕하세요. 요청하신 초안을 공유드립니다."}
    monkeypatch.setattr(
        email_nodes.EMAIL_AGENT_COMPONENTS,
        "llm",
        _mock_llm(f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"),
    )

    updated = email_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "초안 작성"}]}
    )

    assert updated["final_response_content"] == payload["email_content"]


def test_quiz_node_summarizes_generated_quiz(monkeypatch):
    payload = {
        "answer": [
            {
                "question": "FOB 조건에서 운임 부담 주체는 누구인가요?",
                "choices": ["매수인", "매도인", "운송사", "보험사"],
            }
        ],
    }
    monkeypatch.setattr(
        quiz_nodes.QUIZ_AGENT_COMPONENTS,
        "llm",
        _mock_llm(json.dumps(payload, ensure_ascii=False)),
    )

    updated = quiz_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "퀴즈줘"}]}
    )

    assert "[퀴즈 1]" in updated["final_response_content"]
    assert "FOB 조건에서 운임 부담 주체는 누구인가요?" in updated["final_response_content"]
    assert "1. 매수인" in updated["final_response_content"]


def test_quiz_node_parses_json_array_payload(monkeypatch):
    payload = [
        {
            "question": "신용장의 역할은 무엇인가요?",
            "choices": ["운송계약", "대금지급 보증", "통관신고", "보험계약"],
            "answer": 1,
        }
    ]
    monkeypatch.setattr(
        quiz_nodes.QUIZ_AGENT_COMPONENTS,
        "llm",
        _mock_llm(json.dumps(payload, ensure_ascii=False)),
    )

    updated = quiz_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "무역 용어 퀴즈"}]}
    )

    assert "[퀴즈 1]" in updated["final_response_content"]
    assert "신용장의 역할은 무엇인가요?" in updated["final_response_content"]
    assert "대금지급 보증" in updated["final_response_content"]


def test_quiz_node_parses_fenced_json_array_payload(monkeypatch):
    payload = [
        {
            "question": "B/L의 주요 기능은 무엇인가요?",
            "choices": ["운송계약 및 화물 인수 증빙", "관세율 계산", "환율 고정", "보험료 산정"],
            "answer": 0,
        }
    ]
    monkeypatch.setattr(
        quiz_nodes.QUIZ_AGENT_COMPONENTS,
        "llm",
        _mock_llm(f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"),
    )

    updated = quiz_nodes.call_llm_and_parse_response_node(
        {"llm_messages": [{"role": "user", "content": "무역 실무 퀴즈"}]}
    )

    assert "[퀴즈 1]" in updated["final_response_content"]
    assert "B/L의 주요 기능은 무엇인가요?" in updated["final_response_content"]
    assert "운송계약 및 화물 인수 증빙" in updated["final_response_content"]


def test_email_node_review_mode_avoids_echoing_email_content(monkeypatch):
    email_body = "Dear Mr. Johnson,\nPlease review the attached BL draft.\nBest regards,\nKim"
    payload = {"email_content": email_body}
    monkeypatch.setattr(
        email_nodes.EMAIL_AGENT_COMPONENTS,
        "llm",
        _mock_llm(json.dumps(payload, ensure_ascii=False)),
    )
    monkeypatch.setattr(
        email_nodes,
        "_build_rule_based_review",
        lambda *args, **kwargs: "RULE_BASED_REVIEW",
    )

    updated = email_nodes.call_llm_and_parse_response_node(
        {
            "llm_messages": [{"role": "user", "content": "리뷰"}],
            "email_task_type": "review",
            "extracted_email_content": email_body,
            "extracted_recipient_country": "미국",
            "extracted_purpose": "이메일 검토",
        }
    )

    assert updated["final_response_content"] == "RULE_BASED_REVIEW"


def test_email_node_review_mode_with_unknown_country_avoids_country_assumption(monkeypatch):
    email_body = "안녕하세요.\nBL 초안을 검토 부탁드립니다.\n감사합니다."
    monkeypatch.setattr(
        email_nodes.EMAIL_AGENT_COMPONENTS,
        "llm",
        _mock_llm("미국 기업 기준으로는 빠른 답변 요청이 압박으로 보일 수 있습니다."),
    )
    monkeypatch.setattr(
        email_nodes,
        "_build_rule_based_review",
        lambda *args, **kwargs: "RULE_BASED_REVIEW_NO_COUNTRY",
    )

    updated = email_nodes.call_llm_and_parse_response_node(
        {
            "llm_messages": [{"role": "user", "content": "리뷰"}],
            "email_task_type": "review",
            "extracted_email_content": email_body,
            "extracted_recipient_country": "미지정",
            "extracted_purpose": "이메일 검토",
        }
    )

    assert updated["final_response_content"] == "RULE_BASED_REVIEW_NO_COUNTRY"


def test_email_task_detection_prefers_draft_for_review_purpose_wording():
    task = email_nodes._detect_email_task_type(
        "수신자는 미국 바이어고 서류검토를 위한 이메일 초안만들어",
        {},
    )
    assert task == "draft"


def test_email_task_detection_keeps_explicit_review_request():
    task = email_nodes._detect_email_task_type(
        "이거 이메일 리뷰해줘",
        {},
    )
    assert task == "review"
