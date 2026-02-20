"""
IntentClassifier tests (offline, deterministic).
"""
from __future__ import annotations

import pytest

from backend.agents.intent_classifier import IntentClassifier


class StubLLM:
    def __init__(self, response: str = "분류: general_chat", should_raise: bool = False):
        self.response = response
        self.should_raise = should_raise
        self.last_prompt = ""

    def invoke(self, prompt: str, temperature=None) -> str:
        self.last_prompt = prompt
        if self.should_raise:
            raise RuntimeError("forced llm failure")
        return self.response


@pytest.mark.parametrize(
    ("llm_response", "expected"),
    [
        ("분류: email_coach", "email_coach"),
        ("분류: quiz", "quiz"),
        ("분류: risk_detect", "risk_detect"),
        ("분류: out_of_scope", "out_of_scope"),
        ("분류: general_chat", "general_chat"),
    ],
)
def test_classify_parses_expected_intent(llm_response, expected):
    classifier = IntentClassifier(StubLLM(response=llm_response))
    result = classifier.classify("아무 입력", {})
    assert result == expected


def test_classify_uses_prompt_with_user_input():
    llm = StubLLM(response="분류: quiz")
    classifier = IntentClassifier(llm)

    user_input = "퀴즈 내줘"
    classifier.classify(user_input, {})

    assert user_input in llm.last_prompt


def test_parse_intent_keyword_fallback():
    classifier = IntentClassifier(StubLLM(response="unknown format"))

    assert classifier._parse_intent("I think this is email_coach") == "email_coach"
    assert classifier._parse_intent("maybe quiz") == "quiz"
    assert classifier._parse_intent("risk_detect requested") == "risk_detect"
    assert classifier._parse_intent("out_of_scope question") == "out_of_scope"


def test_classify_falls_back_to_general_chat_on_llm_error():
    classifier = IntentClassifier(StubLLM(should_raise=True))
    result = classifier.classify("이메일 검토해줘", {})
    assert result == "general_chat"
