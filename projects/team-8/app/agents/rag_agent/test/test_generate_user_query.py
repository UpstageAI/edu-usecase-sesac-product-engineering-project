import sys
import os
import re
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[4]))

from app.agents.rag_agent.nodes import generate_user_query as generate_user_query_module
from app.agents.rag_agent.state.rag_state import GenerateQueryTextOutput
from app.agents.user_input_template_agent.state import HealthCondition
from app.agents.vet_agent.state import DiseaseInfo, VetAgentState

load_dotenv()


class _FakeStructuredLLM:
    def __init__(self, captured: dict):
        self._captured = captured

    def invoke(self, messages) -> GenerateQueryTextOutput:
        self._captured.setdefault("messages_list", []).append(messages)
        user_prompt = messages[1]["content"]
        match = re.search(r"condition_value:\s*(.+)", user_prompt)
        condition_value = match.group(1).strip() if match else "unknown"
        return GenerateQueryTextOutput(query_text=f"테스트 질의 - {condition_value}")


class _FakeLLM:
    def __init__(self, captured: dict):
        self._captured = captured

    def with_structured_output(self, schema):
        self._captured["schema"] = schema
        return _FakeStructuredLLM(self._captured)


def test_generate_user_query_raises_when_both_health_and_diseases_are_missing():
    state = VetAgentState()

    with pytest.raises(ValueError, match="invalid VetAgentState !"):
        generate_user_query_module.generate_user_query(state)


def test_generate_user_query_uses_only_health_condition_and_diseases(monkeypatch):
    captured: dict = {}

    def _fake_init_chat_model(*, model: str, temperature: float):
        captured["model"] = model
        captured["temperature"] = temperature
        return _FakeLLM(captured)

    monkeypatch.setattr(
        generate_user_query_module, "init_chat_model", _fake_init_chat_model
    )

    state = VetAgentState(
        species="SHOULD_NOT_APPEAR_SPECIES",
        breed="SHOULD_NOT_APPEAR_BREED",
        age=99,
        weight=999,
        health_condition=HealthCondition(
            frequent_illness_area="피부",
            disease_surgery_history="슬개골 수술 이력",
        ),
        diseases=[
            DiseaseInfo(
                name="슬개골 탈구", incidence_rate="높음", onset_period="노령기"
            )
        ],
    )

    result = generate_user_query_module.generate_user_query(state)

    assert "user_query" in result
    assert "query_texts" in result
    assert len(result["query_texts"]) == 2
    assert "테스트 질의 - 피부" in result["query_texts"]
    assert "테스트 질의 - 슬개골 탈구" in result["query_texts"]
    assert result["user_query"] == "- 테스트 질의 - 피부\n- 테스트 질의 - 슬개골 탈구"
    assert captured["model"] == "solar-pro2"
    assert captured["temperature"] == 0.0
    assert captured["schema"] is GenerateQueryTextOutput

    messages_list = captured["messages_list"]
    assert len(messages_list) == 2

    all_user_prompts = []
    for messages in messages_list:
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "반려동물 보험 전문 보험 설계사 출신 CTO" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        all_user_prompts.append(messages[1]["content"])

    joined_user_prompts = "\n".join(all_user_prompts)
    assert "condition_type: frequent_illness_area" in joined_user_prompts
    assert "condition_value: 피부" in joined_user_prompts
    assert "condition_type: disease_name" in joined_user_prompts
    assert "condition_value: 슬개골 탈구" in joined_user_prompts
    assert "SHOULD_NOT_APPEAR_SPECIES" not in joined_user_prompts
    assert "SHOULD_NOT_APPEAR_BREED" not in joined_user_prompts


def test_generate_user_query_creates_one_query_per_condition(monkeypatch):
    captured: dict = {}

    def _fake_init_chat_model(*, model: str, temperature: float):
        return _FakeLLM(captured)

    monkeypatch.setattr(
        generate_user_query_module, "init_chat_model", _fake_init_chat_model
    )

    state = VetAgentState(
        health_condition=HealthCondition(
            frequent_illness_area="피부, 귀",
            disease_surgery_history="이력 없음",
        ),
        diseases=[
            DiseaseInfo(
                name="슬개골 탈구", incidence_rate="높음", onset_period="노령기"
            ),
            DiseaseInfo(name="피부염", incidence_rate="중간", onset_period="성견기"),
            DiseaseInfo(name="외이염", incidence_rate="중간", onset_period="성견기"),
        ],
    )

    result = generate_user_query_module.generate_user_query(state)

    assert len(result["query_texts"]) == 5
    assert len(captured["messages_list"]) == 5
    assert result["user_query"].count("테스트 질의 - ") == 5


@pytest.mark.integration
def test_generate_user_query_with_real_llm_case1_no_health_no_diseases():
    state = VetAgentState()

    with pytest.raises(ValueError, match="invalid VetAgentState !"):
        generate_user_query_module.generate_user_query(state)


@pytest.mark.integration
@pytest.mark.skipif(
    (os.getenv("RUN_REAL_LLM_TESTS") != "1") or (not os.getenv("UPSTAGE_API_KEY")),
    reason="Set RUN_REAL_LLM_TESTS=1 and UPSTAGE_API_KEY to run real LLM integration tests.",
)
@pytest.mark.parametrize(
    "state",
    [
        VetAgentState(
            health_condition=HealthCondition(
                frequent_illness_area="귀",
                disease_surgery_history="질병/수술 이력 없음",
            ),
            diseases=[],
        ),
        VetAgentState(
            health_condition=None,
            diseases=[
                DiseaseInfo(
                    name="슬개골 탈구",
                    incidence_rate="높음",
                    onset_period="노령기",
                )
            ],
        ),
        VetAgentState(
            health_condition=HealthCondition(
                frequent_illness_area="피부",
                disease_surgery_history="피부염 치료 이력",
            ),
            diseases=[
                DiseaseInfo(
                    name="피부염",
                    incidence_rate="중간",
                    onset_period="성견기",
                )
            ],
        ),
    ],
    ids=[
        "case2_health_only",
        "case3_diseases_only",
        "case4_health_and_diseases",
    ],
)
def test_generate_user_query_with_real_llm_cases_2_to_4(state: VetAgentState):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    test_project = os.getenv("LANGSMITH_TEST_PROJECT")
    if test_project:
        os.environ["LANGSMITH_PROJECT"] = test_project

    result = generate_user_query_module.generate_user_query(state)

    assert "user_query" in result
    assert isinstance(result["user_query"], str)
    assert result["user_query"].strip() != ""
