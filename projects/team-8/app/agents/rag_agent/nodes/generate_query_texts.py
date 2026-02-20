from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from rich import print as rprint

from app.agents.rag_agent.state.rag_state import RagState, GenerateQueryTextOutput

from app.agents.vet_agent.state import VetAgentState


class SplitConditionValuesOutput(BaseModel):
    values: list[str] = Field(default_factory=list)


def _split_condition_values(raw_text: str, splitter_llm: Any) -> list[str]:
    if not raw_text:
        return []

    prompt = f"""
아래 문장을 '자주 걸리는 질병/증상/신체부위' 단위로 분리해.

규칙:
1) 의미 단위(명사구)만 추출하고 중복은 제거해.
2) "없음/해당 없음/특이사항 없음" 같은 무의미 표현은 제외해.
3) 원문에 없는 내용을 추측해서 추가하지 마.

[입력 문장]
{raw_text}
""".strip()

    try:
        response: SplitConditionValuesOutput = splitter_llm.invoke(
            [{"role": "user", "content": prompt}]
        )
        values = [value.strip() for value in response.values if value and value.strip()]
        if values:
            return values
    except Exception:
        pass

    return [raw_text.strip()]


def _is_meaningful_value(value: str) -> bool:
    meaningless_tokens = {
        "없음",
        "해당 없음",
        "이력 없음",
        "특이사항 없음",
        "정상",
        "없다",
        "무",
        "none",
        "n/a",
    }
    normalized = value.strip().lower()
    return normalized not in meaningless_tokens


def _build_condition_items(
    state: VetAgentState, splitter_llm: Any
) -> list[tuple[str, str]]:
    frequent_illness_area = (
        state.health_condition.frequent_illness_area if state.health_condition else None
    )
    rprint("💊자주 아픈 부위:", frequent_illness_area)
    rprint("💊예상 질병:", [d.name for d in state.diseases])

    condition_items: list[tuple[str, str]] = []

    if state.health_condition and state.health_condition.frequent_illness_area:
        for area in _split_condition_values(
            state.health_condition.frequent_illness_area, splitter_llm
        ):
            if _is_meaningful_value(area):
                condition_items.append(("frequent_illness_area", area))

    if state.diseases:
        for disease in state.diseases:
            disease_name = (disease.name or "").strip()
            if disease_name and _is_meaningful_value(disease_name):
                condition_items.append(("disease_name", disease_name))

    # 입력 순서를 유지한 채 중복 제거
    unique_items: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in condition_items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    rprint("💊최종 건강 정보:", unique_items)

    return unique_items


def generate_query_texts(state: VetAgentState) -> RagState:
    # rprint(">>> generate_query_texts input state", state)

    if not state or (not state.health_condition and not state.diseases):
        raise ValueError("invalid VetAgentState !")

    system_prompt = """
역할:
너는 반려동물 보험 전문 보험 설계사 출신 CTO다.

목표:
보험 약관 청크를 dense retrieval로 잘 찾을 수 있는 고밀도 query_text를 생성한다.

작성 원칙:
1) 입력으로 제공된 건강 상태 정보만 사용한다.
2) query_text는 약관 본문에 자주 등장하는 용어 중심으로 구성한다.
3) 검색 노이즈를 줄이기 위해 감성적 표현/질문형/불필요한 수식어를 피하고, 핵심 키워드 밀도를 높인다.
4) 출력은 한국어 단일 문장 1개만 생성한다.
""".strip()

    MODEL = "solar-pro2"
    llm = init_chat_model(model=MODEL, temperature=0.0)
    split_values_llm = llm.with_structured_output(SplitConditionValuesOutput)
    structured_llm = llm.with_structured_output(GenerateQueryTextOutput)

    condition_items = _build_condition_items(state, split_values_llm)
    if not condition_items:
        raise ValueError("no meaningful health conditions to build query_text")

    query_texts: list[str] = []
    disease_surgery_history = (
        state.health_condition.disease_surgery_history if state.health_condition else ""
    )

    for condition_type, condition_value in condition_items:
        reference_lines: list[str] = []
        if disease_surgery_history and _is_meaningful_value(disease_surgery_history):
            reference_lines.append(
                f"- disease_surgery_history: {disease_surgery_history}"
            )
        if state.diseases:
            disease_names = [
                (disease.name or "").strip()
                for disease in state.diseases
                if (disease.name or "").strip()
                and _is_meaningful_value((disease.name or "").strip())
            ]
            if disease_names:
                reference_lines.append(f"- disease_names: {', '.join(disease_names)}")

        reference_block = (
            "\n".join(reference_lines) if reference_lines else "- 추가 참고 정보 없음"
        )

        user_prompt = f"""
다음 입력으로 약관 검색용 query_text 1개를 생성해줘.

[타깃 조건]
- condition_type: {condition_type}
- condition_value: {condition_value}

[참고 정보]
{reference_block}

[생성 규칙]
1) condition_value를 반드시 포함하고, 해당 조건과 직접적으로 연관된 약관 용어를 함께 넣는다.
2) 특히 '보장(coverage)'과 관련된 항목을 선택해 자연스럽게 반영한다.
3) "없음", "해당 없음", "이력 없음", "특이사항 없음" 같은 의미 없는 값은 사용하지 않는다.
4) 출력은 질문형이 아닌, 검색 인텐트가 분명한 단일 한국어 문장으로 작성한다.
""".strip()

        # rprint(">>> generated prompt", user_prompt)

        llm_response: GenerateQueryTextOutput = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        query_texts.append(llm_response.query_text)

    if not query_texts:
        raise ValueError("failed to generate query_text")

    return {"query_texts": query_texts}


# Unit tests (mocked LLM)
# uv run pytest -q app/agents/rag_agent/nodes/test_generate_user_query.py
# uv run pytest -q app/agents/rag_agent/nodes/test_generate_user_query.py::test_generate_user_query_uses_only_health_condition_and_diseases
#
# Integration tests (real LLM)
# RUN_REAL_LLM_TESTS=1 uv run pytest -q -m integration app/agents/rag_agent/nodes/test_generate_user_query.py
