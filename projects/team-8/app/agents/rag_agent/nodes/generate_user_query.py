from langchain.chat_models import init_chat_model

from rich import print as rprint

from app.agents.rag_agent.state.rag_state import RagState, GenerateQueryTextOutput
from app.agents.vet_agent.state import VetAgentState


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


def generate_user_query(state: VetAgentState) -> RagState:
    rprint("💊건강 상태:", state.health_condition)
    rprint("💊질병:", state.diseases)

    if not state or (not state.health_condition and not state.diseases):
        raise ValueError("invalid VetAgentState !")

    frequent_illness_area_raw = (
        (state.health_condition.frequent_illness_area or "").strip()
        if state.health_condition
        else ""
    )
    frequent_illness_area = (
        frequent_illness_area_raw
        if frequent_illness_area_raw and _is_meaningful_value(frequent_illness_area_raw)
        else ""
    )
    disease_surgery_history_raw = (
        (state.health_condition.disease_surgery_history or "").strip()
        if state.health_condition
        else ""
    )
    disease_surgery_history = (
        disease_surgery_history_raw
        if disease_surgery_history_raw
        and _is_meaningful_value(disease_surgery_history_raw)
        else ""
    )
    disease_names = [
        (d.name or "").strip()
        for d in (state.diseases or [])
        if d and (d.name or "").strip() and _is_meaningful_value((d.name or "").strip())
    ]

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
"""

    reference_lines: list[str] = []
    if frequent_illness_area:
        reference_lines.append(f"- 자주 아픈 부위: {frequent_illness_area}")
    if disease_surgery_history:
        reference_lines.append(f"- 수술 이력: {disease_surgery_history}")
    if disease_names:
        reference_lines.append(f"- 예상 질병: {', '.join(disease_names)}")

    reference_block = (
        "\n".join(reference_lines) if reference_lines else "- 추가 건강 정보 없음"
    )

    user_prompt = f"""
약관 검색용 query_text 1개를 생성해줘.

[참고 정보]
{reference_block}

[생성 규칙]
1) 참고 정보를 반드시 포함하고, 해당 정보와 직접적으로 연관된 약관 용어를 함께 넣는다.
2) 특히 '보장(coverage)'과 관련된 용어를 자연스럽게 추가한다.
3) "없음", "해당 없음", "이력 없음", "특이사항 없음" 같은 의미 없는 값은 사용하지 않는다.
4) 출력은 질문형이 아닌, 검색 인텐트가 분명한 단일 한국어 문장으로 작성한다.
""".strip()
    # rprint("👉🏻생성된 user_prompt:", user_prompt)

    MODEL = "solar-pro2"
    llm = init_chat_model(model=MODEL, temperature=0.0)
    structured_llm = llm.with_structured_output(GenerateQueryTextOutput)
    llm_response: GenerateQueryTextOutput = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    rprint("❓생성된 query text:", llm_response.query_text)

    return {
        # "user_query": llm_response.query_text,
        "query_texts": [llm_response.query_text],
    }
