import difflib
from app.agents.vet_agent.state.vet_state import VetAgentState
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from ..state import JudgeAgentState, ValidationResult

from rich import print as rprint

# .env 파일 로드 (API Key 때문에 필수)
load_dotenv()


# insurer_code -> 보험사 한글명 매핑 (document_parser page_splitter 참고)
_INSURER_CODE_TO_NAME: dict[str, str] = {
    "samsung": "삼성화재",
    "meritz": "메리츠",
    "kb": "KB손해보험",
}


def _collect_actual_product_names(docs: list) -> set[str]:
    """retrieved_documents에서 metadata.doc.product_name을 수집하여 정확한 상품명 집합 반환."""
    names: set[str] = set()
    for doc in docs:
        doc_meta = (doc.metadata or {}).get("doc") or {}
        pn = doc_meta.get("product_name") if isinstance(doc_meta, dict) else None
        if pn and isinstance(pn, str):
            names.add(pn.strip())
    return names


def _build_product_to_insurer(docs: list) -> dict[str, str]:
    """product_name -> 보험사 한글명 매핑. doc.metadata.doc에서 추출."""
    mapping: dict[str, str] = {}
    for doc in docs:
        doc_meta = (doc.metadata or {}).get("doc") or {}
        if not isinstance(doc_meta, dict):
            continue
        pn = doc_meta.get("product_name")
        code = doc_meta.get("insurer_code")
        if pn and isinstance(pn, str) and code and isinstance(code, str):
            name = _INSURER_CODE_TO_NAME.get(code.lower(), code)
            mapping[pn.strip()] = name
    return mapping


def _correct_product_name(llm_product_name: str, actual_names: set[str]) -> str:
    """
    LLM이 반환한 상품명을 actual_names와 매칭하여 정확한 상품명으로 보정.
    오타 방지(예: 페퍼민트 → 펫퍼민트)를 위해 difflib로 유사도 매칭.
    """
    if not llm_product_name or not actual_names:
        return llm_product_name
    llm_name = llm_product_name.strip()
    if llm_name in actual_names:
        return llm_name
    matches = difflib.get_close_matches(llm_name, list(actual_names), n=1, cutoff=0.85)
    return matches[0] if matches else llm_name


def _deduplicate_policies(policies: list[dict]) -> list[dict]:
    """
    동일 product_name 중복 제거. 같은 상품이 2회 이상 나오면 가장 높은 suitability_score 것만 유지.
    """
    seen: dict[str, dict] = {}
    for p in policies:
        name = (p.get("product_name") or "").strip()
        if not name:
            continue
        score = p.get("suitability_score", 0)
        if name not in seen or seen[name].get("suitability_score", 0) < score:
            seen[name] = p
    return list(seen.values())


# ==========================================
# 검증 노드 핵심 로직
# ==========================================
def validator_node(state: JudgeAgentState):

    # 1. 데이터 꺼내기
    vet_field_keys = VetAgentState.model_fields.keys()
    vet_data = state.model_dump(include=vet_field_keys)
    docs = state.retrieved_documents

    # 2. Documents 객체들을 LLM이 읽을 수 있는 문자열로 변환
    # - document_parser에서 적재한 청크는 metadata.doc.product_name 에 실제 상품명이 있음
    # - (상품명: ...)을 문맥에 포함시켜 LLM이 selected_policies.product_name에 그대로 쓰도록 유도
    # - [추가] RAG evaluation 메타데이터(total_score, reason, judgement)를 포함하여
    #   Judge가 "보장 가능한 문서"를 구분하고, 일반약관 vs 특약 구분 판단에 활용하도록 함
    rag_context = ""
    for idx, doc in enumerate(docs):
        doc_meta = (doc.metadata or {}).get("doc") or {}
        product_name = (
            doc_meta.get("product_name") if isinstance(doc_meta, dict) else None
        ) or "상품명 미표기"

        # RAG relevance scoring 결과 (summary_multi, sparse_scoring에서 추가됨)
        evaluation = (doc.metadata or {}).get("evaluation") or {}
        eval_total_score = evaluation.get("total_score", "N/A")
        eval_judgement = evaluation.get("judgement", "N/A")
        eval_reason = evaluation.get("reason", "")

        # 각 문서에 RAG 평가 정보를 함께 전달하여 Judge가 "보장 가능" 문서를 인지하도록 함
        eval_block = (
            f"[RAG 평가: total_score={eval_total_score}, judgement={eval_judgement}]\n"
            f"reason: {eval_reason}\n\n"
            if eval_reason
            else ""
        )

        rag_context += (
            f"\n[약관 {idx+1}] (상품명: {product_name})\n"
            f"{eval_block}"
            f"{doc.page_content}\n"
        )

    # 3. LLM 설정
    llm = ChatUpstage(model="solar-pro2", temperature=0)
    structured_llm = llm.with_structured_output(ValidationResult)

    # 4. 프롬프트 작성 (User + Vet 정보가 하나로 합쳐짐)
    # - 일반약관 vs 특약(부가약관) 구분 추가: 일반약관 면책이어도 특약에서 보상 가능하면 반영
    # - suitability_score 산정 기준 명시: 특약 보장 가능 시 과도한 감점 지양
    # - reason/review_summary에 "특약에서 보상 가능" 구체적 언급 요구
    system_prompt = """당신은 보험 약관 심사 전문가입니다.
    제공된 [Vet Analysis]와 [Insurance Policies]를 비교하여 가입 적합성을 판단하세요.
    
    # 심사 기준 (엄격 준수)

    1. **일반약관 vs 특약(부가약관) 구분 (매우 중요)**
        - 일반 보통약관에서는 특정 질병이 '면책'이라도, **특약(부가약관)**에서 보상 가능한 경우가 있습니다.
        - 예: 슬개골 탈구 → 일반약관 면책이어도, "슬관절/고관절 치료비 특약"에서 보상 가능할 수 있음.
        - [RAG 평가]의 reason에 "보장 범위", "청구 조건", "슬관절", "치료비" 등 보상 관련 내용이 있으면
          해당 문서는 **특약·보장 가능성**을 담고 있으므로 suitability_score를 과도하게 낮추지 마세요.
        - reason과 review_summary에는 반드시 "일반약관에서는 면책이나, 특약(슬관절 치료비 등)에서 보상 가능" 같은 구체적 내용을 포함하세요.

    2. **질병 면책 확인**
        - 수의사가 진단한 질병(diseases)이 일반약관의 '보상하지 않는 손해(면책 사항)'에 포함되는지 확인하세요.
        - 단, 특약·부가약관에서 해당 질병에 대한 보장 조항이 있으면 면책만으로 거절 판단하지 마세요.

    3. **suitability_score 산정 기준 (1~100)**
        - 특약·부가약관으로 해당 질병 보상 가능: 50~80점 (보장 가능성에 따라 상향)
        - 일반약관만 해당하고 일부 보장: 30~50점
        - 대부분 면책이고 보장 가능성 거의 없음: 10~30점
        - RAG 평가가 total_score 80 이상인 문서에 보장/청구 조건이 명시되어 있으면,
          해당 상품의 suitability_score를 낮게 주지 마세요.

    4. **나이 제한 (만 19세 기준)**
        - 이 보험은 **만 19세까지** 재가입이 가능합니다.
        - 반려동물의 나이가 **19세를 초과하는 경우에만** 나이를 거절 사유로 언급하세요.
        - 19세 이하(예: 10세, 15세 등)라면 나이 때문에 가입이 어렵다는 말을 절대 하지 마세요.

    5. **축종(Species) 확인**
        - 대상이 '개(Dog)'인지 '고양이(Cat)'인지만 확인하여 약관을 적용하세요.

    6. **selected_policies의 product_name (필수)**
        - 각 약관에는 "(상품명: ...)" 형태로 실제 보험 상품명이 적혀 있습니다.
        - selected_policies에 넣을 때 product_name 필드에는 반드시 위 [Insurance Policies]에 적힌 **(상품명: ...)** 값을 한 글자도 틀리지 않고 그대로 사용하세요. "약관 6", "약관 7" 같은 번호만 적지 마세요.
        - **중요: selected_policies에는 서로 다른 상품만 선정하세요. 같은 상품명이 2회 이상 나오면 안 됩니다.**
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
        === [Vet Analysis (유저 정보 + 진단 결과)] ===
        {vet_data}
        
        === [Insurance Policies (약관 검색 결과)] ===
        {rag_context}
        """,
            ),
        ]
    )

    # 5. 실행
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "vet_data": str(vet_data),
            "rag_context": rag_context,
        }
    )

    # 6. 후처리: 중복 상품 제거 + 상품명 오타 보정 + 보험사명 추가
    actual_names = _collect_actual_product_names(docs)
    product_to_insurer = _build_product_to_insurer(docs)
    policies = result.model_dump().get("selected_policies", [])

    # 6-1. 상품명 보정 (retrieved_documents 기반 정확한 상품명으로 교체)
    for p in policies:
        p["product_name"] = _correct_product_name(
            p.get("product_name", ""), actual_names
        )

    # 6-2. 보험사명 추가 (product_name 기준으로 insurer_name 설정)
    for p in policies:
        p["insurer_name"] = product_to_insurer.get(
            (p.get("product_name") or "").strip(), ""
        )

    # 6-3. 동일 상품 중복 제거 (같은 product_name은 가장 높은 점수 것만 유지)
    policies = _deduplicate_policies(policies)

    # 6-4. suitability_score 내림차순 정렬 (상위 노출 순서 유지)
    policies.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)

    return {
        "validation_result": {
            "selected_policies": policies,
            "review_summary": result.model_dump().get("review_summary", ""),
        }
    }
