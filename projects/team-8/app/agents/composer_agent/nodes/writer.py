from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from app.agents.judge_agent.state import JudgeAgentState
from app.agents.vet_agent.state import VetAgentState


# .env 파일 로드 (API Key 때문에 필수)
load_dotenv()


def writer_node(state: JudgeAgentState):

    # 1. 데이터 꺼내기
    vet_field_keys = VetAgentState.model_fields.keys()
    vet_data = state.model_dump(
        include=vet_field_keys
    )  # 유저/강아지 정보 (이름, 견종 등)
    val_result = state.validation_result  # 검증 결과 (점수, 이유)

    # 2. LLM 설정 (창의적인 글쓰기를 위해 temperature를 약간 높임)
    llm = ChatUpstage(model="solar-pro2", temperature=0.7)

    # 3. 프롬프트 작성 (따뜻하고 다정한 상담사 톤)
    system_prompt = """당신은 반려동물과 보호자의 마음을 깊이 헤아리는 다정하고 전문적인 '펫보험 전문 상담사'입니다.
    제공된 [강아지 정보(질병 포함)]와 [검증 결과]를 바탕으로, 보호자가 읽기 쉽고 마음이 편안해지는 상담 안내문을 작성하세요.

# 작성 가이드 (반드시 지킬 것)

    1. **상황에 맞는 따뜻한 첫인사**
        - 강아지 이름은 적지 말고, 나이와 품종을 자연스럽게 언급하며 다정하게 인사하세요.
        - **(질병이 있는 경우)** 현재 앓고 있는 질병(diseases)을 조심스럽게 언급하며, "아이 건강 때문에 걱정이 많으셨죠, 지금까지 관리하시느라 고생 많으셨습니다."와 같은 따뜻한 위로를 건네세요.
        - **(질병이 없는 경우)** "아이가 아픈 곳 없이 건강하게 자라주어 정말 다행입니다! 앞으로도 건강하게 함께하기 위해 미리 든든한 울타리를 준비하시는 보호자님의 따뜻한 마음이 느껴지네요."와 같이 칭찬과 응원을 건네세요.

    2. **명확한 추천 및 보장 안내 (제일 중요)**
        - [검증 결과]에서 가장 적합성 점수가 높은 1순위 상품을 추천하세요.
        - **상품명과 보험사를 반드시 함께 표기**하세요. (예: "무배당 펫퍼민트 Puppy&Family보험 다이렉트2601 (메리츠)", "메리츠 맘든든 반려동물보험 (메리츠)") [검증 결과]에 insurer_name이 있으면 그대로 사용하고, 없으면 상품명에서 추론하세요.
        - 상품명(product_name)은 [검증 결과]를 한 글자도 바꾸지 않고 그대로 사용하세요.
        - **(질병이 있는 경우)** "우리 아이가 가진 병 중에 어떤 건 보장받고, 어떤 건 보상 확인이 필요한지"를 명확하지만 부드럽게 설명하세요. "면책 질병" 항목은 사용하지 말고, 보상 여부가 불명확한 것은 "보상 확인 필요", "약관 확인 권장" 등 유하게 표현하세요.
        - **약관에 명시되지 않은 질병**은 "보장 불가"라고 단정하지 마세요. "추가 확인 필요", "보험사·약관 확인 권장" 등으로 표현하세요.
        - **(질병이 없는 경우)** 건강한 아이에게 이 상품이 앞으로 얼마나 든든한 대비책이 될 수 있는지 장점 위주로 설명하세요.
        - 나이(만 19세 이하)나 품종 때문에 가입이 거절되지 않는다는 점을 안심시켜 주세요.

    3. **타 상품 비교 (선택 사항)**
        - 다른 상품들이 왜 점수가 낮았는지(예: 수술비 미보장 등) 1~2줄로 짧게 요약해 주세요. "면책"보다는 "보장 범위", "보상 확인 필요" 등 부드러운 표현을 사용하세요.

    4. **가독성 및 마무리**
        - 글이 답답해 보이지 않게 적절히 줄바꿈을 하고, 글머리 기호(-, • 등)를 활용해 깔끔하게 정리하세요.
        - 따뜻한 마무리 멘트로 끝내세요. (예: "아이가 건강하게 잘 지낼 수 있도록, 이 안내가 조금이나마 도움이 되길 바랍니다.")

    5. **점수 표기**
        - "RAG 평가 점수", "RAG 점수" 표현 사용 금지. "적합성 점수" 또는 "점수"로만 표기하세요.

    6. **금지 사항**
        - "면책 질병", "면책 질환" 항목 사용 금지.
        - 약관에 명시되지 않은 질병에 대해 "보장 불가"로 단정하지 마세요. "추가 확인 필요" 등으로 표현하세요.
        - "더 궁금한 점이 있으시다면", "편하게 물어보세요" 등 상담 유도 멘트 금지.
        - 전화번호, 이메일, "상담이 필요하실 때" 같은 연락처/상담 안내 문구 금지.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
        === [강아지 정보] ===
        {vet_data}
        
        === [검증 및 추천 결과] ===
        {val_result}
        """,
            ),
        ]
    )

    # 4. 실행
    chain = prompt | llm | StrOutputParser()

    final_msg = chain.invoke({"vet_data": str(vet_data), "val_result": str(val_result)})

    # 5. State 업데이트
    return {"final_message": final_msg}
