"""JudgeAgent 단건 판단 노드: [1질병 x 1약관]에 대해 보장 여부 판단 및 라벨 계산."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

from app.evaluation.state import (
    EvaluationTestCase,
    EvaluatorGroundTruth,
    JudgePrediction,
)

# 실제 validator_node 프롬프트 기준을 단건 [1질병 x 1약관] 판단용으로 변형
JUDGE_SYSTEM_PROMPT = """\
당신은 보험 약관 심사 전문가입니다.
주어진 반려동물 정보와 질병명, 약관 텍스트를 비교하여 해당 질병의 보장 여부를 판단하세요.

# 심사 기준 (엄격 준수, 순서대로 적용)

1. **텍스트 관련성 확인 (최우선 · RAG 예외 처리)**
   - 제공된 약관 텍스트가 질병 보장/면책과 **아예 무관한 내용**이면 즉시 보장 불가(False)로 판단하세요.
   - 무관한 내용 예시: 계약 변경 조건, 계약 해지, 보험금 계산 공식, 청구 서류 안내, 주소 변경 등
   - 해당 텍스트만으로는 보장 여부를 알 수 없으므로, 추측으로 보장된다고 판단하면 안 됩니다.

2. **일반약관 vs 특약(부가약관) 구분 (매우 중요)**
   - 일반 보통약관에서는 특정 질병이 '면책'이라도, **특약(부가약관)**에서 보상 가능한 경우가 있습니다.
   - 예: 슬개골 탈구 → 일반약관 면책이어도, "슬관절/고관절 치료비 특약"에서 보상 가능.
   - 약관 텍스트에 "보장 범위", "청구 조건", "치료비 특약" 등 보상 관련 내용이 있으면, 해당 텍스트는 **특약·보장 가능성**을 담고 있으므로 면책만으로 False 처리하지 마세요.

3. **질병 면책 확인**
   - 해당 질병이 약관의 '보상하지 않는 손해(면책 사항)'에 명시적으로 포함되어 있으면 보장 불가(False)입니다.
   - 질병명이 정확히 일치하지 않더라도, 약관에 명시된 '유사 질병' 또는 '상위/하위 카테고리(관절 질환, 피부 질환 등)'에 속하면 면책(False)입니다.
   - **단, 해당 약관이 특약·부가약관으로서 그 질병에 대한 보장 조항을 포함하고 있다면 면책이 아닌 보장(True)으로 판단하세요.**

4. **기왕증(기존 질병) 확인**
   - 기저질환/수술 이력에 해당 질병이 이미 존재하고, 약관에 기왕증 면책 조항이 있으면 보장 불가(False)입니다.

5. **나이 제한 및 대기 기간**
   - 이 보험은 **만 19세까지** 재가입 가능합니다. 반려동물 나이가 19세 이하이면 나이를 거절 사유로 사용하지 마세요.
   - 반려동물 나이가 19세를 **초과**하는 경우에만 나이 제한으로 보장 불가(False)입니다.
   - '가입 후 90일 이내 발생 시 면책' 등 대기기간 조항이 있고, 발병 시점이 대기기간 내일 가능성이 높으면 보장 불가(False)입니다.

6. **축종(Species) 확인**
   - 대상이 '개(Dog)'인지 '고양이(Cat)'인지 확인하여 해당 축종용 약관인지 확인하세요.

7. **보장 추정의 원칙**
   - 위 1~6에 해당하지 않으면, 보험은 면책 항목을 나열하는 구조(Negative List)이므로 해당 질병이 면책에 적혀있지 않다면 보장 가능(True)으로 간주하세요.

# 출력 규칙
- is_covered: 보장 가능하면 True, 보장 불가/면책/판단 불가면 False
- reason: 판단 근거를 1~2문장(80자 이내)으로 간략히 작성하세요.
"""

JUDGE_HUMAN_PROMPT = """\
다음 정보를 바탕으로 보장 여부를 판단해 주세요.

=== [반려동물 정보] ===
- 종: {species}
- 품종: {breed}
- 나이: {age}세
- 기저질환/수술 이력: {disease_surgery_history}

=== [평가 대상 질병] ===
{disease_name}

=== [약관 텍스트] ===
{policy_text}
"""


async def judge_predict(test_case: EvaluationTestCase) -> JudgePrediction:
    """기존 JudgeAgent 로직을 단건 [1질병 x 1약관]에 대해 실행합니다."""
    llm = ChatUpstage(model="solar-pro2", temperature=0)
    structured_llm = llm.with_structured_output(JudgePrediction)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", JUDGE_HUMAN_PROMPT),
        ]
    )

    chain = prompt | structured_llm
    result: JudgePrediction = await chain.ainvoke(
        {
            "species": test_case.species,
            "breed": test_case.breed,
            "age": test_case.age,
            "disease_surgery_history": test_case.disease_surgery_history or "없음",
            "disease_name": test_case.disease_name,
            "policy_text": test_case.policy_text,
        }
    )

    return result


def compute_label(judge: JudgePrediction, evaluator: EvaluatorGroundTruth) -> str:
    """Judge 예측값과 Evaluator 정답값을 비교하여 TP/TN/FP/FN을 반환합니다.

    Positive = 보장됨(is_covered=True), Negative = 보장 안 됨(is_covered=False)
    """
    if judge.is_covered and evaluator.is_covered:
        return "TP"
    if not judge.is_covered and not evaluator.is_covered:
        return "TN"
    if judge.is_covered and not evaluator.is_covered:
        return "FP"
    return "FN"
