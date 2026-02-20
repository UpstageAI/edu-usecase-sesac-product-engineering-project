"""
LLM-as-a-Judge 평가 파이프라인의 Pydantic 데이터 모델 정의.

평가 흐름 전체에서 사용되는 데이터 구조를 정의합니다:
  1) EvaluationTestCase : 평가할 1개의 문제 (유저상태 + 질병 1개 + 약관 1개)
  2) JudgePrediction   : 기존 JudgeAgent의 판단 결과
  3) EvaluatorGroundTruth : 정답지 역할의 평가 LLM 판단 결과
  4) EvaluationRecord  : 위 세 가지를 합친 최종 평가 기록 1건
"""

from pydantic import BaseModel, Field


# ==========================================
# 1. 평가 입력: 1개의 테스트 케이스
# ==========================================
class EvaluationTestCase(BaseModel):
    """[1질병 x 1약관] 조합으로 구성된 평가 문제 1건.

    Vet Agent가 추출한 질병 하나와, RAG가 검색한 약관 텍스트 하나를
    묶어서 보장 여부를 판단하는 단위입니다.
    """

    file_name: str = Field(description="원본 YAML 파일명 (추적용)")
    species: str = Field(description="종 (강아지/고양이)")
    breed: str = Field(description="견/묘종")
    age: int = Field(description="나이")
    disease_surgery_history: str = Field(
        default="없음", description="기저질환/수술 이력 (health_condition에서 추출)"
    )
    disease_name: str = Field(description="평가 대상 질병명 (Vet Agent가 추출한 질병)")
    policy_text: str = Field(description="평가 대상 약관 원문 텍스트 (RAG 검색 결과)")


# ==========================================
# 2. JudgeAgent의 판단 결과 (예측값)
# ==========================================
class JudgePrediction(BaseModel):
    """기존 JudgeAgent 로직이 내린 [1질병 x 1약관] 보장 여부 판단.

    기존 validator_node의 프롬프트를 단건 평가용으로 변형하여 사용합니다.
    """

    is_covered: bool = Field(description="이 질병이 해당 약관으로 보장되는지 여부")
    reason: str = Field(description="판단 근거 (약관 조항 인용 포함)")


# ==========================================
# 3. Evaluator LLM의 판단 결과 (정답값)
# ==========================================
class EvaluatorGroundTruth(BaseModel):
    """정답지 역할의 강력한 평가 LLM이 내린 판단 결과.

    solar-pro2 모델을 최대한 엄격한 보험 심사 전문가 프롬프트로 호출하여
    JudgeAgent보다 더 정확한 '정답'을 생성합니다.
    """

    is_covered: bool = Field(description="이 질병이 해당 약관으로 보장되는지 여부")
    reason: str = Field(description="판단 근거 (약관의 구체적 문구 인용)")


# ==========================================
# 4. 최종 평가 기록 (1건)
# ==========================================
class EvaluationRecord(BaseModel):
    """테스트 케이스 + Judge 예측 + Evaluator 정답을 합친 최종 기록.

    이 기록이 Confusion Matrix 계산과 CSV 저장의 기본 단위가 됩니다.
    label 필드는 TP/TN/FP/FN 중 하나입니다.
    """

    test_case: EvaluationTestCase
    judge_prediction: JudgePrediction
    evaluator_ground_truth: EvaluatorGroundTruth
    label: str = Field(description="혼동 행렬 라벨 (TP, TN, FP, FN)")
