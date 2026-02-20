from typing import List, Dict, Any, Optional
from enum import Enum
from app.agents.rag_agent.rag_graph import RagGraphState
from app.agents.rag_agent.state.rag_state import RagState
from app.agents.vet_agent.state.vet_state import VetAgentState
from langchain_core.documents import Document
from pydantic import BaseModel, Field


# ==========================================
# 1. [Input] 팀원이 정의한 데이터 모델 (그대로 복사)
# ==========================================
class Gender(str, Enum):
    male = "male"
    female = "female"


class DiseaseInfo(BaseModel):
    name: str = Field(description="질병명")
    incidence_rate: str = Field(description="발병률")
    onset_period: str = Field(description="발병시기")


class VetAgentMockState(BaseModel):
    """유저 정보 + 수의사 진단 내용이 합쳐진 상태"""

    species: str = Field(description="종")
    breed: str = Field(description="품종")
    age: int = Field(description="나이")
    gender: Gender = Field(description="성별")
    weight: int = Field(description="체중(kg)")
    diseases: list[DiseaseInfo] = Field(default_factory=list, description="질병 목록")


# ==========================================
# 2. [Output] 검증 결과 모델 (우리가 정의한 것)
# ==========================================
class ValidatedPolicy(BaseModel):
    product_name: str = Field(description="상품명")
    suitability_score: int = Field(description="적합성 점수 (1~100)")
    reason: str = Field(description="이 상품이 선정된 이유")


class ValidationResult(BaseModel):
    selected_policies: List[ValidatedPolicy] = Field(description="상위 3개 보험 상품")
    review_summary: str = Field(description="전체 검토 요약")


# ==========================================
# 3. [Graph State] 통합 상태 정의
# ==========================================
class JudgeAgentState(VetAgentState):

    # RAG 결과
    retrieved_documents: List[Document] = Field(default_factory=list)

    # 검증 결과
    validation_result: Dict[str, Any] = Field(default_factory=dict)

    # [Composer Output]
    final_message: str = ""
