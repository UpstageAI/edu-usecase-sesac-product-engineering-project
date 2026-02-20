from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid # For default_factory in RiskReport analysis_id

# --- Config Constants (from config.py) ---
SIMILARITY_THRESHOLD = 0.87
RISK_AGENT_TRIGGER_WORDS = [ # Not directly used in state, but useful context if needed
    "실수", "클레임", "지연", "문제", "리스크", "invoice", "선적",
    "hs code", "payment", "계약", "괜찮을까", "확인 부탁",
]
AGENT_PERSONA = { # Not directly used in state, but useful context if needed
    "tone": "담백하고 직설적", "emotional_expression": "금지", "exaggeration": "금지",
    "feedback_style": "실제 회사 상사 피드백 톤 유지",
    "judgment_criteria": ["회사 기준", "실무 기준", "실제 발생 가능한 리스크", "내부 보고 기준"],
    "response_style": "친절한 설명형이 아니라 실무 피드백 형식",
    "always_include": ["무엇이 문제인지", "왜 문제인지", "실제 발생 가능한 상황", "지금 해야 할 행동"],
    "never_include": ["과도한 공감", "감정 위로", "불필요한 장문 설명", "추상적 조언"],
}
RISK_EVALUATION_ITEMS = { # Used for risk_engine logic
    "financial_loss": {"name": "재정적 손실", "description": "프로젝트로 인해 발생할 수 있는 직접적/간접적 금전적 손실"},
    "delay": {"name": "일정 지연", "description": "프로젝트 완료 또는 목표 달성 일정 지연 가능성"},
    "relationship_risk": {"name": "관계 리스크", "description": "내부 팀, 외부 파트너, 고객 등 이해관계자와의 관계 손상 가능성"},
    "compliance_risk": {"name": "규제/법률 준수 리스크", "description": "법규, 회사 정책, 국제 규정 등 준수 여부 및 위반 시 발생할 수 있는 리스크"},
    "internal_blame_risk": {"name": "내부 책임/비난 리스크", "description": "프로젝트 실패 시 내부적으로 발생할 수 있는 책임 추궁 및 비난의 강도"},
}
RISK_LEVEL_THRESHOLDS = { # Used for risk_engine logic
    "critical": 15, "high": 10, "medium": 5, "low": 1,
}
RAG_DATASETS = [
    "claims", "mistakes", "emails", "country_rules", 
    "BL_CHECK", "CUSTOMS", "SHIPPING", "PAYMENT", "CONTRACT", 
    "EMAIL", "NEGOTIATION", "QUALITY", "LOGISTICS", "INSURANCE", 
    "COMMUNICATION", "risk_knowledge"
] # Used for RAG logic


# --- Schemas (from schemas.py) ---
class RiskManagingAgentInput(BaseModel):
    user_input: str = Field(..., description="사용자의 원본 질의")
    context: Optional[Dict[str, Any]] = Field(None, description="추가적인 비즈니스 맥락 정보")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="이전 대화 내역")

class RiskFactor(BaseModel):
    name: str = Field(..., description="평가 항목명 (예: 재정적 손실, 일정 지연)")
    impact: int = Field(..., ge=1, le=5, description="영향도 (1-5)")
    likelihood: int = Field(..., ge=1, le=5, description="발생 가능성 (1-5)")
    risk_score: int = Field(..., ge=1, le=25, description="리스크 점수 (Impact * Likelihood)")
    risk_level: str = Field(..., description="리스크 수준 (low/medium/high/critical)")
    reasoning: str = Field(..., description="해당 점수 및 수준이 나온 구체적인 근거")
    mitigation_suggestions: List[str] = Field([], description="리스크 완화 방안 제안") # Changed default to empty list

class RiskScoring(BaseModel):
    overall_risk_level: str = Field(..., description="전반적인 리스크 수준")
    risk_factors: List[RiskFactor] = Field(..., description="항목별 리스크 요인 및 평가")
    overall_assessment: str = Field(..., description="전반적인 리스크 평가 요약")

class LossSimulation(BaseModel):
    quantitative: Optional[str] = Field(None, description="정량적 손실 시뮬레이션 (예: $10,000 ~ $50,000)")
    qualitative: str = Field(..., description="정성적 손실 시뮬레이션 설명")

class ControlGapAnalysis(BaseModel):
    identified_gaps: List[str] = Field([], description="현재 리스크 관리 체계의 허점") # Changed default
    recommendations: List[str] = Field([], description="허점 보완을 위한 제안") # Changed default

class PreventionStrategy(BaseModel):
    short_term: List[str] = Field([], description="단기 예방 전략") # Changed default
    long_term: List[str] = Field([], description="장기 예방 전략") # Changed default

class ReportRiskFactor(BaseModel):
    name_kr: str = Field(..., description="리스크 요인 한국어 이름")
    impact: int = Field(..., ge=1, le=5, description="영향도 (1-5)")
    likelihood: int = Field(..., ge=1, le=5, description="발생 가능성 (1-5)")
    score: int = Field(..., ge=1, le=25, description="리스크 점수 (Impact * Likelihood)")

class RiskReport(BaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="리스크 분석 고유 ID")
    input_summary: str = Field(..., description="사용자 질의 및 컨텍스트 요약")
    risk_factors: Dict[str, ReportRiskFactor] = Field(..., description="식별된 주요 리스크 요인들을 {name: ReportRiskFactor} 형태로 요약")
    risk_scoring: RiskScoring = Field(..., description="전반적인 리스크 점수화 및 수준")
    loss_simulation: LossSimulation = Field(..., description="예상 손실 시뮬레이션")
    control_gap_analysis: ControlGapAnalysis = Field(..., description="리스크 관리 체계 허점 분석")
    prevention_strategy: PreventionStrategy = Field(..., description="예방 및 대응 전략")
    similar_cases: List[Dict[str, Any]] = Field([], description="RAG를 통해 조회된 유사 사례") # Changed default
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="분석 결과에 대한 에이전트의 신뢰도 (0.0 ~ 1.0)")
    evidence_sources: List[str] = Field([], description="분석에 활용된 근거 문서 출처") # Changed default

class RiskManagingAgentResponse(BaseModel):
    response: str = Field(..., description="에이전트의 최종 응답 (JSON 문자열)")
    agent_type: str = Field("riskmanaging", description="에이전트 유형")
    metadata: Dict[str, Any] = Field({}, description="에이전트 실행 관련 추가 메타데이터")


# --- Define the State for the RiskManaging Graph ---
class RiskManagingGraphState(TypedDict):
    """
    Represents the state of the RiskManaging sub-graph.
    Updated to match actual node function usage.
    """
    # Core inputs
    current_user_input: str  # Current user message
    conversation_history: List[Dict[str, str]]  # Previous conversation turns
    
    # Trigger and similarity detection
    risk_trigger_detected: bool  # Whether trigger words were detected
    risk_similarity_score: float  # Similarity score from SimilarityEngine
    analysis_required: bool  # Whether risk analysis should proceed
    
    # Conversation management
    analysis_ready: bool  # Whether enough info collected for analysis
    analysis_in_progress: bool  # Whether analysis session is ongoing
    report_requested: bool  # Whether user has explicitly requested or approved report generation
    conversation_stage: str  # Current stage of conversation
    extracted_data: Dict[str, Any]  # Data extracted from conversation
    user_profile: Optional[Dict[str, Any]]  # User persona information
    
    # RAG results
    rag_documents: List[Dict[str, Any]]  # Retrieved documents from RAG
    
    # Risk evaluation
    risk_scoring: Optional[RiskScoring]  # Risk scoring results
    
    # Final report
    report_generated: Optional[RiskReport]  # Generated risk report
    
    # Output
    agent_response: str  # Final response to user
    error_message: Optional[str]  # Error message if any
