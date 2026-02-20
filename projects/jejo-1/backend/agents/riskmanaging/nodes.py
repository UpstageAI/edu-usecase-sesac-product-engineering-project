# backend/agents/riskmanaging/nodes.py

from __future__ import annotations # Added as per instruction

# Standard imports
import os
import sys
import json
import uuid
import time
from typing import Dict, Any, List, Optional, cast, TypedDict
from backend.utils.json_utils import safe_json_parse
import openai
from openai import OpenAI
from langsmith import traceable
import numpy as np
from numpy.linalg import norm

# Ensure backend directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

# Local imports (minimal as most will be internal)
from backend.config import get_settings
from backend.utils.logger import get_logger
# RAG functionality now provided by tools.py
from backend.rag.embedder import get_embedding
from pydantic import BaseModel, Field

# Import RiskManagingGraphState explicitly and minimally
from backend.agents.riskmanaging.state import RiskManagingGraphState

# Import other constants and schemas from state.py separately
from backend.agents.riskmanaging.state import (
    SIMILARITY_THRESHOLD, RISK_AGENT_TRIGGER_WORDS, AGENT_PERSONA,
    RISK_EVALUATION_ITEMS, RISK_LEVEL_THRESHOLDS, RAG_DATASETS, RiskManagingAgentInput,
    RiskReport, RiskManagingAgentResponse, RiskScoring, LossSimulation,
    ControlGapAnalysis, PreventionStrategy, ReportRiskFactor # Added ReportRiskFactor
)
logger = get_logger(__name__)

# --- Prompt Loader ---
def _load_prompt(prompt_file_name: str) -> str:
    """Load prompt content from backend/prompts directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # from backend/agents/riskmanaging to project root
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    prompt_path = os.path.join(project_root, 'backend', 'prompts', prompt_file_name)
    
    if not os.path.exists(prompt_path):
        logger.warning("Prompt file not found: %s. Using default prompt.", prompt_path)
        return "당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다."
        
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.warning("Error loading prompt %s: %s", prompt_file_name, e)
        return "당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다."

# Load the system prompt from file
RISK_AGENT_SYSTEM_PROMPT = _load_prompt("riskmanaging_prompt.txt")

def build_user_instruction(user_profile: dict) -> str:
    if not user_profile:
        return "실무자가 이해할 수 있는 수준으로 설명하라."
        
    role = user_profile.get("role_level", "")
    style = user_profile.get("preferred_style", "")
    risk = user_profile.get("risk_tolerance", "")
    weak = ", ".join(user_profile.get("weak_topics", []))

    if role == "junior":
        tone = "교육용으로 쉽게 설명하고 단계별 체크리스트를 제공해라."
    elif role == "senior":
        tone = "관리자 보고 수준으로 간결하게 핵심 리스크와 즉시 조치를 제시해라."
    elif role == "sales":
        tone = "금전 손실과 협상 영향 중심으로 경고하라."
    else:
        tone = "실무자가 이해할 수 있는 수준으로 설명하라."

    if style == "blunt":
        tone += " 불필요한 설명 없이 단정적으로 말해라."
    elif style == "coaching":
        tone += " 왜 위험한지 이유를 포함해 코칭해라."
    elif style == "checklist":
        tone += " 반드시 실행 체크리스트 형태로 작성해라."
    elif style == "concise":
        tone += " 핵심 위주로 아주 짧게 대답해라."

    if risk == "low":
        tone += " 보수적으로 판단하고 위험을 강조해라."
    elif risk == "high":
        tone += " 과도한 경고는 줄이고 핵심만 전달해라."

    if weak:
        tone += f" 특히 {weak} 관련 개념을 보강 설명해라."

    return tone


CONVERSATION_ASSESSMENT_PROMPT = """
당신은 리스크 분석을 위한 정보를 수집하고 사용자와 협업하는 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.

[System Directive: LLM Interaction Mode — Mandatory]
1. 모든 출력은 반드시 사용자와 상호작용하는 대화형(Q&A) 구조로만 진행한다.
2. **자동 보고서 생성 금지**: 정보가 충분하더라도 사용자의 명시적 요청이나 승인 없이 최종 보고서를 출력하지 않는다.
3. **중간 추론 공유**: 정보가 어느 정도 모였다면, 지금까지 파악된 리스크의 핵심을 상사처럼 짧게 언급("중간 추론")하고, 보고서 작성을 원하는지 물어본다.
4. **프로세스 및 기준 문의 대응**: 사용자가 리스크 평가 방식, 영향도 산출 기준, 점수 정의 등을 물어본다면, 시스템 프롬프트에 명시된 기준을 바탕으로 친절하고 전문적인 선배로서 상세히 설명해주시오. 이때는 `status`를 `insufficient`로 유지하며 대화를 이어간다.

<추출 항목>
-   **계약 금액 (contract_amount)**
-   **페널티 정보 (penalty_info)**
-   **예상 손실 (loss_estimate)**
-   **지연 일수 (delay_days)**
-   **지연 리스크 (delay_risk)**

<보고서 생성 허용 조건>
1. [조건 A - 사용자 명시 요청]: 사용자가 "보고서로 만들어", "최종 정리해", "출력해" 등 명시적으로 결과물을 요구한 경우.
2. [조건 B - 제안 후 승인]: 당신이 "보고서로 정리할까요?"라고 제안했고 사용자가 "응", "OK", "진행해" 등 긍정 응답한 경우.

<응답 형식>
반드시 아래 JSON 형식을 지키십시오.
{
    "status": "sufficient" | "insufficient",
    "report_requested": true | false, // 위의 조건 A 또는 B 충족 시에만 true
    "message": "상사/선배 톤의 대화 메시지. 충분한 경우 중간 추론을 포함하고 보고서 생성 여부를 문의하십시오.",
    "follow_up_questions": ["..."],
    "extracted_data": { ... }
}

<이미 파악된 정보>
{{extracted_data}}

<대화 내용>
{{conversation_history}}

<사용자 최신 질의>
{{user_input}}

상황을 평가하여 JSON으로만 응답하십시오.
"""

RISK_ENGINE_EVALUATION_PROMPT = """
당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.
제시된 비즈니스 상황과 관련 정보를 바탕으로 기업의 리스크를 분석하고 평가해야 합니다.
다음 평가 항목별로 영향도(Impact 1-5), 발생 가능성(Likelihood 1-5)을 엄격하게 평가하고,
각각에 대한 구체적인 판단 근거를 "회사 기준", "실무 기준", "실제 발생 가능한 리스크", "내부 보고 기준" 관점에서 설명하십시오.

**중요 지침:**
데이터가 부족한 항목(예: 계약 금액 미상, 페널티 조항 불분명 등)에 대해서는 **일반적인 무역 관행, 인코텀즈(Incoterms) 규정, 또는 유사 업계 사례를 바탕으로 가설(Hypothesis)을 세워 분석을 진행**하십시오. "정보가 없어서 분석 불가"라고 대답하지 말고, "통상적인 경우 ~하므로 ~한 리스크가 예상된다"는 식으로 조언하십시오.

<평가 항목>
{{risk_evaluation_items_json}}

<이미 파악된 정보>
{{extracted_data}}

<사용자 질의>
{{user_input}}

<대화 히스토리>
{{conversation_history}}

<RAG 참조 문서>
{{rag_documents}}

<출력 형식>
반드시 아래 JSON 스키마에 따라 응답하십시오.
risk_score는 impact * likelihood로 계산합니다.
risk_level은 risk_score에 따라 low/medium/high/critical로 결정합니다. (critical: 15점 이상, high: 10점 이상, medium: 5점 이상, low: 1점 이상)

```json
{{
    "overall_risk_level": "low" | "medium" | "high" | "critical",
    "risk_factors": [
        {{
            "name": "재정적 손실",
            "impact": 1-5,
            "likelihood": 1-5,
            "risk_score": 1-25,
            "risk_level": "low" | "medium" | "high" | "critical",
            "reasoning": "왜 이 점수와 수준이 나왔는지 구체적인 근거를 회사/실무/발생가능성/보고 기준에서 설명",
            "mitigation_suggestions": ["구체적인 완화 방안 1", "구체적인 완화 방안 2"]
        }},
        // ... 다른 리스크 항목들 ...
    ],
    "overall_assessment": "전반적인 리스크 평가 요약 (담백하고 직설적으로 문제점과 핵심을 짚어줌)"
}}
```
"""

REPORT_GENERATION_PROMPT = """
당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.
제공된 리스크 분석 결과와 모든 정보를 종합하여 최종 리스크 관리 보고서를 JSON 형식으로 생성해야 합니다.
보고서의 각 필드는 담백하고 직설적인 상사형 피드백 스타일을 유지해야 합니다.
절대 불필요한 장문 설명, 감정적 표현, 추상적 조언은 금지합니다.
무엇이 문제인지, 왜 문제인지, 실제 발생 가능한 상황, 지금 해야 할 행동을 항상 포함하십시오.

<분석 결과>
{{risk_scoring_json}}

<사용자 질의 요약>
{{input_summary}}

<유사 사례 및 증거 자료>
{{similar_cases_json}}

<출력 형식>
반드시 아래 JSON 스키마에 따라 응답하십시오.
(schemas.py의 RiskReport와 동일한 구조여야 합니다.)

```json
{{
    "analysis_id": "분석 고유 ID (UUID)",
    "input_summary": "사용자 질의 및 대화 요약 (핵심만 간결하게)",
    "risk_factors": ["식별된 주요 리스크 요인들을 요약 (단어/구 형태)"],
    "risk_scoring": {{... RiskScoring 스키마 준수 ...}},
    "loss_simulation": {{
        "quantitative": "정량적 예상 손실 (예: $10,000 ~ $50,000 또는 '추정 불가')",
        "qualitative": "정성적 손실 시뮬레이션 설명 (상사 스타일)"
    }},
    "control_gap_analysis": {{
        "identified_gaps": ["현재 관리 체계의 허점 1", "허점 2"],
        "recommendations": ["허점 보완을 위한 제안 1", "제안 2"]
    }},
    "prevention_strategy": {{
        "short_term": ["단기적으로 즉시 수행할 행동 1", "행동 2"],
        "long_term": ["장기적인 관점에서 고려할 전략 1", "전략 2"]
    }},
    "similar_cases": [{{... 유사 사례 (RAG 결과) ...}}],
    "confidence_score": 0.0,
    "evidence_sources": ["분석에 활용된 RAG 문서 출처 1", "출처 2"]
}}
```
"""

INPUT_SUMMARY_PROMPT = """
주어진 사용자 질의와 대화 내용을 바탕으로 핵심 내용을 간결하고 담백하게 요약하십시오.
불필요한 설명 없이, 리스크 분석에 필요한 사실 관계만 요약합니다.

<사용자 질의>
{{user_input}}

<대화 히스토리>
{{conversation_history}}

요약:
"""

LOSS_SIMULATION_QUALITATIVE_PROMPT = """
당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.
제시된 리스크 분석 결과를 바탕으로, 실제 발생 가능한 손실 상황을 담백하고 직설적으로 시뮬레이션하십시오.
재정적, 비재정적 영향 모두 고려하며, 신입 직원이 이해하기 쉽지만 과장 없이 현실적인 톤을 유지합니다.
추상적 조언이나 감정적 표현은 금지합니다.

<리스크 분석 결과 요약>
{{risk_summary}}

<출력 형식>
단순 텍스트로 시뮬레이션 내용을 설명하십시오.
예시: "이대로 진행 시, 최소 2주 이상의 납기 지연이 발생하며, 이는 클라이언트와의 관계에 중대한 손상을 초래하여 향후 계약 수주에 부정적인 영향을 미칠 것입니다. 최악의 경우 계약 해지 및 위약금 발생 가능성도 배제할 수 없습니다."
"""

CONTROL_GAP_ANALYSIS_PROMPT = """
당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.
제시된 리스크 분석 결과를 바탕으로, 현재 리스크 관리 체계의 어떤 허점이 문제 상황을 야기했는지 직설적으로 지적하고, 이를 보완하기 위한 구체적인 제안을 하십시오.

<리스크 분석 결과 요약>
{{risk_summary}}

<출력 형식>
아래 JSON 스키마에 따라 응답하십시오.

```json
{{
    "identified_gaps": ["현재 관리 체계의 허점 1", "허점 2"],
    "recommendations": ["허점 보완을 위한 제안 1", "제안 2"]
}}
```
"""

PREVENTION_STRATEGY_PROMPT = """
당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.
제시된 리스크 분석 결과와 허점 분석을 바탕으로, 문제 재발 방지 및 리스크 완화를 위한 단기/장기 예방 전략을 구체적으로 제시하십시오.
신입 직원이 즉시 행동할 수 있는 지시와 장기적인 관점에서 고려할 사항을 명확히 구분합니다.

<리스크 분석 결과 요약>
{{risk_summary}}
<식별된 관리 허점>
{{control_gaps_json}}

<출력 형식>
아래 JSON 스키마에 따라 응답하십시오.

```json
{{
    "short_term": ["단기적으로 즉시 수행할 행동 1", "행동 2"],
    "long_term": ["장기적인 관점에서 고려할 전략 1", "전략 2"]
}}
```
"""

CONFIDENCE_SCORE_PROMPT = """
당신은 '현실적인 선배/상사형 리스크 관리 에이전트'입니다.
제시된 리스크 분석 결과와 RAG 참조 문서들을 바탕으로, 당신이 생성한 최종 리스크 분석 보고서의 신뢰도를 0.0에서 1.0 사이의 점수로 평가하십시오.
신뢰도는 다음과 같은 요소를 종합적으로 고려하여 결정합니다:
-   제공된 정보의 양과 질
-   RAG 문서의 관련성 및 신뢰도
-   분석 과정의 명확성
-   결과 도출의 논리적 일관성

<리스크 분석 결과>
{{risk_report_json}}

<RAG 참조 문서>
{{rag_documents}}

<출력 형식>
단순히 신뢰도 점수(float)만 출력하십시오.  예: 0.85
"""

# --- Component Classes (integrated from deleted files) ---

class SimilarityEngine:
    """
    Checks if user input is similar to risk-related topics using cosine similarity.
    Uses pre-computed embeddings of risk trigger phrases.
    """
    def __init__(self):
        self.settings = get_settings()
        # Pre-defined risk-related reference phrases
        self.reference_phrases = [
            "선적 지연 발생",
            "클레임 발생 가능성",
            "계약 위반 우려",
            "페널티 조항 확인",
            "리스크 분석 필요",
            "손실 예상",
            "문제 발생",
            "invoice 오류",
            "HS code 문제",
            "payment 지연",
            "리스크 평가 기준",
            "점수 산출 방법",
            "분석 절차 문의",
            "영향도 계산",
            "발생 가능성 판단"
        ]
        self.reference_embeddings = []
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Pre-compute embeddings for reference phrases"""
        logger.debug("SimilarityEngine: initializing reference embeddings")
        for phrase in self.reference_phrases:
            embedding = get_embedding(phrase)
            if embedding:
                self.reference_embeddings.append(embedding)
        logger.debug(
            "SimilarityEngine: loaded %s reference embeddings",
            len(self.reference_embeddings),
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return float(np.dot(vec1_np, vec2_np) / (norm(vec1_np) * norm(vec2_np)))
    
    def check_similarity(self, user_input: str, return_score: bool = False):
        """
        Check if user input is similar to risk-related topics.
        
        Args:
            user_input: User's input text
            return_score: If True, return (is_similar, max_score). If False, return is_similar only.
        
        Returns:
            bool or tuple: Similarity check result
        """
        if not self.reference_embeddings:
            logger.warning("SimilarityEngine: no reference embeddings available")
            return (False, 0.0) if return_score else False
        
        user_embedding = get_embedding(user_input)
        if not user_embedding:
            return (False, 0.0) if return_score else False
        
        max_similarity = 0.0
        for ref_embedding in self.reference_embeddings:
            similarity = self._cosine_similarity(user_embedding, ref_embedding)
            max_similarity = max(max_similarity, similarity)
        
        is_similar = max_similarity >= SIMILARITY_THRESHOLD
        
        if return_score:
            return is_similar, max_similarity
        return is_similar


class ConversationManager:
    """
    Manages multi-turn conversation and assesses if enough information is collected
    for risk analysis.
    """
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.upstage_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
    
    def assess_conversation_progress(self, agent_input: RiskManagingAgentInput, extracted_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess if enough information has been collected for risk analysis.
        
        Args:
            agent_input: User input and conversation history
            extracted_data: Previously extracted information
        
        Returns:
            Dict containing status, analysis_ready, message, follow_up_questions, extracted_data
        """
        conversation_history_str = "\n".join([
            f"{turn.get('role', 'Unknown')}: {turn.get('content', '')}"
            for turn in (agent_input.conversation_history or [])
        ])
        
        extracted_data_str = json.dumps(extracted_data or {}, ensure_ascii=False, indent=2)

        prompt = CONVERSATION_ASSESSMENT_PROMPT.replace(
            "{{conversation_history}}", conversation_history_str
        ).replace(
            "{{user_input}}", agent_input.user_input
        ).replace(
            "{{extracted_data}}", extracted_data_str
        )
        
        try:
            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": RISK_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            assessment = safe_json_parse(response_text)
            
            # Collaborative Mode Logic: 
            # analysis_ready is True ONLY IF (status is sufficient) AND (report_requested is True)
            status = assessment.get("status", "insufficient")
            report_requested = assessment.get("report_requested", False)
            analysis_ready = (status == "sufficient") and report_requested
            
            return {
                "status": status,
                "analysis_ready": analysis_ready,
                "report_requested": report_requested,
                "message": assessment.get("message", ""),
                "follow_up_questions": assessment.get("follow_up_questions", []),
                "extracted_data": assessment.get("extracted_data", {}),
                "conversation_stage": "gathering_info" if not analysis_ready else "ready_for_analysis",
                "analysis_in_progress": True
            }
        
        except Exception as e:
            logger.warning("Error in assess_conversation_progress: %s", e)
            return {
                "status": "error",
                "analysis_ready": False,
                "message": f"정보 평가 중 오류가 발생했습니다: {str(e)}",
                "follow_up_questions": [],
                "extracted_data": {},
                "conversation_stage": "error",
                "analysis_in_progress": True
            }


class RAGConnector:
    """
    Connects to RAG system to retrieve relevant documents for risk analysis.
    """
    def __init__(self):
        self.settings = get_settings()
    
    def get_risk_documents(
        self, 
        user_input: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from RAG system.
        
        Args:
            user_input: Current user input
            conversation_history: Previous conversation turns
            k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with metadata
        """
        # Build full query context
        full_query = user_input
        if conversation_history:
            recent_context = " ".join([
                turn.get("content", "")
                for turn in conversation_history[-3:]  # Last 3 turns
            ])
            full_query = f"{recent_context} {user_input}"
        
        # Use tool: search_risk_cases
        from backend.agents.riskmanaging.tools import search_risk_cases

        filtered_documents = search_risk_cases.invoke(
            {
                "query": full_query,
                "k": k,
                "datasets": RAG_DATASETS,
            }
        )

        logger.debug(
            "RAGConnector: retrieved %s risk-related documents",
            len(filtered_documents),
        )
        return filtered_documents
    
    def extract_similar_cases_and_evidence(
        self, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract similar cases and evidence sources from retrieved documents.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Dict with similar_cases and evidence_sources
        """
        similar_cases = []
        evidence_sources = set()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            content = doc.get("document", "")
            
            case_info = {
                "content": content[:500],  # Truncate for brevity
                "source": metadata.get("source", "unknown"),
                "category": metadata.get("original_category", ""),
                "topic": metadata.get("topic", []),
                "distance": doc.get("distance", 1.0)
            }
            similar_cases.append(case_info)
            
            if metadata.get("source"):
                evidence_sources.add(metadata["source"])
        
        return {
            "similar_cases": similar_cases[:5],  # Top 5 cases
            "evidence_sources": list(evidence_sources)
        }


class RiskEngine:
    """
    Evaluates risk factors and calculates risk scores.
    """
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.upstage_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
    
    def evaluate_risk(
        self,
        agent_input: RiskManagingAgentInput,
        rag_documents: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
        extracted_data: Optional[Dict[str, Any]] = None
    ) -> RiskScoring:
        """
        Evaluate risk based on user input and RAG documents.
        
        Args:
            agent_input: User input and context
            rag_documents: Retrieved documents from RAG
            extracted_data: Previously extracted info
        
        Returns:
            RiskScoring object with risk factors and assessment
        """
        # Prepare RAG documents for prompt
        rag_docs_str = "\n\n".join([
            f"[문서 {i+1}] {doc.get('document', '')[:300]}..."
            for i, doc in enumerate(rag_documents[:5])
        ])
        
        # Prepare conversation history
        conversation_history_str = "\n".join([
            f"{turn.get('role', 'User')}: {turn.get('content', '')}"
            for turn in (agent_input.conversation_history or [])
        ])
        
        # Prepare risk evaluation items
        risk_items_str = json.dumps(RISK_EVALUATION_ITEMS, ensure_ascii=False, indent=2)
        extracted_data_str = json.dumps(extracted_data or {}, ensure_ascii=False, indent=2)
        
        prompt = RISK_ENGINE_EVALUATION_PROMPT.replace(
            "{{risk_evaluation_items_json}}", risk_items_str
        ).replace(
            "{{user_input}}", agent_input.user_input
        ).replace(
            "{{conversation_history}}", conversation_history_str
        ).replace(
            "{{rag_documents}}", rag_docs_str if rag_docs_str else "관련 문서 없음"
        ).replace(
            "{{extracted_data}}", extracted_data_str
        )
        
        try:
            user_instruction = build_user_instruction(user_profile)
            system_prompt = f"{RISK_AGENT_SYSTEM_PROMPT}\n추가 지침:\n{user_instruction}"
            
            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            risk_data = safe_json_parse(response_text)
            
            # Convert to RiskScoring object
            from backend.agents.riskmanaging.state import RiskFactor
            risk_factors = [
                RiskFactor(**factor) for factor in risk_data.get("risk_factors", [])
            ]
            
            return RiskScoring(
                overall_risk_level=risk_data.get("overall_risk_level", "medium"),
                risk_factors=risk_factors,
                overall_assessment=risk_data.get("overall_assessment", "")
            )
        
        except Exception as e:
            logger.warning("Error in evaluate_risk: %s", e)
            # Return default risk scoring
            return RiskScoring(
                overall_risk_level="medium",
                risk_factors=[],
                overall_assessment=f"리스크 평가 중 오류 발생: {str(e)}"
            )


class ReportGenerator:
    """
    Generates comprehensive risk analysis report.
    """
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.upstage_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
    
    @traceable(name="report_generator_generate")
    def generate_report(
        self,
        agent_input: RiskManagingAgentInput,
        risk_scoring: RiskScoring,
        similar_cases: List[Dict[str, Any]],
        evidence_sources: List[str],
        rag_documents: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> RiskReport:
        """
        Generate comprehensive risk report.
        
        Args:
            agent_input: User input and context
            risk_scoring: Risk evaluation results
            similar_cases: Similar cases from RAG
            evidence_sources: Evidence document sources
            rag_documents: All retrieved documents
            user_profile: User persona information
        
        Returns:
            RiskReport object
        """
        self.user_profile = user_profile # Store for private methods
        # Generate input summary
        input_summary = self._generate_input_summary(agent_input)
        
        # Generate loss simulation
        loss_simulation = self._generate_loss_simulation(risk_scoring)
        
        # Generate control gap analysis
        control_gap_analysis = self._generate_control_gap_analysis(risk_scoring)
        
        # Generate prevention strategy
        prevention_strategy = self._generate_prevention_strategy(risk_scoring, control_gap_analysis)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(risk_scoring, rag_documents)
        
        # Convert risk_factors to ReportRiskFactor format
        risk_factors_dict = {}
        for factor in risk_scoring.risk_factors:
            risk_factors_dict[factor.name] = ReportRiskFactor(
                name_kr=factor.name,
                impact=factor.impact,
                likelihood=factor.likelihood,
                score=factor.risk_score
            )
        
        return RiskReport(
            input_summary=input_summary,
            risk_factors=risk_factors_dict,
            risk_scoring=risk_scoring,
            loss_simulation=loss_simulation,
            control_gap_analysis=control_gap_analysis,
            prevention_strategy=prevention_strategy,
            similar_cases=similar_cases,
            confidence_score=confidence_score,
            evidence_sources=evidence_sources
        )
    
    def _generate_input_summary(self, agent_input: RiskManagingAgentInput) -> str:
        """Generate concise summary of user input"""
        conversation_history_str = "\n".join([
            f"{turn.get('role', 'User')}: {turn.get('content', '')}"
            for turn in (agent_input.conversation_history or [])
        ])
        
        prompt = INPUT_SUMMARY_PROMPT.replace(
            "{{user_input}}", agent_input.user_input
        ).replace(
            "{{conversation_history}}", conversation_history_str
        )
        
        user_instruction = build_user_instruction(getattr(self, 'user_profile', None))
        system_prompt = f"{RISK_AGENT_SYSTEM_PROMPT}\n추가 지침:\n{user_instruction}"

        try:
            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Error generating input summary: %s", e)
            return agent_input.user_input
    
    def _generate_loss_simulation(self, risk_scoring: RiskScoring) -> LossSimulation:
        """Generate loss simulation"""
        risk_summary = risk_scoring.overall_assessment
        
        prompt = LOSS_SIMULATION_QUALITATIVE_PROMPT.replace(
            "{{risk_summary}}", risk_summary
        )
        
        try:
            user_instruction = build_user_instruction(getattr(self, 'user_profile', None))
            system_prompt = f"{RISK_AGENT_SYSTEM_PROMPT}\n추가 지침:\n{user_instruction}"

            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            qualitative = response.choices[0].message.content.strip()
            
            return LossSimulation(
                quantitative=None,  # Can be enhanced later
                qualitative=qualitative
            )
        except Exception as e:
            logger.warning("Error generating loss simulation: %s", e)
            return LossSimulation(
                quantitative=None,
                qualitative="손실 시뮬레이션 생성 중 오류 발생"
            )
    
    def _generate_control_gap_analysis(self, risk_scoring: RiskScoring) -> ControlGapAnalysis:
        """Generate control gap analysis"""
        risk_summary = risk_scoring.overall_assessment
        
        prompt = CONTROL_GAP_ANALYSIS_PROMPT.replace(
            "{{risk_summary}}", risk_summary
        )
        
        try:
            user_instruction = build_user_instruction(getattr(self, 'user_profile', None))
            system_prompt = f"{RISK_AGENT_SYSTEM_PROMPT}\n추가 지침:\n{user_instruction}"

            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            gap_data = safe_json_parse(response_text)
            
            return ControlGapAnalysis(
                identified_gaps=gap_data.get("identified_gaps", []),
                recommendations=gap_data.get("recommendations", [])
            )
        except Exception as e:
            logger.warning("Error generating control gap analysis: %s", e)
            return ControlGapAnalysis(
                identified_gaps=["분석 중 오류 발생"],
                recommendations=[]
            )
    
    def _generate_prevention_strategy(
        self, 
        risk_scoring: RiskScoring, 
        control_gap_analysis: ControlGapAnalysis
    ) -> PreventionStrategy:
        """Generate prevention strategy"""
        risk_summary = risk_scoring.overall_assessment
        control_gaps_json = json.dumps({
            "identified_gaps": control_gap_analysis.identified_gaps,
            "recommendations": control_gap_analysis.recommendations
        }, ensure_ascii=False)
        
        prompt = PREVENTION_STRATEGY_PROMPT.replace(
            "{{risk_summary}}", risk_summary
        ).replace(
            "{{control_gaps_json}}", control_gaps_json
        )
        
        try:
            user_instruction = build_user_instruction(getattr(self, 'user_profile', None))
            system_prompt = f"{RISK_AGENT_SYSTEM_PROMPT}\n추가 지침:\n{user_instruction}"

            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            strategy_data = safe_json_parse(response_text)
            
            return PreventionStrategy(
                short_term=strategy_data.get("short_term", []),
                long_term=strategy_data.get("long_term", [])
            )
        except Exception as e:
            logger.warning("Error generating prevention strategy: %s", e)
            return PreventionStrategy(
                short_term=["전략 생성 중 오류 발생"],
                long_term=[]
            )
    
    def _calculate_confidence_score(
        self, 
        risk_scoring: RiskScoring, 
        rag_documents: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the analysis"""
        # Simple heuristic: more documents and detailed risk factors = higher confidence
        base_score = 0.5
        
        # Add points for number of RAG documents
        doc_score = min(len(rag_documents) * 0.05, 0.25)
        
        # Add points for number of risk factors analyzed
        factor_score = min(len(risk_scoring.risk_factors) * 0.05, 0.25)
        
        confidence = base_score + doc_score + factor_score
        return min(confidence, 1.0)


# --- Post-processing Function ---

def format_decision_report(raw_response: str, risk_score: int) -> str:
    """
    Wraps the output into a structured decision report.
    """
    decision = "APPROVED"
    if risk_score >= 8:
        decision = "BLOCKED"
    elif 4 <= risk_score <= 7:
        decision = "WARNING"
    
    # Try to parse the raw_response if it's JSON
    try:
        data = json.loads(raw_response)
        summary = data.get("input_summary", "상황 요약 없음")
        # Extract risk description from overall_assessment in risk_scoring
        risk_scoring = data.get("risk_scoring", {})
        risk_desc = risk_scoring.get("overall_assessment", "리스크 설명 없음")
        # Extract loss from loss_simulation
        loss_sim = data.get("loss_simulation", {})
        loss_desc = loss_sim.get("qualitative", "손실 시뮬레이션 없음")
        # Extract actions from prevention_strategy
        prev_strat = data.get("prevention_strategy", {})
        actions = ", ".join(prev_strat.get("short_term", [])) or "즉시 조치 사항 없음"
        # Extract evidence from evidence_sources
        evidence = ", ".join(data.get("evidence_sources", [])) or "증거 자료 없음"
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Fallback for non-JSON responses (gathering info stage)
        summary = "정보 수집 중"
        risk_desc = "추가 정보 확인 필요"
        loss_desc = "정보 부족으로 산출 불가"
        actions = "요청된 정보를 제공하십시오"
        evidence = "대화 맥락"
        # If it's a conversation message, the message itself is the summary/explanation
        if isinstance(raw_response, str) and raw_response.strip():
            summary = raw_response.split("\n")[0][:100] + "..."
    
    report = f"""[DECISION]
{decision}

[SUMMARY]
{summary}

[RISK]
{risk_desc}

[LOSS]
{loss_desc}

[ACTION]
{actions}

[EVIDENCE]
{evidence}"""
    return report

# --- Node functions (to be defined using the above classes) ---

# Node function for preparing the initial state
def prepare_risk_state_node(state: RiskManagingGraphState) -> Dict[str, Any]:
    logger.debug("risk node: prepare_risk_state")
    # Extract from state directly (no agent_input needed)
    user_input = state.get("current_user_input", "")
    conversation_history = state.get("conversation_history", [])

    return {
        "current_user_input": user_input,
        "conversation_history": conversation_history,
        "analysis_ready": False,
        "risk_trigger_detected": False,
        "risk_similarity_score": 0.0,
        "rag_documents": [],
        "risk_scoring": None,
        "report_generated": None,
        "conversation_stage": "prepared",
        "error_message": None,
        "analysis_in_progress": True, # Keep analysis active by default if entered
        "analysis_required": False,
        "agent_response": None, # Explicitly clear previous response
    }

# Node function for detecting trigger words and similarity
def detect_trigger_and_similarity_node(state: RiskManagingGraphState) -> Dict[str, Any]:
    logger.debug("risk node: detect_trigger_and_similarity")
    user_input = state.get("current_user_input", "")
    
    # Trigger word detection
    trigger_detected = False
    user_input_lower = user_input.lower()
    for trigger in RISK_AGENT_TRIGGER_WORDS:
        if trigger.lower() in user_input_lower:
            trigger_detected = True
            break
    
    # Similarity detection
    similarity_engine = SimilarityEngine() # Initialize here
    is_similar, similarity_score = similarity_engine.check_similarity(user_input, return_score=True)
    
    analysis_required = trigger_detected or is_similar
    
    return {
        "risk_trigger_detected": trigger_detected,
        "risk_similarity_score": similarity_score,
        "analysis_required": analysis_required,
        "conversation_stage": "trigger_detection_completed",
    }

# Node function for assessing conversation progress
def assess_conversation_progress_node(state: RiskManagingGraphState) -> Dict[str, Any]:
    logger.debug("risk node: assess_conversation_progress")
    agent_input = RiskManagingAgentInput(
        user_input=state["current_user_input"],
        conversation_history=state["conversation_history"]
    )
    
    conversation_manager = ConversationManager() # Initialize here
    assessment_result = conversation_manager.assess_conversation_progress(agent_input, state.get("extracted_data"))
    
    # Update state based on assessment result
    # Merge extracted_data to prevent losing previously collected information
    old_extracted_data = state.get("extracted_data", {})
    new_extracted_data = assessment_result.get("extracted_data", {})
    
    # Merge: if new has value, use it; otherwise keep old
    merged_data = old_extracted_data.copy()
    for k, v in new_extracted_data.items():
        if v is not None and v != "":
            merged_data[k] = v

    state_updates = {
        "extracted_data": merged_data,
        "analysis_ready": assessment_result.get("analysis_ready", False),
        "report_requested": assessment_result.get("report_requested", False),
        "conversation_stage": assessment_result.get("conversation_stage", "gathering_info"),
        "analysis_in_progress": assessment_result.get("analysis_in_progress", True),
        "agent_response": assessment_result.get("message"), # The message from CM is the response
    }
    
    # If follow-up questions exist, append them to the response
    follow_up_questions = assessment_result.get("follow_up_questions")
    if follow_up_questions:
        state_updates["agent_response"] = (
            f"{state_updates['agent_response']}\n"
            f"추가 정보가 필요합니다:\n"
            + "\n".join([f"- {q}" for q in follow_up_questions])
        )
    
    return state_updates

# Node function for performing full risk analysis (RAG, Risk Engine, Report Gen)
def perform_full_analysis_node(state: RiskManagingGraphState) -> Dict[str, Any]:
    logger.debug("risk node: perform_full_analysis")
    user_input = state["current_user_input"]
    conversation_history = state["conversation_history"]
    
    agent_input = RiskManagingAgentInput(
        user_input=user_input,
        conversation_history=conversation_history
    )

    # 1. RAG Connector
    rag_connector = RAGConnector()
    rag_documents = rag_connector.get_risk_documents(user_input, conversation_history)
    extracted_info = rag_connector.extract_similar_cases_and_evidence(rag_documents)
    similar_cases = extracted_info["similar_cases"]
    evidence_sources = extracted_info["evidence_sources"]

    user_profile = state.get("user_profile")
    
    # 2. Risk Engine
    risk_engine = RiskEngine()
    risk_scoring = risk_engine.evaluate_risk(
        agent_input, 
        rag_documents, 
        user_profile=user_profile,
        extracted_data=state.get("extracted_data")
    )

    # 3. Report Generator
    report_generator = ReportGenerator()
    report_generated = report_generator.generate_report(
        agent_input=agent_input,
        risk_scoring=risk_scoring,
        similar_cases=similar_cases,
        evidence_sources=evidence_sources,
        rag_documents=rag_documents,
        user_profile=user_profile
    )

    return {
        "rag_documents": rag_documents,
        "risk_scoring": risk_scoring,
        "report_generated": report_generated,
        "conversation_stage": "analysis_completed",
        "analysis_in_progress": False,
        "agent_response": report_generated.model_dump_json(indent=2, exclude_none=True), # Send full report as response
    }

# Node function for formatting the final output
def format_final_output_node(state: RiskManagingGraphState) -> Dict[str, Any]:
    logger.debug("risk node: format_final_output")
    report = state.get("report_generated")
    error_message = state.get("error_message")
    agent_response_from_state = state.get("agent_response")
    final_metadata = {"status": "insufficient_info", "analysis_id": None}

    # Calculate max risk score for decision report
    max_risk_score = 0
    risk_scoring = state.get("risk_scoring")
    if risk_scoring and risk_scoring.risk_factors:
        max_risk_score = max(f.risk_score for f in risk_scoring.risk_factors)
    elif state.get("analysis_in_progress"):
        # If uncertainty exists during info gathering, set a moderate score
        max_risk_score = 5 

    if error_message:
        final_response_content = f"죄송합니다. 처리 중 오류가 발생했습니다: {error_message}"
        final_metadata = {"status": "error", "analysis_id": None}
    elif report:
        final_response_content = report.model_dump_json(indent=2, exclude_none=True)
        final_metadata = {"status": "success", "analysis_id": report.analysis_id}
    elif agent_response_from_state:
        # Use intermediate response (e.g. follow-up questions from conversation assessment)
        if hasattr(agent_response_from_state, "response"):
            final_response_content = agent_response_from_state.response
            # Keep existing metadata if present
            if hasattr(agent_response_from_state, "metadata") and agent_response_from_state.metadata:
                final_metadata = agent_response_from_state.metadata
        elif isinstance(agent_response_from_state, dict) and "response" in agent_response_from_state:
            final_response_content = agent_response_from_state["response"]
            if "metadata" in agent_response_from_state and agent_response_from_state["metadata"]:
                final_metadata = agent_response_from_state["metadata"]
        else:
            final_response_content = str(agent_response_from_state)
            final_metadata = {"status": "insufficient_info", "analysis_id": None}
    else:
        # Fallback for when analysis is not complete but no specific error
        final_response_content = "상세 분석을 위해 먼저 리스크 상황(계약 금액, 페널티, 지연 일수 등)을 설명해 주시거나, 궁금하신 점(평가 기준 등)을 물어봐 주세요."
        final_metadata = {"status": "incomplete", "analysis_id": None}

    # Apply the decision report wrapping
    final_response_content = format_decision_report(final_response_content, max_risk_score)

    agent_response = RiskManagingAgentResponse(
        response=final_response_content,
        agent_type="riskmanaging",
        metadata=final_metadata
    )

    return {
        "agent_response": agent_response,
        "conversation_stage": "completed"
    }

# Error handling node
def handle_risk_error_node(state: RiskManagingGraphState, error: Exception) -> Dict[str, Any]:
    logger.error("risk node error: %s", error)
    return {
            "error_message": str(error),
            "conversation_stage": "error",
            "agent_response": f"오류가 발생하여 리스크 분석을 완료할 수 없습니다: {str(error)}",
        }


# This class will be instantiated once in graph.py to pass components to nodes
class RISKMANAGING_COMPONENTS:
        def __init__(self):
            # Initialize all stateless components here if they are truly stateless
            # For stateful components like SimilarityEngine that need pre-computed embeddings,
            # it's better to instantiate them directly in the nodes or ensure they are properly
            # initialized if passed around. Given the current design, re-instantiating in nodes
            # is fine for now as it ensures fresh state for each node execution.
            pass

    # Assuming backend.rag.retriever.search_with_filter is correctly imported or mocked for testing
from backend.rag.retriever import search_with_filter

if __name__ == "__main__":
    print("\n--- Running nodes.py directly for self-test ---")
    
    # Corrected: Use the globally defined classes and functions
    _settings = get_settings() # ensure settings are loadable
    print(f"Settings loaded: {_settings.upstage_api_key is not None}")

    try:
        # Test if node functions are defined
        print(f"Is prepare_risk_state_node callable? {callable(prepare_risk_state_node)}")
        print(f"Is detect_trigger_and_similarity_node callable? {callable(detect_trigger_and_similarity_node)}")
        print(f"Is assess_conversation_progress_node callable? {callable(assess_conversation_progress_node)}")
        print(f"Is perform_full_analysis_node callable? {callable(perform_full_analysis_node)}")
        print(f"Is format_final_output_node callable? {callable(format_final_output_node)}")
        print(f"Is handle_risk_error_node callable? {callable(handle_risk_error_node)}")
        
        # Test basic instantiation of integrated classes using GLOBAL definitions
        rag_connector = RAGConnector() # Use the global RAGConnector
        print(f"RAGConnector instantiated: {rag_connector is not None}")
        
        similarity_engine = SimilarityEngine() # Use the global SimilarityEngine
        print(f"SimilarityEngine instantiated: {similarity_engine is not None}")
        
        conversation_manager = ConversationManager() # Use the global ConversationManager
        print(f"ConversationManager instantiated: {conversation_manager is not None}")

        risk_engine = RiskEngine() # Use the global RiskEngine
        print(f"RiskEngine instantiated: {risk_engine is not None}")

        report_generator = ReportGenerator() # Use the global ReportGenerator
        print(f"ReportGenerator instantiated: {report_generator is not None}")

        print("\n--- Self-test completed successfully ---")

    except Exception as e:
        print(f"\n--- Self-test FAILED with an error: {e} ---")
