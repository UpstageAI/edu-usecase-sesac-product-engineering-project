# backend/agents/riskmanaging/tools.py

"""
LangChain Tools for RiskManagingAgent

Tools exposed for risk analysis workflow:
- RAG search with dataset filtering
- Risk factor evaluation
- Similar case retrieval
"""

import json
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from backend.rag.retriever import search as rag_search
from backend.rag.retriever import search_with_filter
from backend.config import get_settings


# RAG dataset categories for filtering
RAG_DATASETS = [
    "claims", "mistakes", "emails", "country_rules",
    "BL_CHECK", "CUSTOMS", "SHIPPING", "PAYMENT", "CONTRACT",
    "EMAIL", "NEGOTIATION", "QUALITY", "LOGISTICS", "INSURANCE",
    "COMMUNICATION", "risk_knowledge"
]

DATASET_TO_DOC_TYPES = {
    "claims": ["claim_type"],
    "mistakes": ["common_mistake", "error_checklist"],
    "emails": ["email"],
    "country_rules": ["country_guideline"],
    "negotiation": ["negotiation_strategy"],
    "shipping": ["process_flow"],
    "quality": ["error_checklist", "common_mistake"],
    "logistics": ["claim_type", "process_flow", "country_guideline"],
    "risk_knowledge": ["trade_terminology", "terminology"],
}

FALLBACK_RISK_DOC_TYPES = [
    "claim_type",
    "common_mistake",
    "error_checklist",
    "negotiation_strategy",
    "country_guideline",
    "process_flow",
]


def _dedupe_and_rank(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for doc in sorted(results, key=lambda item: float(item.get("distance", 10.0))):
        key = (
            str(doc.get("document", "")).strip(),
            json.dumps(doc.get("metadata", {}), ensure_ascii=False, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
        if len(deduped) >= k:
            break
    return deduped


@tool
def search_risk_cases(
    query: str,
    k: int = 5,
    datasets: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search risk-related documents from filtered RAG datasets.

    Use this tool to retrieve similar risk cases, mistakes, and compliance info.

    Args:
        query: Search query (e.g., "선적 지연 리스크", "클레임 사례")
        k: Number of documents to retrieve (default: 5)
        datasets: Filter by dataset categories (e.g., ["claims", "mistakes"])
                 If None, searches all RAG_DATASETS

    Returns:
        List of relevant risk cases with metadata

    Example:
        >>> cases = search_risk_cases("선적 지연으로 인한 손해", k=3, datasets=["claims", "mistakes"])
        >>> for case in cases:
        ...     print(case['document'])
    """
    get_settings()  # settings access kept for side effects / config validation

    try:
        requested_datasets = datasets if datasets is not None else RAG_DATASETS
        normalized_dataset_tokens = [str(item).strip().lower() for item in requested_datasets]

        requested_doc_types = set()
        for token in normalized_dataset_tokens:
            requested_doc_types.update(DATASET_TO_DOC_TYPES.get(token, []))

        if not requested_doc_types:
            requested_doc_types.update(FALLBACK_RISK_DOC_TYPES)

        results: List[Dict[str, Any]] = []
        per_type_k = max(1, min(3, k))
        for doc_type in sorted(requested_doc_types):
            results.extend(
                search_with_filter(
                    query=query,
                    k=per_type_k,
                    document_type=doc_type,
                )
            )

        results = _dedupe_and_rank(results, k=max(k, 8))

        if not results:
            broad = rag_search(query=query, k=max(k, 10))
            allowed_doc_types = {doc_type.lower() for doc_type in requested_doc_types}
            filtered = [
                doc
                for doc in broad
                if str(doc.get("metadata", {}).get("document_type", "")).lower() in allowed_doc_types
            ]
            results = filtered if filtered else broad

        results = _dedupe_and_rank(results, k=k)

        # Format results
        formatted_results = []
        for doc in results:
                formatted_results.append({
                    "document": doc["document"],
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("metadata", {}).get("source_dataset", "unknown"),
                    "category": (
                        doc.get("metadata", {}).get("original_category")
                        or doc.get("metadata", {}).get("category", "unknown")
                    ),
                    "priority": doc.get("metadata", {}).get("priority", "medium")
                })

        return formatted_results

    except Exception as e:
        print(f"Error in search_risk_cases: {e}")
        return []


@tool
def evaluate_risk_factors(
    situation_context: str,
    risk_factors: List[str],
    similar_cases: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Evaluate risk factors with impact and likelihood scoring.

    Use this tool to assess multiple risk dimensions and calculate overall risk level.

    Args:
        situation_context: Description of the risk situation
        risk_factors: List of risk factor names to evaluate
                     (e.g., ["재정적 손실", "생산 차질", "고객 신뢰 손실"])
        similar_cases: Optional similar cases from search_risk_cases

    Returns:
        Risk evaluation result with scores and levels
        {
            "evaluated_factors": [
                {
                    "name": "재정적 손실",
                    "impact": 4,
                    "likelihood": 4,
                    "score": 16,
                    "level": "critical",
                    "reasoning": "페널티 조항으로 직접 손실 발생 가능성 높음"
                }
            ],
            "overall_risk_level": "critical",
            "overall_risk_score": 16.0,
            "confidence": 0.85
        }

    Example:
        >>> result = evaluate_risk_factors(
        ...     "선적 5일 지연, 페널티 일당 1%",
        ...     ["재정적 손실", "고객 신뢰 손실"]
        ... )
        >>> print(f"Overall risk: {result['overall_risk_level']}")
    """
    try:
        # Risk level thresholds
        RISK_THRESHOLDS = {
            "critical": 15,
            "high": 10,
            "medium": 5,
            "low": 1
        }

        evaluated_factors = []
        total_score = 0

        # Simple heuristic evaluation (can be enhanced with LLM)
        for factor in risk_factors:
            # Default scoring based on keywords in situation
            impact = 3  # Default medium impact
            likelihood = 3  # Default medium likelihood

            # Adjust based on situation context
            if "페널티" in situation_context or "penalty" in situation_context.lower():
                impact = 4
                likelihood = 4
            elif "클레임" in situation_context or "claim" in situation_context.lower():
                impact = 4
                likelihood = 3
            elif "지연" in situation_context or "delay" in situation_context.lower():
                impact = 3
                likelihood = 4

            score = impact * likelihood

            # Determine risk level
            if score >= RISK_THRESHOLDS["critical"]:
                level = "critical"
            elif score >= RISK_THRESHOLDS["high"]:
                level = "high"
            elif score >= RISK_THRESHOLDS["medium"]:
                level = "medium"
            else:
                level = "low"

            evaluated_factors.append({
                "name": factor,
                "impact": impact,
                "likelihood": likelihood,
                "score": score,
                "level": level,
                "reasoning": f"{factor}: impact={impact}, likelihood={likelihood}"
            })

            total_score += score

        # Calculate overall risk
        avg_score = total_score / len(risk_factors) if risk_factors else 0

        if avg_score >= RISK_THRESHOLDS["critical"]:
            overall_level = "critical"
        elif avg_score >= RISK_THRESHOLDS["high"]:
            overall_level = "high"
        elif avg_score >= RISK_THRESHOLDS["medium"]:
            overall_level = "medium"
        else:
            overall_level = "low"

        # Confidence based on number of similar cases
        confidence = 0.7  # Base confidence
        if similar_cases and len(similar_cases) >= 3:
            confidence = 0.85
        elif similar_cases and len(similar_cases) >= 1:
            confidence = 0.75

        return {
            "evaluated_factors": evaluated_factors,
            "overall_risk_level": overall_level,
            "overall_risk_score": avg_score,
            "confidence": confidence
        }

    except Exception as e:
        print(f"Error in evaluate_risk_factors: {e}")
        return {
            "evaluated_factors": [],
            "overall_risk_level": "unknown",
            "overall_risk_score": 0.0,
            "confidence": 0.0
        }


@tool
def extract_risk_information(conversation_text: str) -> Dict[str, Any]:
    """
    Extract key risk-related information from conversation.

    Use this tool to parse structured information from user input.

    Args:
        conversation_text: Combined conversation history or current input

    Returns:
        Extracted information
        {
            "situation_type": "선적 지연",
            "key_entities": ["A사", "10만 달러"],
            "mentioned_terms": ["페널티", "5일 지연"],
            "urgency_level": "high",
            "missing_info": ["정확한 계약 조건", "대체 공급업체"]
        }

    Example:
        >>> info = extract_risk_information("A사와 10만 달러 계약, 5일 지연 시 1% 페널티")
        >>> print(info['situation_type'])
        "선적 지연"
    """
    import re

    try:
        extracted = {
            "situation_type": "unknown",
            "key_entities": [],
            "mentioned_terms": [],
            "urgency_level": "medium",
            "missing_info": []
        }

        # Extract situation type
        if re.search(r"(지연|delay)", conversation_text, re.IGNORECASE):
            extracted["situation_type"] = "선적 지연"
        elif re.search(r"(클레임|claim|불만)", conversation_text, re.IGNORECASE):
            extracted["situation_type"] = "클레임"
        elif re.search(r"(품질|quality|불량)", conversation_text, re.IGNORECASE):
            extracted["situation_type"] = "품질 이슈"

        # Extract key entities (companies, amounts)
        companies = re.findall(r"([A-Z가-힣]+사)", conversation_text)
        amounts = re.findall(r"([\d,]+\s*(?:달러|원|USD|KRW|만|억))", conversation_text)
        extracted["key_entities"] = companies + amounts

        # Extract mentioned terms
        terms = ["페널티", "penalty", "지연", "delay", "클레임", "claim", "계약", "contract"]
        for term in terms:
            if re.search(term, conversation_text, re.IGNORECASE):
                extracted["mentioned_terms"].append(term)

        # Determine urgency
        if re.search(r"(긴급|urgent|immediately|빨리)", conversation_text, re.IGNORECASE):
            extracted["urgency_level"] = "high"
        elif re.search(r"(천천히|나중에|later)", conversation_text, re.IGNORECASE):
            extracted["urgency_level"] = "low"

        # Check for missing information
        if not re.search(r"(페널티|penalty)", conversation_text, re.IGNORECASE):
            extracted["missing_info"].append("페널티 조항")
        if not re.search(r"(\d{4}-\d{2}-\d{2}|\d+일)", conversation_text, re.IGNORECASE):
            extracted["missing_info"].append("구체적 날짜/기간")

        return extracted

    except Exception as e:
        print(f"Error in extract_risk_information: {e}")
        return {
            "situation_type": "unknown",
            "key_entities": [],
            "mentioned_terms": [],
            "urgency_level": "medium",
            "missing_info": []
        }


@tool
def generate_prevention_strategies(
    risk_evaluation: Dict[str, Any],
    similar_cases: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Generate short-term and long-term prevention strategies.

    Use this tool to create actionable recommendations based on risk assessment.

    Args:
        risk_evaluation: Result from evaluate_risk_factors
        similar_cases: Optional similar cases for reference

    Returns:
        Prevention strategies
        {
            "short_term": "긴급 대체 운송 수단 검토, 고객에게 즉시 통보",
            "long_term": "복수 공급업체 확보, 계약서에 Force Majeure 조항 명시",
            "best_practices": ["재고 버퍼 유지", "정기 공급업체 평가"]
        }

    Example:
        >>> strategies = generate_prevention_strategies(risk_eval, cases)
        >>> print(strategies['short_term'])
    """
    try:
        overall_level = risk_evaluation.get("overall_risk_level", "medium")

        short_term = ""
        long_term = ""
        best_practices = []

        if overall_level == "critical":
            short_term = "즉시 고객에게 상황 통보, 긴급 대체 방안 실행, 손실 최소화 조치"
            long_term = "복수 공급업체 확보, 비상 대응 프로토콜 수립, 보험 가입 검토"
            best_practices = [
                "Critical 리스크는 24시간 모니터링",
                "경영진에게 즉시 에스컬레이션",
                "계약서에 불가항력 조항 명시"
            ]
        elif overall_level == "high":
            short_term = "상황 모니터링 강화, 고객과 커뮤니케이션 유지"
            long_term = "공급망 다변화, 리스크 평가 정기 실시"
            best_practices = [
                "주간 리스크 리뷰 미팅",
                "대체 계획 준비"
            ]
        else:
            short_term = "현재 프로세스 유지, 정기 점검"
            long_term = "표준 운영 절차 준수, 문서화 강화"
            best_practices = [
                "월간 리스크 체크리스트",
                "직원 교육 실시"
            ]

        return {
            "short_term": short_term,
            "long_term": long_term,
            "best_practices": best_practices
        }

    except Exception as e:
        print(f"Error in generate_prevention_strategies: {e}")
        return {
            "short_term": "상황 파악 중",
            "long_term": "추가 분석 필요",
            "best_practices": []
        }


# Export all tools for graph usage
__all__ = [
    "search_risk_cases",
    "evaluate_risk_factors",
    "extract_risk_information",
    "generate_prevention_strategies",
    "RAG_DATASETS"
]
