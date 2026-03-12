# backend/agents/email_agent/tools.py

"""
LangChain Tools for EmailAgent

Tools exposed for email coaching workflow:
- RAG search for email references and mistakes
- Email risk detection
- Tone analysis
- Trade term validation
- Unit consistency validation
"""

import re
import json
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from backend.rag.retriever import search as rag_search
from backend.rag.retriever import search_with_filter
from backend.config import get_settings


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
def search_email_references(
    query: str,
    k: int = 3,
    search_type: str = "all"
) -> List[Dict[str, Any]]:
    """
    Search email templates, mistake cases, and best practices.

    Use this tool to find relevant email examples and common mistakes.

    Args:
        query: Search query (e.g., "payment terms email", "claim response")
        k: Number of documents to retrieve (default: 3)
        search_type: Type of search - "mistakes", "emails", or "all" (default: "all")

    Returns:
        List of documents with email examples or mistake cases

    Example:
        >>> mistakes = search_email_references("FOB 오류", k=5, search_type="mistakes")
        >>> emails = search_email_references("클레임 응답", k=3, search_type="emails")
    """
    get_settings()  # settings access kept for side effects / config validation

    try:
        results: List[Dict[str, Any]] = []
        if search_type == "mistakes":
            target_doc_types = ["common_mistake", "error_checklist"]
        elif search_type == "emails":
            target_doc_types = ["email", "process_flow"]
        else:
            target_doc_types = ["email", "common_mistake", "error_checklist", "process_flow"]

        per_type_k = max(1, min(3, k))
        for doc_type in target_doc_types:
            results.extend(
                search_with_filter(
                    query=query,
                    k=per_type_k,
                    document_type=doc_type,
                )
            )

        results = _dedupe_and_rank(results, k=max(k, 6))

        if not results:
            broad = rag_search(query=query, k=max(k, 8))
            target_set = {doc_type.lower() for doc_type in target_doc_types}
            filtered = [
                doc
                for doc in broad
                if str(doc.get("metadata", {}).get("document_type", "")).lower() in target_set
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
                "type": doc.get("metadata", {}).get("document_type", "unknown")
            })

        return formatted_results

    except Exception as e:
        print(f"Error in search_email_references: {e}")
        return []


@tool
def detect_email_risks(
    email_content: str,
    reference_mistakes: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Detect trading risks in email content.

    Use this tool to identify potential risks like missing terms, incorrect Incoterms, etc.

    Args:
        email_content: The email text to analyze
        reference_mistakes: Optional list of common mistakes from RAG

    Returns:
        List of detected risks with severity levels
        [
            {
                "type": "incoterms_misuse",
                "severity": "critical",
                "current": "FOV incoterms",
                "risk": "존재하지 않는 인코텀즈",
                "recommendation": "FOB [지정 선적항] 사용"
            }
        ]

    Example:
        >>> risks = detect_email_risks("We will ship via FOV terms...")
        >>> for risk in risks:
        ...     print(f"{risk['severity']}: {risk['type']}")
    """
    risks = []

    try:
        # Critical risk patterns
        critical_patterns = {
            "incoterms_invalid": r"\b(FOV|CIV|FOBB|CIIF)\b",
            "liability_admission": r"(책임지겠습니다|전적으로.*책임|모든.*책임)"
        }

        # High risk patterns
        high_patterns = {
            "vague_terms": r"(협의.*결정|나중에|추후|later discuss)",
            "aggressive_tone": r"(반드시|must|immediately|빨리|urgent)",
        }

        # Medium risk patterns
        medium_patterns = {}

        # Presence + absence checks for missing details
        if re.search(r"(결제|payment)", email_content, re.IGNORECASE) and not re.search(
            r"(L/C|T/T|D/P|D/A|CAD)", email_content, re.IGNORECASE
        ):
            risks.append({
                "type": "payment_missing",
                "severity": "critical",
                "current": "",
                "risk": "Critical issue detected: payment_missing",
                "recommendation": "Specify payment terms (L/C, T/T, D/P, D/A, CAD)"
            })

        if re.search(r"(수량|quantity)", email_content, re.IGNORECASE) and not re.search(
            r"\b\d+(?:[.,]\d+)?\b", email_content, re.IGNORECASE
        ):
            risks.append({
                "type": "quantity_missing",
                "severity": "medium",
                "current": "",
                "risk": "Information may be incomplete",
                "recommendation": "Add specific quantity values"
            })

        if re.search(r"(납기|delivery)", email_content, re.IGNORECASE) and not re.search(
            r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?|\d{1,2}\s*월\s*\d{1,2}\s*일|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b)",
            email_content,
            re.IGNORECASE,
        ):
            risks.append({
                "type": "date_missing",
                "severity": "medium",
                "current": "",
                "risk": "Information may be incomplete",
                "recommendation": "Add a specific delivery date"
            })

        # Check critical risks
        for risk_type, pattern in critical_patterns.items():
            matches = re.findall(pattern, email_content, re.IGNORECASE)
            if matches:
                risks.append({
                    "type": risk_type,
                    "severity": "critical",
                    "current": matches[0] if matches else "",
                    "risk": f"Critical issue detected: {risk_type}",
                    "recommendation": "Specify correct terms and conditions"
                })

        # Check high risks
        for risk_type, pattern in high_patterns.items():
            if re.search(pattern, email_content, re.IGNORECASE):
                risks.append({
                    "type": risk_type,
                    "severity": "high",
                    "current": "",
                    "risk": f"High risk: {risk_type}",
                    "recommendation": "Use clear and professional language"
                })

        # Check medium risks
        for risk_type, pattern in medium_patterns.items():
            if re.search(pattern, email_content, re.IGNORECASE):
                risks.append({
                    "type": risk_type,
                    "severity": "medium",
                    "current": "",
                    "risk": f"Information may be incomplete",
                    "recommendation": "Provide specific details"
                })

        # Limit to top 5 risks by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        risks.sort(key=lambda x: severity_order.get(x["severity"], 9))
        risks = risks[:5]

        return risks

    except Exception as e:
        print(f"Error in detect_email_risks: {e}")
        return []


@tool
def analyze_email_tone(
    email_content: str,
    recipient_country: Optional[str] = None,
    purpose: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze email tone and cultural appropriateness.

    Use this tool to assess tone (casual/professional/formal) and cultural fit.

    Args:
        email_content: The email text to analyze
        recipient_country: Target country for cultural check (e.g., "미국", "일본")
        purpose: Email purpose (e.g., "negotiation", "complaint", "inquiry")

    Returns:
        Tone analysis result
        {
            "current_tone": "casual",
            "recommended_tone": "professional",
            "score": 7.0,
            "summary": "Overall professional with minor issues",
            "issues": ["과도한 사과 표현 반복"],
            "improvements": ["더 간결한 표현 사용"]
        }

    Example:
        >>> analysis = analyze_email_tone("Hi! Really sorry about this...", recipient_country="미국")
        >>> print(f"Tone score: {analysis['score']}/10")
    """
    try:
        # Simple tone analysis based on keywords
        casual_markers = len(re.findall(r'\b(hi|hey|thanks|btw)\b', email_content, re.IGNORECASE))
        formal_markers = len(re.findall(r'\b(dear|sincerely|respectfully|kindly)\b', email_content, re.IGNORECASE))
        aggressive_markers = len(re.findall(r'\b(must|immediately|urgent|빨리|반드시)\b', email_content, re.IGNORECASE))
        apologetic_markers = len(re.findall(r'\b(sorry|죄송|미안|apologize)\b', email_content, re.IGNORECASE))

        # Determine tone
        if aggressive_markers > 2:
            current_tone = "aggressive"
            score = 4.0
        elif apologetic_markers > 3:
            current_tone = "overly apologetic"
            score = 5.5
        elif casual_markers > formal_markers:
            current_tone = "casual"
            score = 6.0
        elif formal_markers > casual_markers:
            current_tone = "formal"
            score = 8.5
        else:
            current_tone = "professional"
            score = 7.5

        # Generate recommendations
        issues = []
        improvements = []

        if aggressive_markers > 0:
            issues.append("공격적 표현 사용")
            improvements.append("'We kindly request' 등 부드러운 표현 사용")

        if apologetic_markers > 3:
            issues.append("과도한 사과 표현")
            improvements.append("한 번의 진심 어린 사과로 충분")

        if casual_markers > 3:
            issues.append("비즈니스 이메일로 너무 캐주얼")
            improvements.append("'Dear' 등 격식 있는 표현 사용")

        return {
            "current_tone": current_tone,
            "recommended_tone": "professional",
            "score": score,
            "summary": f"Tone is {current_tone} with score {score}/10",
            "issues": issues if issues else ["No major tone issues"],
            "improvements": improvements if improvements else ["Maintain current professional tone"]
        }

    except Exception as e:
        print(f"Error in analyze_email_tone: {e}")
        return {
            "current_tone": "unknown",
            "recommended_tone": "professional",
            "score": 5.0,
            "summary": "Unable to analyze",
            "issues": [f"Error: {str(e)}"],
            "improvements": []
        }


@tool
def validate_trade_terms(
    email_content: str,
    rag_documents: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Validate trade terminology accuracy using RAG.

    Use this tool to check if trade terms (FOB, CIF, L/C, etc.) are used correctly.

    Args:
        email_content: The email text containing trade terms
        rag_documents: Optional RAG documents for term verification

    Returns:
        Validation result with incorrect/verified terms
        {
            "incorrect_terms": [
                {
                    "found": "FOV",
                    "should_be": "FOB",
                    "confidence": 0.95,
                    "context": "We will ship via FOV...",
                    "definition": "FOB (Free On Board)는..."
                }
            ],
            "verified_terms": [
                {"term": "L/C", "full_name": "Letter of Credit"}
            ],
            "suggestions": ["FOV → FOB로 수정하세요"]
        }

    Example:
        >>> result = validate_trade_terms("Payment via L/C and FOV terms")
        >>> if result['incorrect_terms']:
        ...     print("Found errors:", result['incorrect_terms'])
    """
    try:
        # Extract potential trade terms
        term_patterns = r'\b([A-Z]{2,4})\b'
        found_terms = re.findall(term_patterns, email_content)

        # Known valid trade terms
        valid_terms = {
            "FOB", "CIF", "CFR", "EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP",
            "FAS", "L/C", "B/L", "AWB", "CFS", "FCL", "LCL", "TEU", "FEU"
        }

        # Common typos/mistakes
        common_mistakes = {
            "FOV": "FOB",
            "FOBB": "FOB",
            "CIV": "CIF",
            "CIIF": "CIF",
            "LC": "L/C",
            "BL": "B/L"
        }

        incorrect_terms = []
        verified_terms = []

        for term in set(found_terms):
            if term in common_mistakes:
                incorrect_terms.append({
                    "found": term,
                    "should_be": common_mistakes[term],
                    "confidence": 0.95,
                    "context": f"Found '{term}' in email",
                    "definition": f"{common_mistakes[term]} is the correct term"
                })
            elif term in valid_terms:
                verified_terms.append({
                    "term": term,
                    "full_name": term  # Could be enhanced with RAG lookup
                })

        suggestions = []
        for item in incorrect_terms:
            suggestions.append(f"{item['found']} → {item['should_be']}로 수정하세요")

        return {
            "incorrect_terms": incorrect_terms,
            "verified_terms": verified_terms,
            "suggestions": suggestions
        }

    except Exception as e:
        print(f"Error in validate_trade_terms: {e}")
        return {
            "incorrect_terms": [],
            "verified_terms": [],
            "suggestions": []
        }


@tool
def validate_units(email_content: str) -> Dict[str, Any]:
    """
    Validate unit consistency (weight, volume, containers).

    Use this tool to detect mixed or inconsistent units in trade correspondence.

    Args:
        email_content: The email text containing quantities and units

    Returns:
        Validation result with inconsistencies and standardized format
        {
            "inconsistencies": [
                {
                    "text": "20ton and 20000kg",
                    "issue": "Mixed weight units",
                    "suggestion": "Use 20 MT (20,000 kg)",
                    "severity": "warning"
                }
            ],
            "standardized": "20 MT (20,000 kg)",
            "unit_summary": {
                "weight": ["20 ton", "20000 kg"],
                "volume": [],
                "container": []
            }
        }

    Example:
        >>> result = validate_units("Ship 20ton or 20000kg of goods")
        >>> if result['inconsistencies']:
        ...     print("Unit issues found:", result['inconsistencies'])
    """
    try:
        inconsistencies = []
        unit_summary = {"weight": [], "volume": [], "container": []}

        # Weight units pattern
        weight_pattern = r'(\d+[\.,]?\d*)\s*(ton|tons|mt|kg|kilogram|lb|pound)s?'
        volume_pattern = r'(\d+[\.,]?\d*)\s*(cbm|cft|m3|ft3)'
        container_pattern = r'(\d+)\s*(20|40)\'?\s*(gp|hc|hq|rf|ot)?'

        # Find all units
        weight_units = re.findall(weight_pattern, email_content, re.IGNORECASE)
        volume_units = re.findall(volume_pattern, email_content, re.IGNORECASE)
        container_units = re.findall(container_pattern, email_content, re.IGNORECASE)

        # Check weight unit consistency
        if len(weight_units) > 1:
            units_used = set([unit[1].lower() for unit in weight_units])
            if len(units_used) > 1:
                # Check if mixing ton/MT with kg
                if ('ton' in units_used or 'mt' in units_used) and 'kg' in units_used:
                    inconsistencies.append({
                        "text": ", ".join([f"{w[0]} {w[1]}" for w in weight_units]),
                        "issue": "혼용된 무게 단위 (ton과 kg)",
                        "suggestion": "MT(톤) 단위로 통일 (1 MT = 1,000 kg)",
                        "severity": "warning"
                    })

        # Check volume unit consistency
        if len(volume_units) > 1:
            units_used = set([unit[1].lower() for unit in volume_units])
            if 'cbm' in units_used and 'cft' in units_used:
                inconsistencies.append({
                    "text": ", ".join([f"{v[0]} {v[1]}" for v in volume_units]),
                    "issue": "혼용된 부피 단위 (CBM과 CFT)",
                    "suggestion": "CBM 단위로 통일 (1 CBM ≈ 35.31 CFT)",
                    "severity": "warning"
                })

        # Populate unit summary
        for w in weight_units:
            unit_summary["weight"].append(f"{w[0]} {w[1]}")
        for v in volume_units:
            unit_summary["volume"].append(f"{v[0]} {v[1]}")
        for c in container_units:
            unit_summary["container"].append(f"{c[0]}x{c[1]}' {c[2] if c[2] else 'GP'}")

        # Generate standardized format
        standardized = ""
        if weight_units:
            first_weight = weight_units[0]
            standardized = f"{first_weight[0]} {first_weight[1].upper()}"

        return {
            "inconsistencies": inconsistencies,
            "standardized": standardized,
            "unit_summary": unit_summary
        }

    except Exception as e:
        print(f"Error in validate_units: {e}")
        return {
            "inconsistencies": [],
            "standardized": "",
            "unit_summary": {"weight": [], "volume": [], "container": []}
        }


# Export all tools for graph usage
__all__ = [
    "search_email_references",
    "detect_email_risks",
    "analyze_email_tone",
    "validate_trade_terms",
    "validate_units"
]
