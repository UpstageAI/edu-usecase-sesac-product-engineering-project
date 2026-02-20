# backend/agents/quiz_agent/tools.py

"""
LangChain Tools for QuizAgent

Tools exposed for quiz generation workflow:
- RAG document search
- Quiz quality validation (EvalTool integration)
"""

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
def search_trade_documents(
    query: str,
    k: int = 3,
    document_type: Optional[str] = None,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search trade knowledge base for relevant documents.

    Use this tool to retrieve documents from the RAG system for quiz generation.

    Args:
        query: Search query (e.g., "FOB Incoterms", "무역 용어")
        k: Number of documents to retrieve (default: 3)
        document_type: Filter by document type (e.g., "trade_terminology")
        category: Filter by category (e.g., "Incoterms", "payment_terms")

    Returns:
        List of documents with content and metadata

    Example:
        >>> docs = search_trade_documents("FOB란 무엇인가?", k=5)
        >>> for doc in docs:
        ...     print(doc['document'], doc['metadata'])
    """
    get_settings()  # settings access kept for side effects / config validation

    try:
        results: List[Dict[str, Any]] = []
        preferred_doc_types = [
            "trade_terminology",
            "terminology",
            "faq",
            "quiz_question",
        ]

        if document_type:
            results.extend(
                search_with_filter(
                    query=query,
                    k=max(k, 3),
                    document_type=document_type,
                    category=category,
                )
            )
        else:
            per_type_k = max(1, min(3, k))
            for doc_type in preferred_doc_types:
                results.extend(
                    search_with_filter(
                        query=query,
                        k=per_type_k,
                        document_type=doc_type,
                        category=category,
                    )
                )

            # Category-only probe (if requested by caller)
            if category:
                results.extend(
                    search_with_filter(
                        query=query,
                        k=per_type_k,
                        category=category,
                    )
                )

        results = _dedupe_and_rank(results, k=max(k, 5))

        if not results:
            # Conservative fallback: run broad search then keep quiz-relevant doc types only.
            broad = rag_search(query=query, k=max(k, 8))
            filtered = [
                doc for doc in broad
                if str(doc.get("metadata", {}).get("document_type", "")).lower()
                in set(preferred_doc_types)
            ]
            results = filtered if filtered else broad

        results = _dedupe_and_rank(results, k=k)

        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "document": doc["document"],
                "metadata": doc.get("metadata", {}),
                "source_dataset": doc.get("metadata", {}).get("source_dataset", "unknown"),
                "document_type": doc.get("metadata", {}).get("document_type", "unknown"),
                "topics": doc.get("metadata", {}).get("topic", [])
            })

        return formatted_results

    except Exception as e:
        print(f"Error in search_trade_documents: {e}")
        return []


@tool
def validate_quiz_quality(quiz_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate quiz quality using EvalTool.

    Use this tool to check if generated quiz questions meet quality standards.

    Args:
        quiz_data: Quiz data with questions, answers, and options
            {
                "questions": [
                    {
                        "question": "FOB란 무엇인가?",
                        "correct_answer": "본선 인도 조건",
                        "options": ["본선 인도 조건", "도착지 인도", "공장 인도"],
                        "explanation": "FOB는..."
                    }
                ]
            }

    Returns:
        Validation result with is_valid flag and issues list
        {
            "is_valid": True,
            "total_questions": 5,
            "valid_questions": 5,
            "issues": []
        }

    Example:
        >>> result = validate_quiz_quality(quiz_data)
        >>> if not result['is_valid']:
        ...     print(f"Found issues: {result['issues']}")
    """
    try:
        # Import EvalTool dynamically to avoid circular imports
        from backend.agents.eval_agent import evaluate_quiz_list

        questions = quiz_data.get("questions", [])
        if not questions:
            return {
                "is_valid": False,
                "total_questions": 0,
                "valid_questions": 0,
                "issues": ["No questions provided"]
            }

        # Run EvalTool validation
        validation_results = evaluate_quiz_list(questions)

        # Aggregate results
        total = len(validation_results)
        valid_count = sum(1 for r in validation_results if r.get("is_valid", False))
        all_issues = []

        for i, result in enumerate(validation_results):
            if not result.get("is_valid", False):
                issues = result.get("issues", [])
                all_issues.append(f"Question {i+1}: {', '.join(issues)}")

        return {
            "is_valid": valid_count == total,
            "total_questions": total,
            "valid_questions": valid_count,
            "issues": all_issues
        }

    except Exception as e:
        print(f"Error in validate_quiz_quality: {e}")
        return {
            "is_valid": False,
            "total_questions": 0,
            "valid_questions": 0,
            "issues": [f"Validation error: {str(e)}"]
        }


@tool
def format_quiz_context(
    retrieved_documents: List[Dict[str, Any]],
    include_metadata: bool = True
) -> str:
    """
    Format retrieved documents into a context string for LLM.

    Use this tool to prepare RAG results for quiz generation prompts.

    Args:
        retrieved_documents: List of documents from search_trade_documents
        include_metadata: Whether to include metadata in formatting

    Returns:
        Formatted context string ready for LLM prompt

    Example:
        >>> docs = search_trade_documents("FOB", k=3)
        >>> context = format_quiz_context(docs)
        >>> print(context)
        --- 참조 문서 ---
        문서 1 (출처: icc_trade_terms.json | 유형: trade_terminology):
        FOB (Free On Board)는...
    """
    if not retrieved_documents:
        return ""

    context_str = "\n--- 참조 문서 ---\n"

    for i, doc in enumerate(retrieved_documents):
        context_str += f"\n문서 {i+1}"

        if include_metadata:
            metadata = doc.get("metadata", {})
            source = doc.get("source_dataset", metadata.get("source_dataset", "unknown"))
            doc_type = doc.get("document_type", metadata.get("document_type", "unknown"))
            topics = doc.get("topics", metadata.get("topic", []))

            context_str += f" (출처: {source} | 유형: {doc_type}"
            if topics:
                context_str += f" | 주제: {', '.join(topics)}"
            context_str += ")"

        context_str += ":\n"
        context_str += doc["document"]
        context_str += "\n"

    return context_str


# Export all tools for graph usage
__all__ = [
    "search_trade_documents",
    "validate_quiz_quality",
    "format_quiz_context"
]
