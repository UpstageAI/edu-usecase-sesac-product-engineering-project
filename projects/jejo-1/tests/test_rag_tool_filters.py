from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import backend.agents.email_agent.tools as email_tools
import backend.agents.quiz_agent.tools as quiz_tools
import backend.agents.riskmanaging.tools as risk_tools


def _stub_doc(
    document_type: str,
    *,
    document: str | None = None,
    distance: float = 0.1,
    original_category: str = "stub_category",
) -> Dict[str, Any]:
    return {
        "document": document or f"stub-{document_type}",
        "metadata": {
            "document_type": document_type,
            "source_dataset": "stub.json",
            "topic": ["stub_topic"],
            "original_category": original_category,
            "priority": "high",
        },
        "distance": distance,
    }


def test_quiz_search_trade_documents_uses_preferred_doc_types(monkeypatch):
    calls = []

    monkeypatch.setattr(quiz_tools, "get_settings", lambda: SimpleNamespace(upstage_api_key="test"))

    def fake_search_with_filter(*, query, k, document_type=None, category=None):
        calls.append((document_type, category, k))
        return [_stub_doc(document_type=document_type, distance=0.1 + len(calls) * 0.01)]

    monkeypatch.setattr(quiz_tools, "search_with_filter", fake_search_with_filter)
    monkeypatch.setattr(quiz_tools, "rag_search", lambda *args, **kwargs: [])

    result = quiz_tools.search_trade_documents.invoke({"query": "무역용어 퀴즈내줘", "k": 3})

    assert [doc_type for doc_type, _, _ in calls] == [
        "trade_terminology",
        "terminology",
        "faq",
        "quiz_question",
    ]
    assert len(result) == 3
    assert all(item["document_type"] in {"trade_terminology", "terminology", "faq", "quiz_question"} for item in result)


def test_quiz_search_trade_documents_respects_explicit_document_type(monkeypatch):
    calls = []

    monkeypatch.setattr(quiz_tools, "get_settings", lambda: SimpleNamespace(upstage_api_key="test"))

    def fake_search_with_filter(*, query, k, document_type=None, category=None):
        calls.append((document_type, category, k))
        return [_stub_doc(document_type=document_type, document="faq-doc")]

    monkeypatch.setattr(quiz_tools, "search_with_filter", fake_search_with_filter)
    monkeypatch.setattr(quiz_tools, "rag_search", lambda *args, **kwargs: [])

    result = quiz_tools.search_trade_documents.invoke(
        {
            "query": "FOB란?",
            "k": 2,
            "document_type": "faq",
            "category": "incoterms",
        }
    )

    assert calls == [("faq", "incoterms", 3)]
    assert result
    assert all(item["document_type"] == "faq" for item in result)


def test_email_search_references_uses_mistake_doc_types(monkeypatch):
    calls = []

    monkeypatch.setattr(email_tools, "get_settings", lambda: SimpleNamespace(upstage_api_key="test"))

    def fake_search_with_filter(*, query, k, document_type=None):
        calls.append((document_type, k))
        return [_stub_doc(document_type=document_type)]

    monkeypatch.setattr(email_tools, "search_with_filter", fake_search_with_filter)
    monkeypatch.setattr(email_tools, "rag_search", lambda *args, **kwargs: [])

    result = email_tools.search_email_references.invoke(
        {"query": "BL 오류", "k": 3, "search_type": "mistakes"}
    )

    assert [doc_type for doc_type, _ in calls] == ["common_mistake", "error_checklist"]
    assert result
    assert {item["type"] for item in result} <= {"common_mistake", "error_checklist"}


def test_risk_search_cases_maps_dataset_tokens_to_doc_types(monkeypatch):
    calls = []

    monkeypatch.setattr(risk_tools, "get_settings", lambda: SimpleNamespace(upstage_api_key="test"))

    def fake_search_with_filter(*, query, k, document_type=None):
        calls.append((document_type, k))
        category = "claims" if document_type == "claim_type" else "mistakes"
        return [_stub_doc(document_type=document_type, original_category=category)]

    monkeypatch.setattr(risk_tools, "search_with_filter", fake_search_with_filter)
    monkeypatch.setattr(risk_tools, "rag_search", lambda *args, **kwargs: [])

    result = risk_tools.search_risk_cases.invoke(
        {"query": "선적 지연 페널티", "k": 5, "datasets": ["claims", "mistakes"]}
    )

    assert [doc_type for doc_type, _ in calls] == ["claim_type", "common_mistake", "error_checklist"]
    assert [k_value for _, k_value in calls] == [3, 3, 3]
    assert result
    assert all(item["category"] != "unknown" for item in result)


def test_risk_search_cases_fallback_filters_broad_results(monkeypatch):
    monkeypatch.setattr(risk_tools, "get_settings", lambda: SimpleNamespace(upstage_api_key="test"))
    monkeypatch.setattr(risk_tools, "search_with_filter", lambda **kwargs: [])
    monkeypatch.setattr(
        risk_tools,
        "rag_search",
        lambda *args, **kwargs: [
            _stub_doc("email"),
            _stub_doc("claim_type", document="claim-only"),
            _stub_doc("faq"),
        ],
    )

    result = risk_tools.search_risk_cases.invoke(
        {"query": "지연 클레임", "k": 3, "datasets": ["claims"]}
    )

    assert result
    assert [item["metadata"]["document_type"] for item in result] == ["claim_type"]
