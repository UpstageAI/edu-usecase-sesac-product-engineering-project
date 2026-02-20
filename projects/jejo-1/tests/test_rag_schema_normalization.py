from __future__ import annotations

from backend.rag.schema import normalize_metadata


def test_normalize_metadata_uses_context_metadata_for_scenarios_master():
    entry = {
        "id": "scenario_1",
        "context_metadata": {
            "topic": ["logistics"],
            "situation": ["shipment_delay"],
            "priority": "high",
        },
    }

    normalized = normalize_metadata(entry, "dataset/scenarios_master.json")

    assert normalized["original_category"] == "scenarios_master"
    assert normalized["document_type"] == "scenario_case"
    assert normalized["topic"] == ["logistics"]
    assert normalized["situation"] == ["shipment_delay"]
    assert normalized["priority"] == "high"


def test_normalize_metadata_falls_back_category_to_source_basename():
    normalized = normalize_metadata({"id": 1, "metadata": {}}, "dataset/mistakes_master.json")

    assert normalized["original_category"] == "mistakes_master"
    assert normalized["document_type"] == "common_mistake"


def test_normalize_metadata_doc_type_override_takes_precedence():
    entry = {
        "id": 10,
        "category": "trade_qa",
        "metadata": {"doc_type": "FAQ_CUSTOM"},
    }

    normalized = normalize_metadata(entry, "dataset/trade_qa.json")

    assert normalized["document_type"] == "faq_custom"


def test_normalize_metadata_metadata_overrides_context_metadata():
    entry = {
        "id": "scenario_2",
        "context_metadata": {"priority": "low", "topic": ["incoterms"]},
        "metadata": {"priority": "critical"},
    }

    normalized = normalize_metadata(entry, "dataset/scenarios_master.json")

    assert normalized["priority"] == "critical"
    assert normalized["topic"] == ["incoterms"]


def test_normalize_metadata_ignores_generic_doc_type_hint_and_uses_source_mapping():
    entry = {
        "id": "scenario_3",
        "context_metadata": {"doc_type": "Document"},
        "metadata": {},
    }

    normalized = normalize_metadata(entry, "dataset/scenarios_master.json")

    assert normalized["document_type"] == "scenario_case"


def test_normalize_metadata_maps_raw_trade_terms_to_trade_terminology():
    normalized = normalize_metadata({"id": 1, "metadata": {}}, "dataset/raw_trade_terms.json")

    assert normalized["document_type"] == "trade_terminology"
    assert normalized["topic"] == ["trade_terms"]
    assert normalized["situation"] == ["learning"]


def test_normalize_metadata_ceo_style_priority_is_reset_to_default():
    entry = {
        "id": "ceo_1",
        "category": "ceo_style",
        "metadata": {
            "priority": "거래처신뢰",
            "situation": "선적지연",
        },
    }

    normalized = normalize_metadata(entry, "dataset/ceo_style.json")

    assert normalized["ceo_focus"] == "거래처신뢰"
    assert normalized["priority"] == "normal"
