from typing import Dict, List, Any
import os

# Define the unified metadata schema with default values
UNIFIED_METADATA_SCHEMA = {
    "original_category": "unknown",
    "document_type": "unknown",
    "country": ["Global"],
    "role": ["unknown"],
    "situation": ["general"],
    "topic": ["general_trade"],
    "priority": "normal",
    "level": "working",
    "reliability": "high",
    "use_case": ["general_info"],
    "source_dataset": "unknown"
}


def _is_empty_or_generic(value: Any) -> bool:
    generic_tokens = {"", "unknown", "general", "general_info", "general_trade", "document"}
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in generic_tokens
    if isinstance(value, list):
        if not value:
            return True
        lowered = {str(item).strip().lower() for item in value if str(item).strip()}
        return not lowered or lowered.issubset(generic_tokens)
    return False

def normalize_metadata(entry: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """
    Normalizes metadata for a given entry to ensure all fields in the UNIFIED_METADATA_SCHEMA exist,
    filling missing ones with appropriate defaults. Preserves existing metadata not in the schema.

    Args:
        entry (Dict[str, Any]): The raw entry dictionary containing 'category' and 'metadata'.
        source_file (str): The full path to the source dataset file (e.g., "dataset/emails.json").

    Returns:
        Dict[str, Any]: A new dictionary with all unified metadata fields, with defaults
                        for any missing fields and ensuring correct type (list/string).
    """
    base_name = os.path.basename(source_file).replace(".json", "")
    metadata = entry.get("metadata", {})
    context_metadata = entry.get("context_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if isinstance(context_metadata, dict):
        # scenarios_master 같은 파일은 metadata 대신 context_metadata를 사용
        metadata = {**context_metadata, **metadata}

    normalized = UNIFIED_METADATA_SCHEMA.copy()
    
    # Store all existing metadata fields first
    for key, value in metadata.items():
        normalized[key] = value

    # Set original_category from entry's category field
    normalized["original_category"] = entry.get("category", base_name or "unknown")

    forced_doc_type_by_source = {
        "scenarios_master": "scenario_case",
        "users_master": "user_profile",
        "mistakes_master": "common_mistake",
    }
    forced_doc_type = forced_doc_type_by_source.get(base_name)
    if forced_doc_type:
        normalized["document_type"] = forced_doc_type
    # Infer document_type based on category or source_file name if not explicitly set
    elif "document_type" not in metadata or metadata.get("document_type") == "unknown":
        inferred_type = None

        if metadata.get("doc_type"):
            doc_type_candidate = str(metadata.get("doc_type")).strip().lower()
            if doc_type_candidate not in {"", "unknown", "document"}:
                inferred_type = doc_type_candidate

        source_doc_type_map = {
            "emails": "email",
            "claims": "claim_type",
            "company_domain": "terminology",
            "country_rules": "country_guideline",
            "document_errors": "error_checklist",
            "internal_process": "process_flow",
            "kpi": "kpi_metric",
            "mistakes": "common_mistake",
            "negotiation": "negotiation_strategy",
            "quiz_samples": "quiz_question",
            "trade_qa": "faq",
            "ceo_style": "ceo_guideline",
            "trade_terminology": "trade_terminology",
            "trade_dictionary_full": "trade_terminology",
            "icc_trade_terms": "trade_terminology",
            "raw_trade_dictionary": "trade_terminology",
            "raw_trade_terms": "trade_terminology",
            "mistakes_master": "common_mistake",
            "scenarios_master": "scenario_case",
            "users_master": "user_profile",
        }
        if not inferred_type:
            inferred_type = source_doc_type_map.get(base_name)

        if not inferred_type and normalized["original_category"] != "unknown":
            inferred_type = normalized["original_category"]

        normalized["document_type"] = inferred_type or "document"

    # Enrich sparse topic/situation defaults for high-volume document types.
    fallback_topic_by_doc_type = {
        "trade_terminology": ["trade_terms"],
        "scenario_case": ["risk_scenario"],
        "user_profile": ["coaching_profile"],
    }
    fallback_situation_by_doc_type = {
        "trade_terminology": ["learning"],
        "scenario_case": ["work_review"],
        "user_profile": ["onboarding"],
    }

    if _is_empty_or_generic(normalized.get("topic")):
        inferred_topic = fallback_topic_by_doc_type.get(str(normalized.get("document_type", "")).lower())
        if inferred_topic:
            normalized["topic"] = inferred_topic

    if _is_empty_or_generic(normalized.get("situation")):
        inferred_situation = fallback_situation_by_doc_type.get(str(normalized.get("document_type", "")).lower())
        if inferred_situation:
            normalized["situation"] = inferred_situation

    # Special handling for ceo_style's existing "priority" field (rename to ceo_focus)
    if normalized["original_category"] == "ceo_style" and "priority" in metadata:
        normalized["ceo_focus"] = metadata["priority"]
        # Ensure the new 'priority' field is not overwritten by the old one
        if "priority" not in UNIFIED_METADATA_SCHEMA: # if we have already moved it
            normalized["priority"] = UNIFIED_METADATA_SCHEMA["priority"] # Reset to default if not set by content logic

    # Ensure source_dataset is set
    normalized["source_dataset"] = os.path.basename(source_file)
    
    # Apply defaults for missing fields from UNIFIED_METADATA_SCHEMA
    for key, default_value in UNIFIED_METADATA_SCHEMA.items():
        if key not in normalized or normalized[key] is None or (isinstance(normalized[key], (str, list)) and not normalized[key]):
            normalized[key] = default_value

    # --- Type Enforcement and Cleaning ---
    # Ensure array types are actually lists and clean them
    for key in ["country", "role", "situation", "topic", "use_case"]:
        current_value = normalized.get(key)
        if not isinstance(current_value, list):
            # If not a list, try to convert to list. Handle potential single string values.
            if isinstance(current_value, str) and current_value.strip():
                normalized[key] = [current_value.strip()]
            else: # Use default if empty, None, or non-string non-list
                normalized[key] = UNIFIED_METADATA_SCHEMA[key]
        
        # Clean list: remove empty strings, "unknown", "general", "general_info", and duplicates
        if isinstance(normalized[key], list):
            cleaned_list = []
            for item in normalized[key]:
                if isinstance(item, str) and item.strip() and item.strip().lower() not in ["unknown", "general", "general_info"]:
                    cleaned_list.append(item.strip())
            
            if not cleaned_list: # If list becomes empty after cleaning, use schema default
                normalized[key] = UNIFIED_METADATA_SCHEMA[key]
            else: # Remove duplicates and sort for consistency
                normalized[key] = list(sorted(list(set(cleaned_list))))

    # Ensure string types are actually strings
    for key in ["priority", "level", "reliability", "original_category", "document_type"]:
        current_value = normalized.get(key)
        if not isinstance(current_value, str) or not current_value.strip():
            # If not a string or empty, try to convert or use default
            if isinstance(current_value, list) and current_value:
                normalized[key] = str(current_value[0]).strip() # Take first element if list
            else:
                normalized[key] = UNIFIED_METADATA_SCHEMA[key] # Use default

    # Remove the temporary 'category' field if it was passed through from entry.get("category")
    if "category" in normalized and "category" not in UNIFIED_METADATA_SCHEMA and normalized["category"] == normalized["original_category"]:
        del normalized["category"]

    return normalized


if __name__ == '__main__':
    # Test cases
    print("--- Schema Normalization Tests ---")

    # Test 1: Empty metadata
    print("\nTest 1: Empty metadata")
    entry_1 = {"id": 1, "category": "test_cat", "content": "test content", "metadata": {}}
    normalized_1 = normalize_metadata(entry_1, "test_file_1.json")
    print(f"Normalized: {normalized_1}")
    
    
    assert normalized_1["source_dataset"] == "test_file_1.json"
    assert normalized_1["original_category"] == "test_cat"
    assert normalized_1["document_type"] == "test_cat"
    assert normalized_1["role"] == ["unknown"]
    assert normalized_1["situation"] == ["general"]
    assert normalized_1["priority"] == "normal"

    # Test 2: Partial metadata with some existing fields
    print("\nTest 2: Partial metadata with existing fields")
    entry_2 = {"id": 2, "category": "emails", "content": "email content", "metadata": {
        "role": "sales",
        "priority": "high",
        "some_extra_field": "value"
    }}
    normalized_2 = normalize_metadata(entry_2, "dataset/emails.json")
    print(f"Normalized: {normalized_2}")
    assert "some_extra_field" in normalized_2
    assert normalized_2["original_category"] == "emails"
    assert normalized_2["document_type"] == "email" # Inferred from source_file
    assert normalized_2["role"] == ["sales"]
    assert normalized_2["priority"] == "high"
    assert normalized_2["source_dataset"] == "emails.json"

    # Test 3: Existing list fields, ensure they remain lists and are cleaned
    print("\nTest 3: Existing list fields, with cleaning")
    entry_3 = {"id": 3, "category": "claims", "content": "claim content", "metadata": {
        "situation": ["delay", "customs_issue", "", "  ", "unknown"],
        "topic": ["logistics", "compliance"]
    }}
    normalized_3 = normalize_metadata(entry_3, "dataset/claims.json")
    print(f"Normalized: {normalized_3}")
    assert normalized_3["document_type"] == "claim_type" # Inferred from source_file
    assert normalized_3["situation"] == ["customs_issue", "delay"] # Cleaned, sorted
    assert normalized_3["topic"] == ["compliance", "logistics"] # Cleaned, sorted
    
    # Test 4: ceo_style specific handling - old "priority" to "ceo_focus"
    print("\nTest 4: ceo_style with existing 'priority' (to be renamed 'ceo_focus')")
    entry_4 = {"id": 4, "category": "ceo_style", "content": "선적지연 | 거래처신뢰 | 즉시보고", "metadata": {
        "situation": "선적지연",
        "priority": "거래처신뢰", # This is the old priority, should become ceo_focus
        "action": "즉시보고"
    }}
    normalized_4 = normalize_metadata(entry_4, "dataset/ceo_style.json")
    print(f"Normalized: {normalized_4}")
    assert normalized_4["ceo_focus"] == "거래처신뢰"
    assert normalized_4["priority"] == "normal" # Should revert to default normal, as it was not a severity priority
    assert normalized_4["situation"] == ["선적지연"]
    assert normalized_4["document_type"] == "ceo_guideline"

    # Test 5: Mixed types, ensure conversion and default if necessary
    print("\nTest 5: Mixed types (string for list-expected, list for string-expected)")
    entry_5 = {"id": 5, "category": "kpi", "content": "some kpi", "metadata": {
        "role": ["manager", "executive"],
        "level": ["expert"], # List when schema expects string
        "topic": "incoterms", # String when schema expects list
        "priority": "normal"
    }}
    normalized_5 = normalize_metadata(entry_5, "dataset/kpi.json")
    print(f"Normalized: {normalized_5}")
    assert normalized_5["document_type"] == "kpi_metric"
    assert normalized_5["role"] == ["executive", "manager"]
    assert normalized_5["level"] == "expert"
    assert normalized_5["topic"] == ["incoterms"]
    assert normalized_5["priority"] == "normal"

    # Test 6: Empty string or None in list-expected field
    print("\nTest 6: Empty string or None in list-expected field")
    entry_6 = {"id": 6, "category": "trade_qa", "content": "some content", "metadata": {
        "topic": ["", None, "customs_procedure"]
    }}
    normalized_6 = normalize_metadata(entry_6, "dataset/trade_qa.json")
    print(f"Normalized: {normalized_6}")
    assert normalized_6["topic"] == ["customs_procedure"] # Cleaned
    assert normalized_6["use_case"] == UNIFIED_METADATA_SCHEMA["use_case"] # Default

    print("\nAll tests passed (visually inspected).")
