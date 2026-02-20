#!/usr/bin/env python3
"""
RAG 품질 지표 리포트 생성 스크립트.

Usage:
  .venv/bin/python scripts/rag_quality_report.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

# Ensure project root is importable when run as a script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag.chroma_client import get_or_create_collection
from backend.rag.ingest import _prepare_content
from backend.rag.schema import normalize_metadata


def _load_dataset_rows(dataset_dir: str = "dataset") -> List[tuple[str, List[Dict[str, Any]]]]:
    rows: List[tuple[str, List[Dict[str, Any]]]] = []
    for file_path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            rows.append((os.path.basename(file_path), data))
    return rows


def _dataset_metrics() -> Dict[str, Any]:
    dataset_rows = _load_dataset_rows()

    total_rows = 0
    ingestable_rows = 0
    short_lt15 = 0
    short_lt30 = 0
    default_topic_situation = 0
    unknown_or_generic_doc_type = 0
    doc_type_counter: Counter[str] = Counter()
    non_list_files = []

    dataset_paths = sorted(glob.glob(os.path.join("dataset", "*.json")))
    for file_path in dataset_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            non_list_files.append(os.path.basename(file_path))

    for file_name, entries in dataset_rows:
        for entry in entries:
            total_rows += 1
            if not isinstance(entry, dict):
                continue

            content = _prepare_content(entry, file_name).strip()
            if not content:
                continue

            ingestable_rows += 1
            if len(content) < 15:
                short_lt15 += 1
            if len(content) < 30:
                short_lt30 += 1

            metadata = normalize_metadata(entry, os.path.join("dataset", file_name))
            document_type = str(metadata.get("document_type", "unknown"))
            doc_type_counter[document_type] += 1

            if document_type.lower() in {"unknown", "document"}:
                unknown_or_generic_doc_type += 1

            if metadata.get("topic") == ["general_trade"] and metadata.get("situation") == ["general"]:
                default_topic_situation += 1

    return {
        "dataset_file_count": len(sorted(glob.glob(os.path.join("dataset", "*.json")))),
        "total_rows": total_rows,
        "ingestable_rows": ingestable_rows,
        "ingestable_pct": round((ingestable_rows / max(total_rows, 1)) * 100, 2),
        "short_lt15_pct": round((short_lt15 / max(ingestable_rows, 1)) * 100, 2),
        "short_lt30_pct": round((short_lt30 / max(ingestable_rows, 1)) * 100, 2),
        "default_topic_situation_pct": round((default_topic_situation / max(ingestable_rows, 1)) * 100, 2),
        "unknown_or_generic_doc_type_count": unknown_or_generic_doc_type,
        "top_document_types": doc_type_counter.most_common(12),
        "non_list_dataset_files": non_list_files,
    }


def _collection_metrics() -> Dict[str, Any]:
    collection = get_or_create_collection()
    count = collection.count()
    raw = collection.get(include=["documents", "metadatas"])

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    if documents and isinstance(documents[0], list):
        documents = documents[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]

    short_lt30 = sum(1 for doc in documents if len((doc or "").strip()) < 30)
    default_topic_situation = sum(
        1
        for metadata in metadatas
        if (metadata or {}).get("topic") == ["general_trade"]
        and (metadata or {}).get("situation") == ["general"]
    )
    generic_doc_type = sum(
        1
        for metadata in metadatas
        if str((metadata or {}).get("document_type", "unknown")).lower() in {"unknown", "document"}
    )

    source_counter: Counter[str] = Counter((metadata or {}).get("source_dataset", "unknown") for metadata in metadatas)
    doc_type_counter: Counter[str] = Counter((metadata or {}).get("document_type", "unknown") for metadata in metadatas)

    dataset_sources = {os.path.basename(path) for path in glob.glob(os.path.join("dataset", "*.json"))}
    indexed_sources = {source for source in source_counter.keys() if source != "unknown"}
    missing_sources = sorted(dataset_sources - indexed_sources)

    return {
        "collection_count": count,
        "short_lt30_pct": round((short_lt30 / max(len(documents), 1)) * 100, 2),
        "default_topic_situation_pct": round((default_topic_situation / max(len(metadatas), 1)) * 100, 2),
        "unknown_or_generic_doc_type_count": generic_doc_type,
        "missing_sources": missing_sources,
        "top_document_types": doc_type_counter.most_common(12),
        "top_sources": source_counter.most_common(12),
    }


def _dictionary_data_quality() -> Dict[str, Any]:
    file_path = os.path.join("dataset", "trade_dictionary_full.json")
    if not os.path.exists(file_path):
        return {"trade_dictionary_empty_rhs": None}

    with open(file_path, "r", encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        return {"trade_dictionary_empty_rhs": None}

    empty_rhs = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        content = row.get("content", "")
        if isinstance(content, str) and "|" in content:
            rhs = content.split("|", 1)[1].strip().strip('"')
            if not rhs:
                empty_rhs += 1

    return {
        "trade_dictionary_empty_rhs_count": empty_rhs,
        "trade_dictionary_total_count": len(rows),
    }


def main() -> None:
    report = {
        "dataset": _dataset_metrics(),
        "collection": _collection_metrics(),
        "dictionary": _dictionary_data_quality(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
