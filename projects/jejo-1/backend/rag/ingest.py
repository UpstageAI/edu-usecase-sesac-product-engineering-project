import json
import os
import glob
import argparse
import sys
import hashlib
import time
from typing import Any, Dict, List

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.rag.embedder import get_embedding
from backend.rag.chroma_client import get_or_create_collection, reset_collection
from backend.rag.schema import normalize_metadata
from backend.config import get_settings

DATASET_DIR = "dataset" # Relative to project root
VECTOR_DB_DIR = "backend/vectorstore" # Relative to project root
INGEST_MANIFEST_PATH = os.path.join(VECTOR_DB_DIR, "ingest_manifest.json")


def compute_dataset_fingerprint(dataset_dir: str = DATASET_DIR) -> str:
    """
    Build a stable fingerprint for dataset/*.json files based on filename, size, mtime.
    Used to detect dataset changes and trigger re-ingestion.
    """
    records: List[str] = []
    for file_path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        try:
            stat = os.stat(file_path)
        except OSError:
            continue
        records.append(
            f"{os.path.basename(file_path)}:{stat.st_size}:{int(stat.st_mtime)}"
        )
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return digest


def load_ingest_manifest(manifest_path: str = INGEST_MANIFEST_PATH) -> Dict[str, Any]:
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_ingest_manifest(data: Dict[str, Any], manifest_path: str = INGEST_MANIFEST_PATH) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _parse_pipe_content(content: str) -> str:
    parts = [part.strip().strip('"') for part in content.split("|") if part.strip()]
    if len(parts) <= 1:
        return content.strip()
    return " | ".join(parts)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned)
    return str(value).strip()


def _build_content_from_structured_fields(entry: Dict[str, Any], file_name: str) -> str:
    # Fallback for list entries without explicit `content`.
    priority_fields_by_file = {
        "raw_trade_terms.json": [
            "term",
            "full_name",
            "korean_name",
            "definition",
            "usage",
            "risk_transfer",
            "cost_division",
        ],
        "users_master.json": [
            "user_id",
            "role_level",
            "experience_months",
            "weak_topics",
            "risk_tolerance",
            "preferred_style",
        ],
        "scenarios_master.json": [
            "scenario_id",
            "base_case_id",
            "input_text",
            "expected_answers",
            "context_metadata",
        ],
    }
    fields = priority_fields_by_file.get(file_name, [])
    if not fields:
        return ""

    chunks: List[str] = []
    for field in fields:
        value = entry.get(field)
        text = _to_text(value)
        if text:
            chunks.append(f"{field}: {text}")
    return " | ".join(chunks)


def _enrich_short_content(content: str, entry: Dict[str, Any], file_name: str) -> str:
    """
    Enrich short label-like contents with structured context so embeddings are more informative.
    """
    if len(content.strip()) >= 30:
        return content.strip()

    metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    context_metadata = (
        entry.get("context_metadata", {})
        if isinstance(entry.get("context_metadata"), dict)
        else {}
    )

    context_parts: List[str] = []

    for key in ["category", "risk_type", "situation_context"]:
        text = _to_text(entry.get(key))
        if text:
            context_parts.append(f"{key}: {text}")

    for key in ["document_type", "topic", "situation", "priority", "source", "title"]:
        value = metadata.get(key)
        if value is None:
            value = context_metadata.get(key)
        text = _to_text(value)
        if text:
            context_parts.append(f"{key}: {text}")

    if file_name == "emails.json" and "|" in content:
        parsed = [part.strip().strip('"') for part in content.split("|")]
        parsed = [part for part in parsed if part]
        if len(parsed) >= 2:
            content = (
                f"이메일 사례 - 상황: {parsed[0]}; 대상: {parsed[1]}; "
                f"표현: {parsed[2] if len(parsed) > 2 else ''}"
            ).strip()
    elif file_name == "mistakes.json" and "|" in content:
        parsed = [part.strip() for part in content.split("|") if part.strip()]
        if len(parsed) >= 4:
            content = (
                f"실수 사례 - 실수: {parsed[0]}; 원인: {parsed[1]}; "
                f"리스크: {parsed[2]}; 예방: {parsed[3]}"
            )

    if context_parts:
        return f"{content.strip()} | {' | '.join(context_parts)}".strip()
    return content.strip()


def _prepare_content(entry: Dict[str, Any], file_name: str) -> str:
    content = _to_text(entry.get("content"))
    if content:
        content = _parse_pipe_content(content)
    if not content:
        content = _build_content_from_structured_fields(entry, file_name)
    if not content:
        return ""
    return _enrich_short_content(content, entry, file_name)


def ingest_data(reset: bool = False):
    """
    Ingests data from JSON files in the dataset directory into the Chroma vector store.

    Args:
        reset (bool): If True, resets the Chroma collection before ingestion.
    """
    print("--- Starting Data Ingestion ---")

    # Get Chroma collection
    if reset:
        collection = reset_collection()
    else:
        collection = get_or_create_collection()

    documents_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    total_inserted_count = 0
    indexed_files = set()

    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    if not json_files:
        print(f"No JSON files found in {DATASET_DIR}. Exiting ingestion.")
        return

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing file: {file_name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {file_name} does not contain a list of entries. Skipping.")
            continue

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                print(f"  Skipping row {i} in {file_name}: Non-dict entry.")
                continue

            content = _prepare_content(entry, file_name)
            entry_id = f"{os.path.splitext(file_name)[0]}_{entry.get('id', i)}" # Unique ID
            
            if not content or not content.strip():
                print(f"  Skipping entry {entry_id} in {file_name}: Empty content.")
                continue
            
            # Normalize metadata
            normalized_metadata = normalize_metadata(entry, file_path) # Pass full entry and file_path
            
            # Generate embedding
            embedding = get_embedding(content)
            if embedding is None:
                print(f"  Skipping entry {entry_id} in {file_name}: Failed to generate embedding.")
                continue

            # Add to lists for batch insertion
            documents_to_add.append(content)
            embeddings_to_add.append(embedding) # Store the generated embedding
            metadatas_to_add.append(normalized_metadata)
            ids_to_add.append(entry_id)
            total_inserted_count += 1
            indexed_files.add(file_name)

            if total_inserted_count % 10 == 0:
                print(f"  Processed {total_inserted_count} entries...")
    
    if documents_to_add:
        print(f"\nAdding {len(documents_to_add)} documents to Chroma collection...")
        collection.upsert(
            embeddings=embeddings_to_add, # Use the stored embeddings
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f"  Successfully added {len(documents_to_add)} documents.")
    else:
        print("No documents to add after processing files.")

    print(f"\n--- Data Ingestion Complete ---")
    print(f"Total entries processed: {total_inserted_count}")
    collection_count = collection.count()
    print(f"Current collection count: {collection_count}")

    manifest = {
        "updated_at": int(time.time()),
        "dataset_fingerprint": compute_dataset_fingerprint(DATASET_DIR),
        "indexed_files": sorted(indexed_files),
        "indexed_documents": total_inserted_count,
        "collection_count": collection_count,
        "embedding_provider": get_settings().embedding_provider,
        "local_embedding_dim": get_settings().local_embedding_dim,
    }
    save_ingest_manifest(manifest)
    print(f"Ingestion manifest updated: {INGEST_MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into Chroma vector store.")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma collection before ingestion.")
    args = parser.parse_args()

    ingest_data(reset=args.reset)
#테스트완료
