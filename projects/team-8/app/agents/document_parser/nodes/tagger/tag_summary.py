from __future__ import annotations

from collections import Counter
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Iterable, List, Literal

from langchain_core.documents import Document

from app.agents.document_parser.constants import TERMS_DIR


def summarize_counts(
    tagged_chunks: List[Document],
    *,
    tag_type: str = Literal["normal", "simple"],
    clause_types: Iterable[str],
    term_types: Iterable[str],
) -> Dict[str, Dict[str, int]]:
    clause_summary = summarize_clause_type_counts(
        tagged_chunks, clause_types, save_file=False
    )
    term_summary = summarize_term_type_counts(
        tagged_chunks, term_types, save_file=False
    )
    combined_summary = {
        "clause_type": clause_summary,
        "term_type": term_summary,
    }

    if tagged_chunks:
        target_dir = (
            TERMS_DIR / tagged_chunks[0].metadata["doc"]["file_name"].split(".")[0]
        )
        create_combined_summary_file(combined_summary, target_dir, tag_type)

    return combined_summary


def summarize_term_type_counts(
    tagged_chunks: List[Document], term_types: Iterable[str], save_file: bool = True
) -> Dict[str, int]:
    return _summarize_label_counts(
        tagged_chunks=tagged_chunks,
        labels=term_types,
        summary_type="term_type",
        value_getter=lambda chunk: chunk.metadata.get("term_type") or "other",
        save_file=save_file,
    )


def summarize_clause_type_counts(
    tagged_chunks: List[Document], clause_types: Iterable[str], save_file: bool = True
) -> Dict[str, int]:
    return _summarize_label_counts(
        tagged_chunks=tagged_chunks,
        labels=clause_types,
        summary_type="clause_type",
        value_getter=lambda chunk: chunk.metadata.get("clause", {}).get("clause_type"),
        save_file=save_file,
    )


def _summarize_label_counts(
    *,
    tagged_chunks: List[Document],
    labels: Iterable[str],
    summary_type: str,
    value_getter: Callable[[Document], str | None],
    save_file: bool = True,
) -> Dict[str, int]:
    if not tagged_chunks:
        return {"total": 0}

    summary: Dict[str, int] = {"total": len(tagged_chunks)}
    label_counter = Counter(
        value for value in (value_getter(chunk) for chunk in tagged_chunks) if value
    )

    for label in labels:
        summary[label] = label_counter.get(label, 0)

    for label, count in label_counter.items():
        if label not in summary:
            summary[label] = count

    if save_file:
        target_dir = (
            TERMS_DIR / tagged_chunks[0].metadata["doc"]["file_name"].split(".")[0]
        )
        create_summary_file(summary, target_dir, summary_type=summary_type)

    return summary


def create_summary_file(
    summary: Dict[str, int],
    target_dir: Path,
    summary_type: str = "unkown_type",
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file_name = f"{target_dir.name}_{summary_type}_summary.py"
    output_file_path = target_dir / "chunks" / output_file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    summary_literal = pformat(summary, sort_dicts=False)
    output_file_path.write_text(f"summary = {summary_literal}\n", encoding="utf-8")


def create_combined_summary_file(
    summary: Dict[str, Dict[str, int]],
    target_dir: Path,
    tag_type: str = Literal["normal", "simple"],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file_name = f"chunks_{tag_type}_summary.py"
    output_file_path = target_dir / "chunks" / tag_type / output_file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    summary_literal = pformat(summary, sort_dicts=False)
    output_file_path.write_text(f"summary = {summary_literal}\n", encoding="utf-8")
