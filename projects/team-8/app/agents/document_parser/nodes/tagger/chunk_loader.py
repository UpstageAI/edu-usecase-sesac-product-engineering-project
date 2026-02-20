from __future__ import annotations

import re
from runpy import run_path
from pathlib import Path
from typing import List, Literal

from langchain_core.documents import Document

from app.agents.document_parser.constants import TERMS_DIR

_CHUNK_FILE_PATTERN = re.compile(r"^chunks_(\d{6})_(simple|normal)\.py$")


def load_chunk_files(
    *, file_name: str, tag_type: Literal["normal", "simple"] = "simple"
) -> List[Document]:
    doc_stem = Path(file_name).stem
    chunk_dir = TERMS_DIR / doc_stem / "chunks" / tag_type
    if not chunk_dir.exists():
        return []

    chunk_files = sorted(
        (
            path
            for path in chunk_dir.glob("*.py")
            if _CHUNK_FILE_PATTERN.match(path.name)
        ),
        key=lambda path: int(_CHUNK_FILE_PATTERN.match(path.name).group(1)),
    )

    loaded_chunks: List[Document] = []
    for chunk_file in chunk_files:
        namespace = run_path(str(chunk_file))
        chunk_obj = namespace.get("chunk")
        if isinstance(chunk_obj, Document):
            loaded_chunks.append(chunk_obj)
            continue

        raise ValueError(
            f"Invalid chunk payload: {chunk_file}. Expected 'chunk' to be a Document."
        )

    return loaded_chunks
