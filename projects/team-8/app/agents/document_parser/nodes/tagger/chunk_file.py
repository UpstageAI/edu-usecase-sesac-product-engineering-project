from __future__ import annotations

from pathlib import Path
from pprint import pformat

from langchain_core.documents import Document


def create_chunk_file(
    *,
    chunk: Document,
    target_dir: Path,
    output_file_name: str,
    overwrite: bool = False,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file_path = target_dir / output_file_name
    if not overwrite and output_file_path.exists():
        print(f"⚠️ duplicated chunk: {output_file_name}")
        return

    page_content_literal = pformat(chunk.page_content)
    metadata_literal = pformat(chunk.metadata, sort_dicts=False)
    output_file_path.write_text(
        (
            "from langchain_core.documents import Document\n\n"
            "chunk = Document(\n"
            f"    page_content={page_content_literal},\n"
            f"    metadata={metadata_literal},\n"
            ")\n"
        ),
        encoding="utf-8",
    )
