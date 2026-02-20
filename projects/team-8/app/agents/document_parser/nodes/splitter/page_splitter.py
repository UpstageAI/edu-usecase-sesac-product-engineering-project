from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup, Tag

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat

from app.agents.document_parser.constants import TERMS_DIR


product_name_by_code = {
    ("meritz", "1"): "메리츠 마음든든 반려동물보험",
    ("meritz", "2"): "무배당 펫퍼민트 Cat&Family보험 다이렉트2601",
    ("meritz", "3"): "무배당 펫퍼민트 Puppy&Family보험 다이렉트2601",
    ("samsung", "1"): "(일반)반려견보험 애니펫",
    ("samsung", "2"): "(일반)반려묘보험 애니펫",
    ("samsung", "3"): "(장기)무배당 삼성화재 다이렉트 착한펫보험(강아지)",
    ("samsung", "4"): "(장기)무배당 삼성화재 펫보험 위풍댕댕",
    ("samsung", "5"): "(장기)무배당 삼성화재 펫보험 의기냥냥",
}

output_extension_by_format = {
    "html": "html",
    "text": "txt",
    "markdown": "md",
}


@dataclass
class PageDoc:
    page_number: int
    html: str
    text: str
    anchor_ids: List[str]


_PAGE_RE = re.compile(r"^\s*(?:-\s*)?(\d{1,4})(?:\s*-\s*)?\s*$")


def _norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_table_text(table_tag: Tag) -> str:
    """
    <table> → 행(row) 단위 텍스트로 변환
    각 셀(cell)은 ' | ' 로 연결
    """
    rows = []

    for tr in table_tag.find_all("tr", recursive=True):
        cells = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            text = cell.get_text(" ", strip=True)
            if text:
                cells.append(text)

        if cells:
            row_text = " | ".join(cells)
            rows.append(row_text)

    return "\n".join(rows)


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def split_pages_and_add_metadata(
    full_document: Document,
    file_name: str,
    *,
    basic_term_start: int,
    basic_term_end: int,
    special_term_start: int,
    special_term_end: int,
    output_format: OutputFormat = "html",
) -> List[Document]:
    """
    Upstage Document Parse 결과(단일 HTML Document)를
    footer의 페이지 마커 기준으로 분리하고
    각 페이지에 문서/약관 메타데이터를 추가한다.

    Args:
        full_document: Upstage Document Parser가 반환한 단일 HTML Document.
        file_name: 원본 PDF 파일명(확장자 포함).
        basic_term_start: 보통약관 시작 페이지 번호(문서 footer 기준).
        basic_term_end: 보통약관 종료 페이지 번호(문서 footer 기준).
        special_term_start: 특별약관 시작 페이지 번호(문서 footer 기준).
        special_term_end: 특별약관 종료 페이지 번호(문서 footer 기준).
        output_format: 페이지별 로컬 저장 파일 포맷(`html`, `text`, `markdown`).

    Returns:
        페이지별로 분리되고 metadata가 추가된 `Document` 리스트.

    Raises:
        ValueError: 필수 인자가 비어 있거나 유효하지 않은 경우.
    """

    if (
        not full_document
        or not file_name
        or not basic_term_start
        or not basic_term_end
        or not special_term_start
        or not special_term_end
    ):
        raise ValueError("❗️split_pages_and_add_metadata invalid params")

    result: List[Document] = []

    full_html = full_document.page_content
    soup = BeautifulSoup(full_html, "html.parser")

    root = soup.body if soup.body else soup

    page_docs: List[PageDoc] = []
    buf_text: List[str] = []
    buf_html: List[str] = []
    buf_ids: List[str] = []

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue

        # -------------------------
        # 1. 페이지 마커 처리
        # -------------------------
        if node.name == "footer":
            raw = node.get_text(strip=True)
            m = _PAGE_RE.match(raw)
            if m:
                page_num = int(m.group(1))
                text = _norm_text("\n".join(buf_text))

                page_docs.append(
                    PageDoc(
                        page_number=page_num,
                        text=text,
                        html="\n".join(buf_html).strip(),
                        anchor_ids=_dedup_keep_order(buf_ids),
                    )
                )

                buf_text.clear()
                buf_html.clear()
                buf_ids.clear()

            continue

        # -------------------------
        # 2. Table 처리 (중복 방지)
        # -------------------------
        if node.name == "table":
            table_text = _extract_table_text(node)
            if table_text:
                buf_text.append(table_text)
            buf_html.append(str(node))

            node_id = node.get("id")
            if node_id:
                buf_ids.append(str(node_id))

            # table 내부는 여기서 이미 처리했으므로
            # descendants 순회 중복 방지 위해 skip
            continue

        # -------------------------
        # 3. 일반 텍스트 태그 처리
        # -------------------------
        if node.name in {
            "p",
            "h1",
            "h2",
            "h3",
            "li",
            "div",
            "span",
        }:
            text = node.get_text(" ", strip=True)
            if text:
                buf_text.append(text)
            buf_html.append(str(node))

            node_id = node.get("id")
            if node_id:
                buf_ids.append(str(node_id))

    # 마지막 페이지 footer가 누락된 경우를 대비한 fallback
    if buf_text:
        inferred_page = page_docs[-1].page_number + 1 if page_docs else 1
        page_docs.append(
            PageDoc(
                page_number=inferred_page,
                text=_norm_text("\n".join(buf_text)),
                html="\n".join(buf_html).strip(),
                anchor_ids=_dedup_keep_order(buf_ids),
            )
        )

    total_pages = len(page_docs)
    last_page = page_docs[-1]
    insurer_code = file_name.split("_")[0]
    product_code = file_name.split("_")[1]
    product_name = product_name_by_code[
        (file_name.split("_")[0], file_name.split("_")[1])
    ]

    for page_doc in page_docs:
        # 정상적으로 페이지 분할이 된 경우
        if last_page.page_number > 1:
            if basic_term_start <= page_doc.page_number <= basic_term_end:
                term_type = "basic"
            elif special_term_start <= page_doc.page_number <= special_term_end:
                term_type = "special"
            else:
                term_type = "unknown"
        # 페이지 분할을 실패한 경우
        else:
            term_type = "unknown"

        new_metadata = {
            "source_doc": {**full_document.metadata},
            "doc": {
                "doc_type": "terms",  # 약관
                "file_name": file_name,  # 약관 파일명 (확장자 포함)
                "insurer_code": insurer_code,  # 보험사 코드 (e.g. samsung, kb, meritz, ...)
                "product_code": product_code,  # 보험 상품 코드 (e.g. 1, 2, 3, ...)
                "product_name": product_name,  # 보험 상품 명 (e.g. 메리츠 마음든든 반려동물보험)
                "total_pages": total_pages,  # 총 페이지 수
                "page": page_doc.page_number,  # 현재 페이지 번호
                # "anchor_ids": page_doc.anchor_ids,  # 파싱 과정에서 식별되는 HTML 태그 ID (e.g. id="23")
            },
            "term_type": term_type,  # 약관 유형 (e.g. basic: 보통 약관, special: 특별 약관, unkown: 식별 불가)'
        }
        new_doc = Document(page_content=page_doc.text, metadata=new_metadata)
        result.append(new_doc)

        target_dir = TERMS_DIR / file_name.split(".")[0] / output_format
        create_page_html_and_text_files(
            page_doc=page_doc,
            target_dir=target_dir,
            output_format=output_format,
            overwrite=True,
        )

    return result


def create_page_html_and_text_files(
    *,
    page_doc: PageDoc,
    target_dir: Path,
    output_format: OutputFormat = "html",
    overwrite: bool = True,
):
    target_dir.mkdir(parents=True, exist_ok=True)

    output_extension = output_extension_by_format.get(output_format)
    if output_extension is None:
        raise ValueError(f"Unsupported output_format: {output_format}")

    file_name = f"{target_dir.parent.name}_page_{page_doc.page_number}"

    page_file_name = f"{file_name}.{output_extension}"
    page_file_path = target_dir / page_file_name
    if overwrite or not page_file_path.exists():
        page_file_path.write_text(page_doc.html, encoding="utf-8")

    text_file_name = f"{file_name}.txt"
    text_file_path = target_dir / text_file_name
    if overwrite or not text_file_path.exists():
        text_file_path.write_text(page_doc.text, encoding="utf-8")
