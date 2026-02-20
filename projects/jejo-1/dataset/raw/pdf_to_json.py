"""
Upstage Document Parse API로 PDF를 파싱하여 JSON으로 변환

실행:
    uv run python dataset/raw/pdf_to_json.py
"""
import os
import sys
import json
import requests
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.config import get_settings

PDF_PATH = "dataset/raw/국제 상업회의소 무역용어집.pdf"
OUTPUT_PATH = "dataset/icc_trade_terms.json"

UPSTAGE_API_URL = "https://api.upstage.ai/v1/document-digitization"
MAX_PAGES_PER_REQUEST = 100


def split_pdf(pdf_path: str, max_pages: int) -> list:
    """PDF를 max_pages 단위로 나눠서 임시 파일 경로 리스트 반환."""
    import fitz
    doc = fitz.open(pdf_path)
    total = doc.page_count
    parts = []
    for start in range(0, total, max_pages):
        end = min(start + max_pages, total)
        part_doc = fitz.open()
        part_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        part_path = f"{pdf_path}_part{start+1}-{end}.pdf"
        part_doc.save(part_path)
        part_doc.close()
        parts.append(part_path)
        print(f"  분할 저장: {part_path} ({start+1}~{end}페이지)")
    doc.close()
    return parts


def call_api(pdf_path: str, api_key: str) -> str:
    """Upstage Document Parse API 단일 호출."""
    with open(pdf_path, "rb") as f:
        response = requests.post(
            UPSTAGE_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"document": f},
            data={"output_formats": '["text"]', "model": "document-parse"},
            timeout=300,
        )
    if response.status_code != 200:
        print(f"API 오류: {response.status_code} — {response.text[:300]}")
        return ""
    result = response.json()
    content = result.get("content", {})
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def parse_pdf_with_upstage(pdf_path: str, api_key: str) -> str:
    """PDF가 100페이지 초과면 분할 후 각각 API 호출, 텍스트 합쳐서 반환."""
    import fitz
    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()
    print(f"총 페이지: {total}")

    if total <= MAX_PAGES_PER_REQUEST:
        return call_api(pdf_path, api_key)

    # 분할 처리
    print(f"100페이지 초과 - {MAX_PAGES_PER_REQUEST}페이지씩 분할 처리")
    part_paths = split_pdf(pdf_path, MAX_PAGES_PER_REQUEST)
    all_text = []
    for i, part_path in enumerate(part_paths):
        print(f"\n파트 {i+1}/{len(part_paths)} API 호출 중...")
        text = call_api(part_path, api_key)
        print(f"  추출 텍스트: {len(text)}자")
        all_text.append(text)
        os.remove(part_path)  # 임시 파일 삭제

    combined = "\n".join(all_text)
    print(f"\n전체 추출 텍스트: {len(combined)}자")
    return combined


def _make_entry(idx: int, term: str, desc_lines: list) -> dict | None:
    """용어+설명 라인으로 엔트리 딕셔너리 생성. 설명이 짧으면 None 반환."""
    description = " ".join(desc_lines).strip()
    if len(description) < 20:
        return None
    content = f"{term} | {description}"
    return {
        "id": idx,
        "category": "trade_terminology",
        "content": content,
        "metadata": {
            "title": term,
            "document_type": "trade_terminology",
            "category": "icc_trade_terms",
            "priority": "high",
            "source": "국제상업회의소 무역용어집"
        }
    }


def text_to_entries(text: str) -> list:
    """추출된 텍스트를 용어-설명 쌍의 JSON 엔트리로 변환.

    전략:
    1. 목차 섹션을 건너뛰고 실제 본문 시작점(첫 번째 【...】 섹션 이후)부터 파싱.
    2. 용어 라인 조건:
       - 길이 < 100자
       - 숫자로 끝나지 않음 (목차 페이지 번호 제외)
       - 페이지 구분선(-N-), 섹션 헤더(【), 표 행(|) 제외
       - 번호 항목(①②, (1)) 제외
       - 최소 3자 이상의 한글 또는 영문 포함
    """
    entries = []
    idx = 1

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── 1) 본문 시작점 탐색:
    #    목차는 "-9-" 페이지 구분선으로 끝나고 "-11-" 이후부터 실제 본문이 시작됨.
    #    "-9-" 이후 등장하는 첫 번째 "-11-" 위치를 찾고 그 다음 줄부터 파싱.
    content_start = 0
    found_toc_end = False
    for i, line in enumerate(lines):
        if line == "-9-":
            found_toc_end = True
        if found_toc_end and line == "-11-":
            content_start = i + 1
            break
    # 폴백: "-11-"이 없으면 "【"로 시작하는 첫 줄 사용
    if content_start == 0:
        for i, line in enumerate(lines):
            if line.startswith("【"):
                content_start = i + 1
                break

    # ── 2) 스킵 패턴
    skip_patterns = [
        re.compile(r'^-\d+-$'),           # 페이지 구분선: -13-
        re.compile(r'^\|'),               # 마크다운 표
        re.compile(r'^[①②③④⑤⑥⑦⑧⑨⑩]'),  # 원문자 번호
        re.compile(r'^\(\d+\)'),          # (1) (2) 번호
        re.compile(r'^【'),               # 섹션 헤더
    ]

    # ── 3) 용어 라인 판별
    #   핵심 규칙: 대부분의 무역 용어는 "한글(漢字, English)" 형태로 괄호로 끝남
    #   보조 규칙: FEU, TEU, Depot 같은 짧은 영문 약어도 포함
    def is_term_line(line: str) -> bool:
        if len(line) > 80 or len(line) < 2:
            return False
        for pat in skip_patterns:
            if pat.match(line):
                return False
        # 목차 페이지 번호 제외: "용어명 13"
        if re.search(r'\s+\d+$', line):
            return False
        # 번호 항목 제외
        if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩\(\d]', line):
            return False

        # 1순위: 괄호로 끝나는 라인 → 용어
        if line.endswith(')'):
            return True

        # 2순위: 괄호 없는 짧은 영문 약어 또는 영문+한글 명사구
        #   예: "FEU", "TEU", "Depot", "Knock-Down 수출", "Piggy Back 운송"
        if len(line) <= 30 and re.match(r'^[A-Za-z]', line):
            if not line.endswith('.') and not line.endswith(','):
                return True

        return False

    # ── 4) 파싱
    current_term = None
    current_desc_lines = []

    for line in lines[content_start:]:
        # 스킵 대상
        skip = any(pat.match(line) for pat in skip_patterns)
        if skip:
            if current_term:
                current_desc_lines.append(line) if not any(pat.match(line) for pat in skip_patterns[:1]) else None
            continue

        if is_term_line(line):
            # 이전 용어 저장
            if current_term and current_desc_lines:
                entry = _make_entry(idx, current_term, current_desc_lines)
                if entry:
                    entries.append(entry)
                    idx += 1
            current_term = line
            current_desc_lines = []
        else:
            if current_term:
                current_desc_lines.append(line)

    # 마지막 용어 저장
    if current_term and current_desc_lines:
        entry = _make_entry(idx, current_term, current_desc_lines)
        if entry:
            entries.append(entry)

    return entries


def main():
    settings = get_settings()
    if not settings.upstage_api_key:
        print("오류: .env 파일에 UPSTAGE_API_KEY가 없습니다.")
        sys.exit(1)

    # 1) PDF 파싱
    text = parse_pdf_with_upstage(PDF_PATH, settings.upstage_api_key)
    if not text:
        print("텍스트 추출 실패.")
        sys.exit(1)

    # 원본 텍스트 저장 (디버그용)
    raw_txt_path = "dataset/raw/icc_raw_text.txt"
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"원본 텍스트 저장: {raw_txt_path}")

    # 2) 텍스트 → JSON 엔트리 변환
    entries = text_to_entries(text)
    print(f"변환된 엔트리 수: {len(entries)}")

    if entries:
        # 샘플 출력
        print("\n--- 샘플 엔트리 ---")
        for e in entries[:3]:
            print(json.dumps(e, ensure_ascii=True, indent=2))

    # 3) JSON 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUTPUT_PATH} ({len(entries)}개 엔트리)")


if __name__ == "__main__":
    main()
