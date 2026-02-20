"""
퀴즈 품질 검증 도구 (EvalTool)

[역할]
  QuizAgent가 생성한 퀴즈가 RAG 원본 데이터를 올바르게 반영했는지
  문제별로 검증하는 Tool.

[전체 흐름]
  QuizAgent._generate_quiz() 내부에서 호출됨
    1) 각 문제의 question + 정답 선택지로 ChromaDB RAG 검색
    2) 원본 데이터와 퀴즈를 함께 LLM에 전달
    3) LLM이 문제별 is_valid + issues를 배열로 반환
    4) JSON 파싱하여 결과 반환

[이전 방식과 차이]
  - 이전: 독립 에이전트(EvalAgent), 3축 점수(grounding/educational/insight)
  - 현재: Tool 함수, 문제별 is_valid(bool) + issues(list) 배열 반환
"""
import json
import os
from typing import Dict, Any, List

from langsmith import traceable

from backend.utils.llm import call_llm
from backend.rag.retriever import search_with_filter

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'prompts'
)


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _load_prompt(filename: str) -> str:
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_json(raw: str):
    """LLM 응답에서 JSON 배열을 추출.
    앞뒤 텍스트, ```json ... ``` 코드블록 모두 처리.
    """
    from backend.utils.json_utils import safe_json_parse
    return safe_json_parse(raw)


def _format_reference_data(docs: list) -> str:
    """RAG 검색 결과를 LLM 프롬프트에 삽입할 텍스트로 변환."""
    if not docs:
        return "(참고 데이터 없음)"
    lines = []
    for doc in docs:
        content = doc.get("document", "")
        metadata = doc.get("metadata", {})
        if metadata:
            detail = " | ".join(f"{k}: {v}" for k, v in metadata.items() if v)
            lines.append(f"- {content} ({detail})" if detail else f"- {content}")
        else:
            lines.append(f"- {content}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# EvalTool
# ──────────────────────────────────────────────

# 프롬프트 한 번만 로드
_eval_prompt = _load_prompt("eval_prompt.txt")


@traceable(name="eval_tool", run_type="chain")
async def evaluate_quiz_list(
    quiz_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """퀴즈 배열을 검증하고 문제별 결과를 반환한다.

    Args:
        quiz_list: [{"quiz_id": "...", "question": "...", "choices": [...],
                      "answer": 0, "explanation": "...", ...}, ...]

    Returns:
        [{"quiz_id": "...", "is_valid": True/False, "issues": [...]}, ...]
    """
    if not quiz_list:
        return []

    # 1) 각 문제의 정답 텍스트로 RAG 검색 → 원본 데이터 수집
    all_reference_texts = []
    for q in quiz_list:
        question_text = q.get("question", "")
        correct_idx = q.get("answer", 0)
        choices = q.get("choices", [])
        correct_text = choices[correct_idx] if correct_idx < len(choices) else ""
        search_query = f"{question_text} {correct_text}"
        docs = search_with_filter(query=search_query, k=5)
        all_reference_texts.append(_format_reference_data(docs))

    # 2) 프롬프트 조립
    #    각 문제와 해당 RAG 결과를 묶어서 전달
    quiz_with_refs = []
    for i, q in enumerate(quiz_list):
        entry = {**q, "_reference_data": all_reference_texts[i]}
        quiz_with_refs.append(entry)

    quiz_data_text = json.dumps(quiz_with_refs, ensure_ascii=False, indent=2)
    combined_refs = "\n\n".join(
        f"[문제 {i+1}: {q.get('quiz_id', '')}]\n{ref}"
        for i, (q, ref) in enumerate(zip(quiz_list, all_reference_texts))
    )

    prompt = (
        _eval_prompt
        .replace("{quiz_data}", quiz_data_text)
        .replace("{reference_data}", combined_refs)
    )

    # 3) LLM 호출 (temperature=0.3 — 일관된 판단)
    raw = await call_llm(
        user_message="위 퀴즈들을 원본 데이터와 대조하여 문제별로 검증해주세요.",
        system_prompt=prompt,
        temperature=0.3,
    )

    # 4) 파싱
    try:
        results = _parse_json(raw)
        if isinstance(results, dict):
            results = [results]
    except (json.JSONDecodeError, KeyError):
        # 파싱 실패 시 전체 불합격 처리
        return [
            {"quiz_id": q.get("quiz_id", ""), "is_valid": False,
             "issues": ["EvalTool 응답 파싱 실패"]}
            for q in quiz_list
        ]

    # quiz_id가 누락된 결과가 있으면 원본 quiz_list 순서로 매핑
    result_map = {r.get("quiz_id"): r for r in results if r.get("quiz_id")}
    final = []
    for i, q in enumerate(quiz_list):
        qid = q.get("quiz_id", "")
        if qid in result_map:
            r = result_map[qid]
            final.append({
                "quiz_id": qid,
                "is_valid": bool(r.get("is_valid", False)),
                "issues": r.get("issues", []),
            })
        elif i < len(results):
            # quiz_id 매칭 실패 시 순서 기반 매핑
            r = results[i]
            final.append({
                "quiz_id": qid,
                "is_valid": bool(r.get("is_valid", False)),
                "issues": r.get("issues", []),
            })
        else:
            final.append({
                "quiz_id": qid,
                "is_valid": False,
                "issues": ["EvalTool 결과 매핑 실패"],
            })

    return final
