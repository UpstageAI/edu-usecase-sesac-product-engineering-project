import json

from pydantic import BaseModel, Field

from app.agents.vet_agent.cache import get_cached_diseases, set_cached_diseases
from app.agents.vet_agent.model.model import llm
from app.agents.vet_agent.state import VetAgentState
from app.agents.vet_agent.tools.search import search_breed_diseases

# 검증 통과 기준 (일치율)
VALIDATION_THRESHOLD = 0.5


# --- 구조화 출력 스키마 ---


class ExtractedDiseases(BaseModel):
    """검색 결과에서 추출한 질병 목록"""

    diseases: list[str] = Field(
        description="검색 결과에서 언급된 해당 품종의 흔한 질병명 (영문)"
    )


class DiseaseMatch(BaseModel):
    """개별 질병의 매칭 결과"""

    disease_ko: str = Field(description="한국어 질병명")
    matched_disease_en: str = Field(
        description="매칭된 영문 질병명. 매칭 없으면 빈 문자열."
    )
    is_matched: bool = Field(description="매칭 여부")


class MatchResult(BaseModel):
    """전체 대조 결과"""

    matches: list[DiseaseMatch] = Field(description="각 질병의 매칭 결과")


# --- 내부 함수 ---


def _extract_diseases_from_search(search_results: list[dict]) -> list[str]:
    """검색 결과 페이지 내용에서 질병명 리스트를 추출합니다. (LLM 1회)"""
    contents = "\n\n".join(
        f"[{r.get('url', '')}]\n{r.get('content', '')}" for r in search_results
    )
    structured_llm = llm.with_structured_output(ExtractedDiseases)
    result = structured_llm.invoke(
        f"Extract all disease/health condition names mentioned "
        f"in the following veterinary search results.\n"
        f"Return only the English disease names as a list.\n\n"
        f"{contents}"
    )
    return result.diseases


def _match_diseases(
    diagnosis_diseases: list[str], search_diseases: list[str]
) -> MatchResult:
    """LLM 진단 질병(한국어)과 검색 추출 질병(영문)을 대조합니다. (LLM 1회)"""
    structured_llm = llm.with_structured_output(MatchResult)
    result = structured_llm.invoke(
        f"Match each Korean disease name to its English equivalent "
        f"from the search list.\n\n"
        f"Korean diseases (from LLM diagnosis):\n"
        f"{json.dumps(diagnosis_diseases, ensure_ascii=False)}\n\n"
        f"English diseases (from search results):\n"
        f"{json.dumps(search_diseases, ensure_ascii=False)}\n\n"
        f"For each Korean disease, find the matching English disease. "
        f"If no match exists, set is_matched=False and matched_disease_en=''."
    )
    return result


def _build_feedback(match_result: MatchResult, search_diseases: list[str]) -> str:
    """검증 실패 시 피드백을 생성합니다."""
    unmatched = [m.disease_ko for m in match_result.matches if not m.is_matched]
    feedback_lines = []
    for name in unmatched:
        feedback_lines.append(
            f"- '{name}'은(는) 해당 품종의 주요 질병 근거를 찾을 수 없습니다."
        )
    feedback_lines.append(
        f"\n검색에서 확인된 해당 품종의 질병: {', '.join(search_diseases)}"
    )
    feedback_lines.append("위 질병을 참고하여 진단 결과를 수정해주세요.")
    return "\n".join(feedback_lines)


# --- 노드 함수 ---


def vet_validation_node(state: VetAgentState) -> dict:
    """LLM 진단 결과를 검색 근거와 교차 대조하여 검증하는 노드"""

    breed = state.breed or ""

    # 1. 캐시 조회
    cached = get_cached_diseases(breed)

    if cached is not None:
        # 캐시 히트: 검색 + 추출 건너뜀
        search_diseases = cached
    else:
        # 캐시 미스: 검색 + 추출 수행
        search_results = search_breed_diseases(breed)

        # 검색 결과 없음 -> 검증 건너뜀
        if not search_results:
            return {
                "is_validated": True,
                "validation_feedback": "검색 근거를 찾을 수 없어 검증을 건너뜁니다.",
                "retry_count": state.retry_count + 1,
            }

        # 2. 검색 결과에서 질병 목록 추출 (LLM 1회)
        search_diseases = _extract_diseases_from_search(search_results)

        # 캐시 저장
        set_cached_diseases(breed, "", search_diseases)

    # 3. LLM 진단 질병 vs 검색 추출 질병 대조 (LLM 1회)
    diagnosis_names = [d.name for d in state.diseases]
    match_result = _match_diseases(diagnosis_names, search_diseases)

    # 4. 일치율 계산 및 판정
    total = len(match_result.matches)
    matched = sum(1 for m in match_result.matches if m.is_matched)
    match_ratio = matched / total if total > 0 else 0

    is_validated = match_ratio >= VALIDATION_THRESHOLD
    validation_feedback = ""
    if not is_validated:
        validation_feedback = _build_feedback(match_result, search_diseases)

    return {
        "is_validated": is_validated,
        "validation_feedback": validation_feedback,
        "retry_count": state.retry_count + 1,
    }


if __name__ == "__main__":
    from rich import print as rprint

    from app.agents.vet_agent.nodes.vet_diagnosis_node import vet_diagnosis_node
    from app.agents.vet_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, VetAgentState)

    # 진단 노드를 먼저 실행하여 diseases를 채움
    diagnosis_result = vet_diagnosis_node(state)
    rprint("[bold]진단 결과:[/bold]")
    rprint(diagnosis_result)

    state = state.model_copy(update=diagnosis_result)

    # 검증 노드 실행
    validation_result = vet_validation_node(state)
    rprint("\n[bold]검증 결과:[/bold]")
    rprint(validation_result)
