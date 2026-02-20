from pathlib import Path

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from app.agents.vet_agent.model.model import llm

ENV_PATH = Path(__file__).parents[4] / ".env"
load_dotenv(ENV_PATH)

# 신뢰할 수 있는 수의학 도메인 목록
TRUSTED_DOMAINS = [
    # 국제 수의학 소스
    "ncbi.nlm.nih.gov",  # PubMed (수의학 논문)
    "merckvetmanual.com",  # Merck 수의학 매뉴얼
    "vcahospitals.com",  # VCA 동물병원
    "petmd.com",  # PetMD (수의사 검증 콘텐츠)
    "vet.cornell.edu",  # 코넬대 수의과대학
    # 국내 수의학 소스
    "kvma.or.kr",  # 대한수의사회
    "qia.go.kr",  # 농림축산검역본부
    "vet.anidap.kr",  # 애니답 (수의학 정보 플랫폼)
]

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
    include_domains=TRUSTED_DOMAINS,
)


class BreedTranslation(BaseModel):
    """품종명의 영문 번역 결과"""

    breed_en: str = Field(description="영문 품종명 (예: Chihuahua)")


def _translate_breed(breed: str) -> str:
    """한국어 품종명을 영문으로 번역합니다."""
    structured_llm = llm.with_structured_output(BreedTranslation)
    result = structured_llm.invoke(
        f"Translate the following Korean pet breed name to English.\n"
        f"Breed: {breed}\n"
        f"Use the standard breed name."
    )
    return result.breed_en


def search_breed_diseases(breed: str) -> list[dict]:
    """품종에 흔한 질병 정보를 신뢰할 수 있는 수의학 소스에서 일괄 검색합니다.

    한국어 품종명을 영문으로 번역한 뒤 검색하여
    국제 수의학 소스에서 해당 품종의 흔한 질병 목록을 가져옵니다.

    Args:
        breed: 반려동물 품종 (예: "치와와")

    Returns:
        검색 결과 리스트. 각 항목은 url, content 등을 포함합니다.
        검색 결과가 없으면 빈 리스트를 반환합니다.
    """
    breed_en = _translate_breed(breed)
    query = f"{breed_en} common diseases health problems"
    response = tavily_search.invoke(query)
    if isinstance(response, str):
        return []
    return response.get("results", [])


if __name__ == "__main__":
    from rich import print as rprint

    breed = "치와와"

    # 1. 번역 확인
    breed_en = _translate_breed(breed)
    rprint(f"[bold]번역 결과[/bold]: {breed} -> {breed_en}")

    # 2. 품종 일괄 검색 확인
    results = search_breed_diseases(breed)
    rprint(f"\n[bold]검색 결과 ({len(results)}건)[/bold]:")
    for i, result in enumerate(results, 1):
        rprint(f"\n--- 결과 {i} ---")
        rprint(f"  URL: {result.get('url', 'N/A')}")
        rprint(f"  점수: {result.get('score', 'N/A')}")
        content = result.get("content", "N/A")
        rprint(f"  내용: {content[:300]}...")
