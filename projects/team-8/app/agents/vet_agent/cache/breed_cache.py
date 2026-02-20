import json
from datetime import datetime, timedelta
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "breed_diseases.json"
CACHE_TTL_DAYS = 30


def load_cache() -> dict:
    """JSON 파일에서 캐시를 로드합니다. 파일이 없으면 빈 dict를 반환합니다."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict) -> None:
    """캐시를 JSON 파일에 저장합니다."""
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_cached_diseases(breed: str) -> list[str] | None:
    """품종명으로 캐시된 질병 목록을 조회합니다.

    Args:
        breed: 한국어 품종명 (예: "치와와")

    Returns:
        캐시된 영문 질병 목록. 캐시 미스 또는 TTL 만료 시 None.
    """
    cache = load_cache()
    entry = cache.get(breed)
    if entry is None:
        return None

    # TTL 체크
    cached_at = datetime.fromisoformat(entry["cached_at"])
    if datetime.now() - cached_at > timedelta(days=CACHE_TTL_DAYS):
        return None

    return entry["search_diseases"]


def set_cached_diseases(
    breed: str, breed_en: str, diseases: list[str]
) -> None:
    """검색 + 추출 결과를 캐시에 저장합니다.

    Args:
        breed: 한국어 품종명 (예: "치와와")
        breed_en: 영문 품종명 (예: "Chihuahua")
        diseases: 추출된 영문 질병 목록
    """
    cache = load_cache()
    cache[breed] = {
        "breed_en": breed_en,
        "search_diseases": diseases,
        "cached_at": datetime.now().isoformat(),
    }
    save_cache(cache)


if __name__ == "__main__":
    from rich import print as rprint

    # 캐시 저장 테스트
    set_cached_diseases(
        breed="치와와",
        breed_en="Chihuahua",
        diseases=["periodontal disease", "patellar luxation", "heart disease"],
    )
    rprint("[bold]캐시 저장 완료[/bold]")

    # 캐시 조회 테스트
    result = get_cached_diseases("치와와")
    rprint(f"[bold]캐시 조회[/bold]: {result}")

    # 캐시 미스 테스트
    result_miss = get_cached_diseases("골든 리트리버")
    rprint(f"[bold]캐시 미스[/bold]: {result_miss}")

    # 캐시 파일 내용 확인
    rprint(f"\n[bold]캐시 파일 내용:[/bold]")
    rprint(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
