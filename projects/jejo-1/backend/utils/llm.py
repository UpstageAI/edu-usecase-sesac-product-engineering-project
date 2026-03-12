"""
공통 LLM 호출 유틸리티

[역할]
  quiz_agent.py, eval_agent.py 등 여러 에이전트에서 공통으로 사용하는
  Upstage Solar LLM 호출 기능을 제공한다.

[사용 예시]
  from backend.utils.llm import call_llm

  response = await call_llm(
      user_message="무역 퀴즈를 출제해주세요.",
      system_prompt="당신은 무역 전문가입니다.",
      temperature=0.7,  # 높을수록 창의적, 낮을수록 일관적
  )

[API 키 설정]
  .env 파일에 UPSTAGE_API_KEY를 설정해야 한다.
  → backend/config.py의 get_settings()에서 읽어온다.
"""
from functools import lru_cache

from langchain_upstage import ChatUpstage  # Upstage Solar 모델 전용 LangChain 래퍼
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings


@lru_cache()
def get_llm(model: str = "solar-pro", temperature: float = 0.7) -> ChatUpstage:
    """LLM 인스턴스를 생성하고 캐싱하여 반환한다.

    @lru_cache()로 같은 (model, temperature) 조합이면 기존 인스턴스를 재사용한다.
    → 매번 새 인스턴스를 만드는 오버헤드를 방지.

    예: get_llm("solar-pro", 0.7) 을 여러 번 호출해도 실제 생성은 1번만 된다.
    """
    settings = get_settings()
    return ChatUpstage(
        api_key=settings.upstage_api_key,
        model=model,
        temperature=temperature,
    )


async def call_llm(
    user_message: str,
    system_prompt: str = "",
    model: str = "solar-pro",
    temperature: float = 0.7,
) -> str:
    """LLM을 호출하고 응답 텍스트를 반환한다.

    내부 동작:
      1) get_llm()으로 캐싱된 LLM 인스턴스 가져오기
      2) system_prompt가 있으면 SystemMessage로, user_message는 HumanMessage로 구성
      3) ainvoke()로 비동기 호출 → 응답 텍스트 반환

    Args:
        user_message: LLM에게 보낼 사용자 메시지 (필수)
        system_prompt: LLM의 역할/규칙을 지정하는 시스템 프롬프트 (선택)
        model: 사용할 모델명 (기본: solar-pro)
        temperature: 응답의 무작위성 (0.0=결정적, 1.0=창의적, 기본: 0.7)

    Returns:
        LLM 응답 텍스트 (str)
    """
    llm = get_llm(model=model, temperature=temperature)

    # LangChain 메시지 형식으로 구성
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))  # 역할 지정
    messages.append(HumanMessage(content=user_message))        # 실제 요청

    # 비동기 호출 (FastAPI의 async 엔드포인트와 호환)
    response = await llm.ainvoke(messages)
    return response.content
