"""
LLM Gateway Port (Interface)

LLM 호출을 추상화하여 비즈니스 로직이 특정 LLM 프레임워크에 의존하지 않도록 함.
"""
from abc import ABC, abstractmethod
from typing import Optional


class LLMGateway(ABC):
    """
    LLM 호출 추상화 인터페이스

    구현체:
        - UpstageLLMGateway: Upstage Solar API
        - OpenAILLMGateway: OpenAI API
        - MockLLMGateway: 테스트용

    Example:
        llm = UpstageLLMGateway(api_key="...")
        response = llm.invoke("Translate: Hello")
        # response = "안녕하세요"
    """

    @abstractmethod
    def invoke(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        LLM에 프롬프트 전송 및 응답 받기

        Args:
            prompt: LLM에 전달할 프롬프트 (시스템 메시지 포함)
            temperature: 생성 온도 (0.0-1.0, None이면 기본값 사용)

        Returns:
            str: LLM 응답 텍스트 (공백 제거됨)

        Raises:
            LLMAPIError: LLM API 호출 실패 시
            LLMTimeoutError: 타임아웃 발생 시
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        현재 사용 중인 모델 이름 반환

        Returns:
            str: 모델 이름 (예: "solar-pro", "gpt-4")
        """
        pass


class LLMAPIError(Exception):
    """LLM API 호출 실패"""
    pass


class LLMTimeoutError(Exception):
    """LLM API 타임아웃"""
    pass
