"""
Upstage LLM Gateway Implementation

Upstage Solar API를 사용한 LLMGateway 구현체.
Retry with exponential backoff 포함.
"""
import logging
from typing import Optional
from langchain_upstage import ChatUpstage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from backend.ports.llm_gateway import LLMGateway, LLMAPIError, LLMTimeoutError


logger = logging.getLogger(__name__)


class UpstageLLMGateway(LLMGateway):
    """
    Upstage Solar API를 사용한 LLM Gateway

    Features:
        - Automatic retry (최대 3회)
        - Exponential backoff (1초 → 2초 → 4초)
        - Timeout handling

    Example:
        gateway = UpstageLLMGateway(api_key="...", model="solar-pro")
        response = gateway.invoke("Translate: Hello")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "solar-pro",
        timeout: int = 30
    ):
        """
        Args:
            api_key: Upstage API 키
            model: 모델 이름 (solar-pro, solar-mini 등)
            timeout: API 호출 타임아웃 (초)
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

        try:
            self._llm = ChatUpstage(
                api_key=api_key,
                model=model,
                timeout=timeout
            )
            logger.info(f"UpstageLLMGateway initialized: model={model}, timeout={timeout}s")
        except Exception as e:
            logger.error(f"Failed to initialize Upstage LLM: {e}")
            raise LLMAPIError(f"LLM initialization failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((LLMAPIError, LLMTimeoutError)),
        reraise=True
    )
    def invoke(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        LLM 호출 (자동 재시도 포함)

        Args:
            prompt: LLM 프롬프트
            temperature: 생성 온도 (0.0-1.0)

        Returns:
            str: LLM 응답 (공백 제거)

        Raises:
            LLMAPIError: API 호출 실패
            LLMTimeoutError: 타임아웃
        """
        try:
            logger.debug(f"Invoking LLM: prompt_length={len(prompt)}, temperature={temperature}")

            # Temperature 설정이 있으면 임시로 LLM 재생성
            if temperature is not None:
                llm = ChatUpstage(
                    api_key=self._api_key,
                    model=self._model,
                    timeout=self._timeout,
                    temperature=temperature
                )
                response = llm.invoke(prompt)
            else:
                response = self._llm.invoke(prompt)

            content = response.content.strip()
            logger.info(f"LLM response received: length={len(content)}")

            return content

        except TimeoutError as e:
            logger.error(f"LLM timeout: {e}")
            raise LLMTimeoutError(f"LLM call timed out after {self._timeout}s")

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise LLMAPIError(f"LLM call failed: {e}")

    def get_model_name(self) -> str:
        """현재 모델 이름 반환"""
        return self._model

    def __repr__(self) -> str:
        return f"<UpstageLLMGateway model={self._model}>"
