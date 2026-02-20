"""
Intent Classifier - 사용자 의도 분류

책임:
- 사용자 입력을 5가지 의도로 분류 (quiz, email_coach, risk_detect, general_chat, out_of_scope)
- LLM 기반 Few-shot 분류
- 프롬프트 템플릿 로딩
"""

import logging
import re
from typing import Literal, Dict, Any
from pathlib import Path

from backend.ports import LLMGateway


class IntentClassifier:
    """LLM 기반 의도 분류기"""

    # 5가지 의도 타입
    INTENTS = Literal["quiz", "email_coach", "risk_detect", "general_chat", "out_of_scope"]

    def __init__(self, llm: LLMGateway):
        """
        Args:
            llm: LLM Gateway
        """
        self._llm = llm
        self._logger = logging.getLogger(__name__)
        self._prompt_template = self._load_prompt()

    def classify(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        사용자 입력을 5가지 의도로 분류

        Args:
            user_input: 사용자 입력 텍스트
            context: 세션 컨텍스트 (사용 안 함, 향후 확장용)

        Returns:
            "quiz" | "email_coach" | "risk_detect" | "general_chat" | "out_of_scope"
        """
        try:
            # 프롬프트 생성
            prompt = self._build_classification_prompt(user_input)

            # LLM 호출
            response = self._llm.invoke(prompt, temperature=0.0)

            # 응답 파싱
            intent = self._parse_intent(response)

            self._logger.info(f"Intent classified: {user_input[:50]} -> {intent}")
            return intent

        except Exception as e:
            self._logger.error(f"Intent classification error: {e}")
            # 폴백: general_chat
            return "general_chat"

    def _load_prompt(self) -> str:
        """프롬프트 템플릿 로딩"""
        try:
            prompt_path = Path("backend/prompts/intent_classification_prompt.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self._logger.error(f"Prompt loading error: {e}")
            # 폴백: 간단한 기본 프롬프트
            return """사용자 입력을 다음 중 하나로 분류하세요:
quiz, email_coach, risk_detect, general_chat, out_of_scope

사용자 입력: {user_input}
분류:"""

    def _build_classification_prompt(self, user_input: str) -> str:
        """Few-shot 프롬프트 생성"""
        return self._prompt_template.format(user_input=user_input)

    def _parse_intent(self, response: str) -> str:
        """
        LLM 응답에서 의도 추출

        예상 형식: "분류: email_coach"

        Args:
            response: LLM 응답

        Returns:
            추출된 의도
        """
        # "분류: " 패턴 찾기
        match = re.search(r'분류:\s*(\w+)', response, re.IGNORECASE)
        if match:
            intent = match.group(1).strip().lower()
            # 유효한 의도인지 확인
            valid_intents = ["quiz", "email_coach", "risk_detect", "general_chat", "out_of_scope"]
            if intent in valid_intents:
                return intent

        # 폴백: 응답 텍스트에서 키워드 직접 찾기
        response_lower = response.lower()
        if "email_coach" in response_lower:
            return "email_coach"
        elif "quiz" in response_lower:
            return "quiz"
        elif "risk_detect" in response_lower:
            return "risk_detect"
        elif "out_of_scope" in response_lower:
            return "out_of_scope"
        else:
            return "general_chat"
