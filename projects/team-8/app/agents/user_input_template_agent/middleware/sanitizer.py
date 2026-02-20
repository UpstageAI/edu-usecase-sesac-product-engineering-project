"""
입력 Sanitization 미들웨어

사용자 입력의 자유 텍스트 필드에서 프롬프트 인젝션 패턴,
위험 특수문자, 과도한 길이 등을 정화(sanitize)한다.
"""

import re
from dataclasses import dataclass, field

from app.agents.user_input_template_agent.state import UserInputTemplateState


# ── 상수 ─────────────────────────────────────────────────
MAX_LEN_BREED = 50
MAX_LEN_HEALTH_FIELD = 200
DEFAULT_BREED = "미상"

# 프롬프트 인젝션 패턴 (한국어)
_INJECTION_PATTERNS_KO: list[str] = [
    r"이전\s*지시를?\s*무시",
    r"시스템\s*프롬프트",
    r"역할을?\s*바꿔",
    r"관리자\s*모드",
    r"너는?\s*이제\s*부터",
    r"지금부터\s*너는",
    r"명령을?\s*무시",
    r"프롬프트를?\s*(보여|알려)",
    r"내부\s*지침",
]

# 프롬프트 인젝션 패턴 (영어)
_INJECTION_PATTERNS_EN: list[str] = [
    r"ignore\s+previous\s+instructions?",
    r"disregard\s+(all\s+)?previous",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"enter\s+admin\s+mode",
    r"forget\s+(your\s+)?instructions?",
    r"override\s+(your\s+)?instructions?",
    r"reveal\s+(your\s+)?(system|internal)",
    r"act\s+as\s+(if\s+)?(you\s+are|a)",
    r"jailbreak",
    r"DAN\s+mode",
]

# 모든 인젝션 패턴을 하나의 정규식으로 결합
_INJECTION_REGEX = re.compile(
    "|".join(_INJECTION_PATTERNS_KO + _INJECTION_PATTERNS_EN),
    re.IGNORECASE,
)

# 제어 문자 (탭·줄바꿈 제외)
_CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 연속 공백 정규화
_MULTI_SPACE_REGEX = re.compile(r" {2,}")


# ── 데이터 클래스 ────────────────────────────────────────
@dataclass
class SanitizeLog:
    """개별 필드의 정화 로그"""

    field: str
    original: str
    sanitized: str | None
    actions: list[str] = field(default_factory=list)


@dataclass
class SanitizeResult:
    """전체 정화 결과"""

    state_dict: dict
    logs: list[SanitizeLog] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return len(self.logs) > 0

    @property
    def has_injection(self) -> bool:
        """인젝션 패턴이 감지된 로그가 있는지 확인"""
        return any(
            "injection_detected_field_nullified" in log.actions
            for log in self.logs
        )


# ── 핵심 함수 ───────────────────────────────────────────
def sanitize_text(
    text: str, max_length: int = 200
) -> tuple[str | None, list[str]]:
    """
    개별 텍스트를 정화한다.

    프롬프트 인젝션이 감지되면 필드 전체를 신뢰할 수 없으므로
    None을 반환하여 필드를 비운다.
    그 외 경미한 위반(제어 문자, 공백, 길이 초과)은 정화 후 계속 진행한다.

    Args:
        text: 정화 대상 텍스트
        max_length: 최대 허용 길이

    Returns:
        (정화된 텍스트 또는 None, 적용된 액션 목록)
    """
    if not text or not isinstance(text, str):
        return text, []

    actions: list[str] = []
    result = text

    # 1. 프롬프트 인젝션 패턴 감지 → 필드 전체 비움
    if _INJECTION_REGEX.search(result):
        actions.append("injection_detected_field_nullified")
        return None, actions

    # 2. 제어 문자 제거
    cleaned = _CONTROL_CHAR_REGEX.sub("", result)
    if cleaned != result:
        actions.append("control_char_removed")
        result = cleaned

    # 3. 연속 공백 정규화
    cleaned = _MULTI_SPACE_REGEX.sub(" ", result).strip()
    if cleaned != result:
        actions.append("whitespace_normalized")
        result = cleaned

    # 4. 길이 제한
    if len(result) > max_length:
        result = result[:max_length]
        actions.append(f"truncated_to_{max_length}")

    return result, actions


def sanitize_user_input(
    state: UserInputTemplateState,
) -> SanitizeResult:
    """
    UserInputTemplateState의 자유 텍스트 필드를 정화한다.

    대상 필드:
        - breed (품종)
        - health_condition.frequent_illness_area (자주 아픈 부위)
        - health_condition.disease_surgery_history (질병/수술 이력)

    Args:
        state: 정화 대상 사용자 입력 상태

    Returns:
        SanitizeResult: 정화된 state dict와 로그
    """
    state_dict = state.model_dump(exclude_none=True)
    logs: list[SanitizeLog] = []

    # ── breed 정화 (필수값: 인젝션 시 기본값 대체) ──
    if state.breed is not None:
        sanitized, actions = sanitize_text(state.breed, max_length=MAX_LEN_BREED)
        if actions:
            # 인젝션 감지로 None이 반환되면 기본값으로 대체
            replacement = sanitized if sanitized is not None else DEFAULT_BREED
            logs.append(
                SanitizeLog(
                    field="breed",
                    original=state.breed,
                    sanitized=replacement,
                    actions=actions,
                )
            )
            state_dict["breed"] = replacement

    # ── health_condition 정화 (선택값: 인젝션 시 None) ──
    if state.health_condition is not None:
        hc = state.health_condition

        # frequent_illness_area
        if hc.frequent_illness_area is not None:
            sanitized, actions = sanitize_text(
                hc.frequent_illness_area, max_length=MAX_LEN_HEALTH_FIELD
            )
            if actions:
                logs.append(
                    SanitizeLog(
                        field="health_condition.frequent_illness_area",
                        original=hc.frequent_illness_area,
                        sanitized=sanitized,
                        actions=actions,
                    )
                )
                if sanitized is None:
                    state_dict["health_condition"].pop(
                        "frequent_illness_area", None
                    )
                else:
                    state_dict["health_condition"][
                        "frequent_illness_area"
                    ] = sanitized

        # disease_surgery_history
        if hc.disease_surgery_history is not None:
            sanitized, actions = sanitize_text(
                hc.disease_surgery_history, max_length=MAX_LEN_HEALTH_FIELD
            )
            if actions:
                logs.append(
                    SanitizeLog(
                        field="health_condition.disease_surgery_history",
                        original=hc.disease_surgery_history,
                        sanitized=sanitized,
                        actions=actions,
                    )
                )
                if sanitized is None:
                    state_dict["health_condition"].pop(
                        "disease_surgery_history", None
                    )
                else:
                    state_dict["health_condition"][
                        "disease_surgery_history"
                    ] = sanitized

    return SanitizeResult(state_dict=state_dict, logs=logs)
