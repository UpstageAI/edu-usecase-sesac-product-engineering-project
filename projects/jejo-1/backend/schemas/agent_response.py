from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ChatResponse(BaseModel):
    """
    모든 에이전트 응답을 통일된 형식으로 반환하기 위한 스키마.
    """
    type: str = Field(..., description="응답의 유형 (예: chat, report, error)")
    message: str = Field(..., description="사용자에게 보여줄 메시지")
    report: Optional[Dict[str, Any]] = Field(None, description="보고서 유형 응답일 경우 상세 데이터")
    meta: Optional[Dict[str, Any]] = Field({}, description="추가 메타데이터")