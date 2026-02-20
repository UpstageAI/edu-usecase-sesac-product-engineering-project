import operator
from typing import Annotated, Optional

from langchain_core.documents import Document
from pydantic import Field

from app.agents.judge_agent.state import JudgeAgentState
from app.agents.vet_agent.state import VetAgentState
from app.agents.rag_agent.state.rag_state import RagState


class OrchestratorState(JudgeAgentState, RagState, VetAgentState):
    recommendation_history: Annotated[list[list[Document]], operator.add] = Field(
        default_factory=list, description="사이클별 추천 보험 상품 이력"
    )
    is_blocked: bool = False
    blocked_reason: Optional[str] = None
