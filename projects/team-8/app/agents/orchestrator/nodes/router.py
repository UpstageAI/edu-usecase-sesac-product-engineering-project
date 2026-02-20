from typing import Literal

from langgraph.graph import END

from app.agents.orchestrator.state.orchestrator_state import OrchestratorState


def route_after_user_input(
    state: OrchestratorState,
) -> Literal["vet_diagnosis", "RAG", "__end__"]:
    """diseases 존재 여부에 따라 다음 노드 결정. 차단 시 즉시 종료."""
    if state.is_blocked:
        print(f"  [Router] BLOCKED: {state.blocked_reason} → END")
        return END
    if state.diseases:
        print("  [Router] diseases 존재 → RAG로 이동")
        return "RAG"
    else:
        print("  [Router] diseases 없음 → vet_diagnosis로 이동")
        return "vet_diagnosis"
