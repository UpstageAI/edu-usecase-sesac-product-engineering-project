from langgraph.graph import END, START, StateGraph

from .state import JudgeAgentState
from .nodes.validator import validator_node

# ==========================================
#  그래프 정의 (변하지 않음)
# ==========================================
builder = StateGraph(JudgeAgentState)
builder.add_node("validator", validator_node)
builder.add_edge(START, "validator")
builder.add_edge("validator", END)

graph = builder.compile()

# ==========================================
#  메인 실행 블록
# ==========================================
if __name__ == "__main__":
    print("🚀 실행 시작! (이 메시지가 보여야 합니다)")
    from rich import print as rprint

    # [NEW] 목데이터 모듈 Import
    from .mocks.vet_agent_mock import create_mock_vet_agent_state
    from .mocks.rag_mock import get_mock_rag_data

    # 1. Mock 데이터 생성 (함수 호출만 하면 됨)
    vet_mock = create_mock_vet_agent_state()
    rag_mock = get_mock_rag_data()

    # 2. State 조립
    initial_state = {
        "vet_result": vet_mock.model_dump(),  # Pydantic -> Dict 변환
        "retrieved_documents": rag_mock,
        "validation_result": {},
    }

    # 3. 그래프 실행
    print("running...")
    result = graph.invoke(initial_state)

    # 4. 결과 출력
    rprint("\n[최종 검증 결과]")
    rprint(result["validation_result"])
