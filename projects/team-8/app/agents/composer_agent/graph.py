from langgraph.graph import END, START, StateGraph

# [중요] Judge Agent에 있는 State를 가져와서 씁니다
from app.agents.judge_agent.state import JudgeAgentState
from .nodes.writer import writer_node

# ==========================================
#  그래프 빌드
# ==========================================
builder = StateGraph(JudgeAgentState)
builder.add_node("writer", writer_node)
builder.add_edge(START, "writer")
builder.add_edge("writer", END)

graph = builder.compile()

# ==========================================
#  메인 실행 (테스트)
# ==========================================
if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.judge_agent.mocks.vet_agent_mock import create_mock_vet_agent_state
    from .mocks.judge_mock import get_mock_validation_result # 방금 만든 Mock

    print("🚀 [TEST] Composer Agent 실행")

    # 1. Mock 데이터 준비
    # (Judge가 앞단에서 다 처리하고 넘겨줬다고 가정)
    vet_mock = create_mock_vet_agent_state()
    validation_mock = get_mock_validation_result()

    # 2. State 조립
    initial_state = {
        "vet_result": vet_mock.model_dump(),
        "validation_result": validation_mock,
        "retrieved_documents": [], # Writer는 약관 원본 안 봐도 됨 (검증 결과만 봄)
        "final_message": ""
    }

    # 3. 실행
    print("running...")
    result = graph.invoke(initial_state)

    # 4. 결과 출력
    rprint("\n[최종 생성된 답변]")
    rprint(result["final_message"])