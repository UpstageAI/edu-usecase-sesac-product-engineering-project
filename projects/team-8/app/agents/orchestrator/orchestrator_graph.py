from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from rich import print as rprint

from app.agents.utils import create_graph_image, get_parent_path

from app.agents.orchestrator.state.orchestrator_state import OrchestratorState
from app.agents.orchestrator.nodes import route_after_user_input

from app.agents.composer_agent.graph import graph as composer_graph
from app.agents.judge_agent.graph import graph as judge_graph
from app.agents.rag_agent.rag_graph import graph as rag_graph
from app.agents.user_input_template_agent.graph import graph as user_input_graph
from app.agents.user_input_template_agent.utils.cli import (
    create_arg_parser,
    load_state_from_yaml,
    make_config,
)
from app.agents.vet_agent.graph import graph as vet_graph


def save_recommendation(state: OrchestratorState) -> dict:
    """현재 사이클의 retrieved_documents를 recommendation_history에 누적합니다."""

    rprint("state", [a.metadata["evaluation"] for a in state.retrieved_documents])
    return {"recommendation_history": [state.retrieved_documents]}


def build_orchestrator_graph(checkpointer=None):
    graph_builder = StateGraph(OrchestratorState)

    graph_builder.add_node("user_input_template", user_input_graph)
    graph_builder.add_node("vet_diagnosis", vet_graph)
    graph_builder.add_node("RAG", rag_graph)
    graph_builder.add_node("save_recommendation", save_recommendation)
    graph_builder.add_node("judge", judge_graph)
    graph_builder.add_node("composer", composer_graph)

    graph_builder.add_edge(START, "user_input_template")
    graph_builder.add_conditional_edges("user_input_template", route_after_user_input)
    graph_builder.add_edge("vet_diagnosis", "RAG")
    graph_builder.add_edge("RAG", "save_recommendation")
    graph_builder.add_edge("save_recommendation", "judge")
    graph_builder.add_edge("judge", "composer")
    graph_builder.add_edge("composer", END)

    return graph_builder.compile(checkpointer=checkpointer)


in_memory_saver = InMemorySaver()
graph = build_orchestrator_graph(checkpointer=in_memory_saver)
create_graph_image(
    graph,
    file_name="orchestrator_graph",
    base_dir=get_parent_path(__file__),
)


def run_orchestration(yaml_path: str | Path, config: dict) -> dict:
    """
    주어진 YAML 파일 경로에서 사용자의 입력 상태를 불러와
    Orchestrator 그래프 파이프라인을 실행합니다.

    Args:
        yaml_path (str | Path): 사용자 입력 데이터가 저장된 YAML 파일 경로
        config (dict): LangGraph config (thread_id 등 configurable 포함)
            >>> config = {"configurable": {"thread_id": "test_user"}}

    Returns:
        dict: 전체 그래프 실행 후 OrchestratorState 결과를 반환합니다.
    """
    state = load_state_from_yaml(yaml_path, OrchestratorState)
    # YAML에 명시된 필드만 전달하여 체크포인터 상태를 덮어쓰지 않도록 함
    return graph.invoke(
        state.model_dump(exclude_unset=True),
        config=config,
    )


def print_orchestration_result(result: dict) -> None:
    """Orchestrator 실행 결과를 콘솔에 출력합니다."""
    if result.get("is_blocked"):
        rprint(f"[BLOCKED] {result.get('blocked_reason')}")
        return
    # rprint(f"질병 목록: {result['diseases']}")
    # rprint(
    #     "RAG 결과: ",
    #     [doc.page_content for doc in result["retrieved_documents"]],
    # )
    rprint("Judge 검증 결과:", result.get("validation_result"))
    rprint("최종 유저 답변:", result.get("final_message"))


def main():
    args = create_arg_parser().parse_args()
    config = make_config(args.thread_id)
    result = run_orchestration(args.input, config=config)
    print_orchestration_result(result)


if __name__ == "__main__":
    main()


# uv run python -m app.agents.orchestrator.orchestrator_graph
