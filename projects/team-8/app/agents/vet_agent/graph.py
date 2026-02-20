from langgraph.graph import END, START, StateGraph

from app.agents.user_input_template_agent.state import UserInputTemplateState
from app.agents.vet_agent.nodes import vet_diagnosis_node, vet_validation_node
from app.agents.vet_agent.state import VetAgentOutputState, VetAgentState

MAX_RETRIES = 3


def validation_router(state: VetAgentState) -> str:
    """검증 결과에 따라 다음 노드를 결정하는 라우터"""
    if state.is_validated:
        return END
    if state.retry_count >= MAX_RETRIES:
        return END
    return "vet_diagnosis"


builder = StateGraph(
    VetAgentState,
    input_schema=UserInputTemplateState,
    output_schema=VetAgentOutputState,
)

builder.add_node("vet_diagnosis", vet_diagnosis_node)
builder.add_node("vet_validation", vet_validation_node)

builder.add_edge(START, "vet_diagnosis")
builder.add_edge("vet_diagnosis", "vet_validation")
builder.add_conditional_edges("vet_validation", validation_router)

graph = builder.compile()

if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.vet_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)

    # stream으로 실행하여 각 노드의 중간 결과를 출력
    for step in graph.stream(state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        rprint(f"\n{'='*60}")
        rprint(f"[bold]노드: {node_name}[/bold]")
        rprint(f"{'='*60}")

        if node_name == "vet_diagnosis":
            rprint("[bold]진단 결과:[/bold]")
            for i, d in enumerate(node_output.get("diseases", []), 1):
                rprint(f"  {i}. {d.name} (발병률: {d.incidence_rate}, 발병시기: {d.onset_period})")

        elif node_name == "vet_validation":
            is_validated = node_output.get("is_validated", False)
            retry_count = node_output.get("retry_count", 0)
            feedback = node_output.get("validation_feedback", "")

            status = "통과" if is_validated else "실패"
            rprint(f"  검증 결과: {status}")
            rprint(f"  재시도 횟수: {retry_count}/{MAX_RETRIES}")
            if feedback:
                rprint(f"  피드백:\n{feedback}")

    rprint(f"\n{'='*60}")
    rprint("[bold]최종 결과[/bold]")
    rprint(f"{'='*60}")
    final = graph.invoke(state)
    rprint(final)
