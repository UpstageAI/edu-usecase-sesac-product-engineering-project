from langgraph.graph import END, START, StateGraph

from app.agents.user_input_template_agent.graph import graph as user_input_graph
from app.agents.user_input_template_agent.state import UserInputTemplateState
from app.agents.vet_agent.graph import graph as vet_graph
from app.agents.vet_agent.state import VetAgentState

builder = StateGraph(
    VetAgentState,
    input_schema=UserInputTemplateState,
    output_schema=VetAgentState,
)

builder.add_node("user_input_template", user_input_graph)
builder.add_node("vet_diagnosis", vet_graph)
builder.add_edge(START, "user_input_template")
builder.add_edge("user_input_template", "vet_diagnosis")
builder.add_edge("vet_diagnosis", END)

graph = builder.compile()

if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.user_input_template_agent.utils.cli import (
        create_arg_parser,
        load_state_from_yaml,
    )

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)
    result = graph.invoke(state)
    rprint(result)
