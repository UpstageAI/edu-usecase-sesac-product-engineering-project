# backend/agents/quiz_agent/graph.py

from langgraph.graph import StateGraph, END

# Internal imports
from .state import QuizGraphState
from .nodes import (
    perform_rag_search_node,
    prepare_llm_messages_node,
    call_llm_and_parse_response_node,
    format_quiz_output_node,
)

# Define the graph
def create_quiz_agent_graph() -> StateGraph:
    workflow = StateGraph(QuizGraphState)

    # Define nodes
    workflow.add_node("perform_rag_search", perform_rag_search_node)
    workflow.add_node("prepare_llm_messages", prepare_llm_messages_node)
    workflow.add_node("call_llm_and_parse_response", call_llm_and_parse_response_node)
    workflow.add_node("format_output", format_quiz_output_node)

    # Define edges (linear workflow)
    workflow.set_entry_point("perform_rag_search")
    workflow.add_edge("perform_rag_search", "prepare_llm_messages")
    workflow.add_edge("prepare_llm_messages", "call_llm_and_parse_response")
    workflow.add_edge("call_llm_and_parse_response", "format_output")
    workflow.add_edge("format_output", END)

    return workflow

quiz_agent_graph = create_quiz_agent_graph()
# compiled_quiz_agent_app = quiz_agent_graph.compile()
