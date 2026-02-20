# backend/agents/orchestrator/graph.py

from typing import Callable, Literal
# Assuming langchain_core.graph is used for StateGraph based on LangGraph principles
from langgraph.graph import StateGraph, END

# Internal imports
from .state import OrchestratorGraphState
from .nodes import (
    load_session_state_node,
    detect_intent_and_route_node,
    call_agent_node,
    finalize_and_save_state_node,
    normalize_response_node,
    DEFAULT_AGENT_NAME # To use in conditional edges if needed in the future
)

# Define the graph
def create_orchestrator_graph() -> StateGraph:
    workflow = StateGraph(OrchestratorGraphState)

    # Define nodes
    workflow.add_node("load_session_state", load_session_state_node)
    workflow.add_node("detect_intent", detect_intent_and_route_node)
    workflow.add_node("call_agent", call_agent_node)
    workflow.add_node("finalize_and_save", finalize_and_save_state_node)
    workflow.add_node("normalize_response", normalize_response_node)

    # Define edges
    workflow.set_entry_point("load_session_state")
    
    # After loading session state, detect intent
    workflow.add_edge("load_session_state", "detect_intent")
    
    # After detecting intent, call the selected agent
    # The detect_intent node determines selected_agent_name in the state.
    # The call_agent_node then uses this state variable.
    # All paths from detect_intent lead to call_agent for now.
    # More complex conditional routing can be added here if the graph needs to explicitly branch.
    workflow.add_edge("detect_intent", "call_agent")

    # After calling the agent, finalize and save the session state
    workflow.add_edge("call_agent", "finalize_and_save")

    # After finalizing and saving, normalize the response and end
    workflow.add_edge("finalize_and_save", "normalize_response")
    workflow.add_edge("normalize_response", END) # Final node outputs the normalized response

    return workflow

orchestrator_graph = create_orchestrator_graph()
# The compiled app will be generated and used in the API endpoint (routes.py)
# e.g., compiled_app = orchestrator_graph.compile()
