"""
API Routes
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.schemas.agent_response import ChatResponse # Import the new ChatResponse schema
from backend.agents.orchestrator.graph import orchestrator_graph # Import the orchestrator graph
from backend.agents.orchestrator.state import OrchestratorGraphState # Import the state definition
from backend.core.response_converter import normalize_response

router = APIRouter()

# Compile the orchestrator graph globally
compiled_orchestrator_app = orchestrator_graph.compile()


class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: str # Added session_id
    message: str
    context: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - routes to appropriate agent based on intent
    """
    # Initial state for the graph
    initial_state: OrchestratorGraphState = {
        "session_id": request.session_id,
        "user_input": request.message,
        "context": request.context if request.context is not None else {},
        "conversation_history": [], # Will be loaded by load_session_state_node
        "active_agent": None,      # Will be loaded by load_session_state_node
        "agent_specific_state": {},# Will be loaded by load_session_state_node
        "orchestrator_response": None,
        "llm_intent_classification": None,
        "selected_agent_name": None
    }
    
    # Invoke the compiled graph
    orchestrator_result = await compiled_orchestrator_app.ainvoke(initial_state)
    
    # The orchestrator_result is the final output of the graph (normalize_response_node)
    if (
        isinstance(orchestrator_result, dict)
        and "type" in orchestrator_result
        and "message" in orchestrator_result
    ):
        return ChatResponse(**orchestrator_result)

    # Backward-safe fallback: normalize from orchestrator_response when graph
    # returns full state dict without final response fields.
    normalized = normalize_response(
        orchestrator_result.get("orchestrator_response", orchestrator_result)
        if isinstance(orchestrator_result, dict)
        else orchestrator_result
    )
    return ChatResponse(**normalized)
