from typing import TypedDict, List, Dict, Any, Optional

class OrchestratorGraphState(TypedDict, total=False):
    """
    Represents the state of our Orchestrator graph.
    This is the input for each node in the graph.
    """
    session_id: str
    user_input: str
    context: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]
    active_agent: Optional[str]
    agent_specific_state: Dict[str, Any]
    orchestrator_response: Optional[Dict[str, Any]] # The raw output from the selected agent, before final normalization
    llm_intent_classification: Optional[Dict[str, Any]] # Raw output from LLM intent classification for debugging/tracing.
    selected_agent_name: Optional[str] # The name of the agent chosen by the orchestrator for the current turn.
    type: str
    message: str
    report: Optional[Dict[str, Any]]
    meta: Dict[str, Any]
