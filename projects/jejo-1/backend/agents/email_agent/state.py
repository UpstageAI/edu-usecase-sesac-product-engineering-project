from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field # Just in case if BaseModel is needed for complex nested states

# --- Define the State for the Email Agent Graph ---
class EmailGraphState(TypedDict):
    """
    Represents the state of the Email Agent graph.
    """
    # Inputs from Orchestrator
    user_input: str
    conversation_history: List[Dict[str, str]] # Although not used directly by EmailAgent.run, included for consistency
    analysis_in_progress: bool # Although not used directly by EmailAgent.run, included for consistency
    context: Optional[Dict[str, Any]]

    # Email Agent specific components/data (will be initialized in EmailAgentComponents)
    system_prompt: str
    settings: Any # Assuming backend.config.Settings object
    llm: Any # Assuming OpenAI client object

    # Intermediate states of the Email Agent workflow
    retrieved_documents: Optional[List[Dict[str, Any]]]
    used_rag: Optional[bool]
    rag_context_str: Optional[str]
    llm_messages: Optional[List[Dict[str, Any]]]
    llm_raw_response_content: Optional[str]
    llm_parsed_response: Optional[Dict[str, Any]]
    final_response_content: Optional[str] # The user-facing message
    llm_output_details: Optional[Dict[str, Any]]
    model_used: Optional[str]
    email_task_type: Optional[str]
    extracted_email_content: Optional[str]
    extracted_recipient_country: Optional[str]
    extracted_purpose: Optional[str]
    final_metadata: Optional[Dict[str, Any]] # The metadata to be returned

    # Output for Orchestrator
    agent_output_for_orchestrator: Optional[Dict[str, Any]]
