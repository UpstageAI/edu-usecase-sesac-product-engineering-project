from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field # Just in case if BaseModel is needed for complex nested states

# --- Define the State for the Quiz Agent Graph ---
class QuizGraphState(TypedDict):
    """
    Represents the state of the Quiz Agent graph.
    """
    # Inputs from Orchestrator
    user_input: str
    conversation_history: List[Dict[str, str]] # Although not used directly by QuizAgent.run, included for consistency
    analysis_in_progress: bool # Although not used directly by QuizAgent.run, included for consistency
    context: Optional[Dict[str, Any]]

    # Quiz Agent specific components/data (will be initialized in QuizAgentComponents)
    system_prompt: str
    settings: Any # Assuming backend.config.Settings object
    llm: Any # Assuming OpenAI client object

    # Intermediate states of the Quiz Agent workflow
    retrieved_documents: Optional[List[Dict[str, Any]]]
    used_rag: Optional[bool]
    rag_context_str: Optional[str]
    llm_messages: Optional[List[Dict[str, Any]]]
    llm_raw_response_content: Optional[str]
    llm_parsed_response: Optional[Dict[str, Any]]
    final_response_content: Optional[str] # The user-facing message
    llm_output_details: Optional[Dict[str, Any]]
    model_used: Optional[str]
    quiz_generation_difficulty: Optional[str]
    quiz_question_count: Optional[int]
    final_metadata: Optional[Dict[str, Any]] # The metadata to be returned

    # Output for Orchestrator
    agent_output_for_orchestrator: Optional[Dict[str, Any]]
