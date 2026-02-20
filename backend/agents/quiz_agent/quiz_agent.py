# backend/agents/quiz_agent/quiz_agent.py

import os
import sys
import re
from typing import Dict, Any, List, Optional

# Ensure backend directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

# Local imports for the new graph structure
from backend.agents.base import BaseAgent
from .state import QuizGraphState
from .graph import quiz_agent_graph

# Compile the graph globally once
compiled_quiz_agent_app = quiz_agent_graph.compile()


def _extract_choice_index(user_input: str) -> Optional[int]:
    matched = re.fullmatch(r"\s*([1-4])\s*번?\s*", user_input or "")
    if not matched:
        return None
    return int(matched.group(1)) - 1


def _normalize_pending_quiz(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    question = str(raw.get("question", "")).strip()
    choices_raw = raw.get("choices", [])
    choices = [str(choice) for choice in choices_raw] if isinstance(choices_raw, list) else []
    answer_raw = raw.get("answer")

    answer_idx: Optional[int] = None
    if isinstance(answer_raw, int):
        answer_idx = answer_raw
    elif isinstance(answer_raw, str) and answer_raw.isdigit():
        answer_idx = int(answer_raw)

    if not question or answer_idx is None or answer_idx < 0:
        return None

    return {
        "question": question,
        "choices": choices,
        "answer": answer_idx,
        "explanation": str(raw.get("explanation", "")).strip(),
    }


def _build_quiz_feedback_message(pending_quiz: Dict[str, Any], selected_idx: int) -> str:
    correct_idx = int(pending_quiz.get("answer", -1))
    choices = pending_quiz.get("choices", [])
    explanation = str(pending_quiz.get("explanation", "")).strip()

    if selected_idx == correct_idx:
        header = f"정답입니다. ({selected_idx + 1}번)"
    else:
        correct_choice = ""
        if isinstance(choices, list) and 0 <= correct_idx < len(choices):
            correct_choice = f" - {choices[correct_idx]}"
        header = f"오답입니다. 정답은 {correct_idx + 1}번{correct_choice}"

    lines = [header]
    if explanation:
        lines.append(f"해설: {explanation}")
    lines.append("다음 문제를 원하면 '다음 문제'라고 입력하세요.")
    return "\n".join(lines)

class QuizAgent(BaseAgent):
    """
    Quiz Agent, now implemented as a thin wrapper around a LangGraph workflow.
    """
    agent_type: str = "quiz"
    
    def __init__(self):
        # The actual components (like LLM, RAG, etc.) are now managed within the graph's nodes.py
        # and initialized globally by QuizAgentComponents.
        pass

    async def run(
            self,
            user_input: str,
            conversation_history: List[Dict[str, str]], # Retained for signature consistency
            analysis_in_progress: bool, # Retained for signature consistency
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs the Quiz Agent's logic by invoking its internal LangGraph.

        Args:
            user_input (str): The user's query or description.
            conversation_history (List[Dict[str, str]]): Conversation history (retained for signature).
            analysis_in_progress (bool): Flag for multi-turn (retained for signature).
            context (Optional[Dict[str, Any]]): Additional context.

        Returns:
            Dict[str, Any]: Output for the orchestrator.
        """
        if context is None:
            context = {}
        context = dict(context)
        agent_state = context.pop("_agent_specific_state", {})
        pending_quiz = _normalize_pending_quiz(
            agent_state.get("pending_quiz") if isinstance(agent_state, dict) else None
        )

        selected_choice_idx = _extract_choice_index(user_input)
        if pending_quiz is not None and selected_choice_idx is not None:
            response_text = _build_quiz_feedback_message(pending_quiz, selected_choice_idx)
            updated_history = list(conversation_history)
            updated_history.append({"role": "User", "content": user_input})
            updated_history.append({"role": "Agent", "content": response_text})
            return {
                "response": {
                    "response": response_text,
                    "agent_type": self.agent_type,
                    "metadata": {
                        "mode": "quiz_answer",
                        "selected_choice": selected_choice_idx + 1,
                    },
                },
                "conversation_history": updated_history,
                "analysis_in_progress": False,
                "agent_specific_state": {
                    "awaiting_follow_up": False,
                    "pending_quiz": None,
                },
            }

        # Initialize the state for the quiz agent graph
        initial_state: QuizGraphState = {
            "user_input": user_input,
            "conversation_history": conversation_history, # Pass through
            "analysis_in_progress": analysis_in_progress, # Pass through
            "context": context,
            # Components are initialized globally in nodes.py
            "system_prompt": None, # Will be set by components
            "settings": None,      # Will be set by components
            "llm": None,           # Will be set by components
            # Other fields will be populated by nodes
            "retrieved_documents": None,
            "used_rag": None,
            "rag_context_str": None,
            "llm_messages": None,
            "llm_raw_response_content": None,
            "llm_parsed_response": None,
            "final_response_content": None,
            "llm_output_details": None,
            "model_used": None,
            "quiz_generation_difficulty": None,
            "quiz_question_count": None,
            "final_metadata": None,
            "agent_output_for_orchestrator": None,
        }

        # Invoke the compiled graph
        final_state = await compiled_quiz_agent_app.ainvoke(initial_state)

        # Extract the final output for the orchestrator
        final_output = final_state.get("agent_output_for_orchestrator")
        
        if final_output is None:
            # Fallback for unexpected graph termination
            error_message = "퀴즈 에이전트: 그래프 실행 후 최종 출력을 얻지 못했습니다."
            return {
                "response": {
                    "response": error_message,
                    "agent_type": self.agent_type,
                    "metadata": {"error": error_message}
                },
                "conversation_history": final_state.get("conversation_history", conversation_history),
                "analysis_in_progress": final_state.get("analysis_in_progress", False),
                "agent_specific_state": {"awaiting_follow_up": False},
            }
        
        response_text = str(final_output.get("response", ""))
        metadata = final_output.get("metadata", {}) if isinstance(final_output, dict) else {}
        llm_details = metadata.get("llm_output_details", {}) if isinstance(metadata, dict) else {}
        required_fields = []
        next_pending_quiz = None
        if isinstance(llm_details, dict):
            required_fields = llm_details.get("required_fields") or []
            nested_error = llm_details.get("error")
            if isinstance(nested_error, dict):
                required_fields = required_fields or nested_error.get("required_fields") or []
            questions = llm_details.get("questions")
            if isinstance(questions, list) and questions:
                next_pending_quiz = _normalize_pending_quiz(questions[0])

        awaiting_follow_up = bool(required_fields) or (
            "필요" in response_text and "정보" in response_text
        ) or (next_pending_quiz is not None)

        updated_history = final_state.get("conversation_history", conversation_history)
        if not isinstance(updated_history, list) or len(updated_history) <= len(conversation_history):
            updated_history = list(conversation_history)
            updated_history.append({"role": "User", "content": user_input})
            updated_history.append({"role": "Agent", "content": response_text})

        return {
            "response": final_output,
            "conversation_history": updated_history,
            "analysis_in_progress": final_state.get("analysis_in_progress", False),
            "agent_specific_state": {
                "awaiting_follow_up": awaiting_follow_up,
                "pending_quiz": next_pending_quiz,
            },
        }
