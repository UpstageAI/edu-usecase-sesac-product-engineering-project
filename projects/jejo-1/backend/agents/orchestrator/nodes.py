# backend/agents/orchestrator/nodes.py

import os
import sys
import json
import time
import re
import inspect
from typing import Dict, Any, List, Optional, Type, cast

import openai
from openai import AsyncOpenAI
from langsmith import traceable # traceable will be applied at graph level or individual agent level if needed

# External imports
from backend.config import get_settings
from backend.core.response_converter import normalize_response
from backend.utils.logger import get_logger

# Internal imports for Orchestrator package
from .state import OrchestratorGraphState
from .session_store import create_conversation_store

from backend.agents.default_chat.default_chat_agent import DefaultChatAgent # Actual DefaultChatAgent import
from backend.agents.riskmanaging.graph import RiskManagingAgent # Actual RiskManagingAgent import
from backend.agents.quiz_agent.quiz_agent import QuizAgent                   # Actual QuizAgent import
from backend.agents.email_agent.email_agent import EmailAgent               # Actual EmailAgent import


# --- Prompt Loader Function (from old orchestrator.py) ---
def _load_prompt(prompt_file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # from backend/agents/orchestrator to project root
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    prompt_path = os.path.join(project_root, 'backend', 'prompts', prompt_file_name) # Adjusted path
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()




AGENT_CLASS_MAPPING: Dict[str, Type[Any]] = {
    "riskmanaging": RiskManagingAgent,
    "quiz": QuizAgent, 
    "email": EmailAgent,
    "default_chat": DefaultChatAgent,
}

DEFAULT_AGENT_NAME = "default_chat"
logger = get_logger(__name__)

# --- Global Components ---
# These components are initialized once and used across nodes.
# In a real application, these might be managed by a dependency injection framework or passed explicitly.
class OrchestratorComponents:
    def __init__(self):
        self.settings = get_settings()
        self.conversation_store = create_conversation_store()  # Factory function chooses InMemory or Redis
        
        self.llm = None
        if self.settings.upstage_api_key:
            self.llm = AsyncOpenAI(
                base_url="https://api.upstage.ai/v1",
                api_key=self.settings.upstage_api_key
            )
        else:
            logger.warning(
                "UPSTAGE_API_KEY is not set; orchestrator LLM intent classification will fallback."
            )
        
        self.orchestrator_intent_prompt = _load_prompt("orchestrator_intent_prompt.txt")

        # Configure Langsmith tracing
        if self.settings.langsmith_tracing and self.settings.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.settings.langsmith_project
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
        self.agents_instances: Dict[str, Any] = {}
        for agent_name, agent_class in AGENT_CLASS_MAPPING.items():
            self.agents_instances[agent_name] = agent_class()
            logger.info("Orchestrator initialized agent: %s", agent_name)

ORCHESTRATOR_COMPONENTS = OrchestratorComponents()


# --- Orchestrator Node Functions ---

def load_session_state_node(state: OrchestratorGraphState) -> Dict[str, Any]:
    # Need to convert TypedDict to dict for manipulation
    state_dict = cast(Dict[str, Any], state)
    
    session_id = state_dict["session_id"]
    conversation_store = ORCHESTRATOR_COMPONENTS.conversation_store
    
    session_data = conversation_store.get_state(session_id)
    if not session_data:
        session_data = {
            "active_agent": None,
            "conversation_history": [],
            "agent_specific_state": {},
            "last_interaction_timestamp": time.time(),
        }
        logger.debug("Initialized new session: %s", session_id)
    else:
        logger.debug(
            "Loaded existing session: %s, active_agent=%s",
            session_id,
            session_data.get("active_agent"),
        )
    
    # Update state with loaded session data
    state_dict["conversation_history"] = session_data["conversation_history"]
    state_dict["active_agent"] = session_data["active_agent"]
    state_dict["agent_specific_state"] = session_data["agent_specific_state"]
    # state_dict["last_interaction_timestamp"] = session_data["last_interaction_timestamp"] # Handled by update node

    return state_dict


async def _classify_intent_with_llm(user_input: str) -> str:
    llm = ORCHESTRATOR_COMPONENTS.llm
    orchestrator_intent_prompt = ORCHESTRATOR_COMPONENTS.orchestrator_intent_prompt

    if not llm:
        logger.debug("LLM client not initialized. Falling back to default.")
        return DEFAULT_AGENT_NAME
    
    try:
        system_message_content = orchestrator_intent_prompt.split('---')[0].strip()
        user_message_content = f"사용자 요청: {user_input}"

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        
        logger.debug(
            "LLM intent classification request prepared (messages=%s, user_input_len=%s)",
            len(messages),
            len(user_input),
        )

        response_completion = await llm.chat.completions.create(
            model="solar-pro2",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        response = response_completion.choices[0].message.content
        
        logger.debug(
            "LLM intent classification raw response (truncated): %s",
            str(response)[:400],
        )
        from backend.utils.json_utils import safe_json_parse
        parsed_response = safe_json_parse(response)
        agent_type = parsed_response.get("agent_type", DEFAULT_AGENT_NAME)
        reason = parsed_response.get("reason", "LLM based classification.")
        logger.info("LLM classified intent: %s (reason=%s)", agent_type, reason)
        return agent_type

    except Exception as e:
        logger.warning("Intent classification failed (%s). Falling back to default.", e)
        return DEFAULT_AGENT_NAME


async def detect_intent_and_route_node(state: OrchestratorGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    user_input = state_dict["user_input"]
    context = state_dict.get("context") or {}
    agents_instances = ORCHESTRATOR_COMPONENTS.agents_instances

    # 1. Explicit mode from frontend context always overrides session stickiness.
    if context.get("mode") and context["mode"] in agents_instances:
        logger.debug("Routing overridden by context.mode=%s", context["mode"])
        state_dict["selected_agent_name"] = context["mode"]
        return state_dict

    # 2. 강력한 키워드 기반 우선 라우팅
    risk_keywords = ["클레임", "선적 지연", "리스크", "손해배상", "대처 방안", "유사 사례", "패널티", "지연"]
    if any(keyword in user_input for keyword in risk_keywords):
        logger.debug("Keyword routing to riskmanaging")
        state_dict["selected_agent_name"] = "riskmanaging"
        return state_dict

    quiz_keywords = ["퀴즈", "학습"]
    if any(keyword in user_input for keyword in quiz_keywords):
        logger.debug("Keyword routing to quiz")
        state_dict["selected_agent_name"] = "quiz"
        return state_dict

    email_keywords = ["이메일", "메일", "회신", "답장", "초안", "mail", "리뷰", "검토", "첨삭", "교정"]
    if any(keyword in user_input for keyword in email_keywords):
        logger.debug("Keyword routing to email")
        state_dict["selected_agent_name"] = "email"
        return state_dict

    current_session_state = { # Reconstruct current_session_state for _detect_intent_and_route logic
        "active_agent": state_dict["active_agent"],
        "agent_specific_state": state_dict["agent_specific_state"],
        "conversation_history": state_dict["conversation_history"]
    }

    # 3. Prioritize active agent in session state.
    # Only riskmanaging stays sticky while multi-turn analysis is in progress.
    active_agent_name = current_session_state.get("active_agent")
    if active_agent_name and active_agent_name in agents_instances:
        agent_specific_state = current_session_state.get("agent_specific_state", {})
        if active_agent_name == "riskmanaging":
            if agent_specific_state.get("analysis_in_progress"):
                logger.debug(
                    "Continuing with active agent %s (analysis in progress)",
                    active_agent_name,
                )
                state_dict["selected_agent_name"] = active_agent_name
                return state_dict # Return updated state to flow to call_agent_node
        elif active_agent_name in {"quiz", "email"}:
            normalized_user_input = user_input.strip().lower()
            clarification_keywords = [
                "어떤정보",
                "어떤 정보",
                "무슨정보",
                "무슨 정보",
                "뭐가 필요",
                "뭐가 필요한",
                "필요한 정보",
                "추가 정보",
            ]
            email_followup_keywords = [
                "한국어",
                "영어",
                "번역",
                "다시",
                "수정",
                "고쳐",
                "톤",
                "제목",
                "간단",
                "짧게",
                "길게",
                "공손",
            ]
            quiz_followup_keywords = [
                "정답",
                "해설",
                "힌트",
                "다음 문제",
                "다음문제",
                "한문제",
                "난이도",
                "쉽게",
                "어렵게",
                "더 문제",
                "문제 더",
                "퀴즈 더",
                "더 내줘",
                "더줘",
                "더 달라고",
                "더 만들어",
                "추가 문제",
                "추가 퀴즈",
            ]

            if (
                agent_specific_state.get("awaiting_follow_up")
                and (
                    any(keyword in user_input for keyword in clarification_keywords)
                    or ("\n" in user_input and len(user_input.strip()) > 40)
                )
            ):
                logger.debug(
                    "Continuing with active agent %s (awaiting follow-up details)",
                    active_agent_name,
                )
                state_dict["selected_agent_name"] = active_agent_name
                return state_dict

            # Keep short follow-up edits in the same active quiz/email flow.
            # Example: "한국어로 만들어줄래?" right after email draft generation.
            if (
                active_agent_name == "email"
                and len(normalized_user_input) <= 80
                and any(keyword in user_input for keyword in email_followup_keywords)
            ):
                logger.debug("Continuing with active agent email (short follow-up edit request)")
                state_dict["selected_agent_name"] = active_agent_name
                return state_dict
            if (
                active_agent_name == "quiz"
                and len(normalized_user_input) <= 80
                and (
                    any(keyword in user_input for keyword in quiz_followup_keywords)
                    or (
                        state_dict.get("agent_specific_state", {}).get("pending_quiz")
                        and re.fullmatch(r"\s*([1-4])\s*번?\s*", user_input)
                    )
                )
            ):
                logger.debug("Continuing with active agent quiz (short follow-up quiz request)")
                state_dict["selected_agent_name"] = active_agent_name
                return state_dict
        else:
            logger.debug("Active agent is %s; re-evaluating intent", active_agent_name)

    # 4. LLM-based intent classification
    llm_predicted_agent_type = await _classify_intent_with_llm(user_input)
    state_dict["llm_intent_classification"] = {"predicted_type": llm_predicted_agent_type}
    
    if llm_predicted_agent_type == "out_of_scope":
        logger.debug("LLM classified out_of_scope; routing to %s", DEFAULT_AGENT_NAME)
        state_dict["selected_agent_name"] = DEFAULT_AGENT_NAME
        return state_dict
    elif llm_predicted_agent_type in AGENT_CLASS_MAPPING:
        logger.debug("LLM routing to %s", llm_predicted_agent_type)
        state_dict["selected_agent_name"] = llm_predicted_agent_type
        return state_dict
    else:
        logger.warning(
            "LLM predicted unknown agent type '%s'. Falling back to default.",
            llm_predicted_agent_type,
        )
        state_dict["selected_agent_name"] = DEFAULT_AGENT_NAME
        return state_dict


async def call_agent_node(state: OrchestratorGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    selected_agent_name = state_dict["selected_agent_name"]
    user_input = state_dict["user_input"]
    conversation_history = state_dict["conversation_history"]
    agent_specific_state = state_dict["agent_specific_state"]
    context = state_dict.get("context") or {}
    agents_instances = ORCHESTRATOR_COMPONENTS.agents_instances

    agent_instance = agents_instances.get(selected_agent_name)

    if not agent_instance:
        response_content = f"죄송합니다. 현재 '{selected_agent_name}' 에이전트는 사용할 수 없습니다."
        response_payload = {
            "response": response_content,
            "agent_type": "orchestrator",
            "metadata": {"reason": "Agent not found or initialized"}
        }
        state_dict["orchestrator_response"] = response_payload
        state_dict["conversation_history"].append({"role": "Agent", "content": response_content})
        state_dict["active_agent"] = None
        state_dict["agent_specific_state"] = {}
        return state_dict
    
    agent_context = dict(context)
    # Ensure session_id is passed to the agent for state persistence (MemorySaver/SqliteSaver)
    agent_context["session_id"] = state_dict.get("session_id")
    
    if selected_agent_name == "quiz":
        agent_context["_agent_specific_state"] = agent_specific_state

    agent_output = await agent_instance.run(
        user_input=user_input,
        conversation_history=conversation_history,
        analysis_in_progress=agent_specific_state.get("analysis_in_progress", False),
        context=agent_context
    )

    state_dict["orchestrator_response"] = agent_output.get("response", {"response": "에이전트 응답 오류", "agent_type": "orchestrator", "metadata": {}})
    state_dict["conversation_history"] = agent_output.get("conversation_history", conversation_history)
    
    if "analysis_in_progress" in agent_output:
        state_dict["agent_specific_state"]["analysis_in_progress"] = agent_output["analysis_in_progress"]

    agent_specific_update = agent_output.get("agent_specific_state")
    if isinstance(agent_specific_update, dict):
        state_dict["agent_specific_state"].update(agent_specific_update)
    
    if selected_agent_name == "riskmanaging" and not state_dict["agent_specific_state"].get("analysis_in_progress"):
         state_dict["active_agent"] = None
         logger.debug(
             "RiskManagingAgent completed analysis. Resetting active_agent for session %s.",
             state_dict["session_id"],
         )
    else:
        state_dict["active_agent"] = selected_agent_name

    return state_dict


def finalize_and_save_state_node(state: OrchestratorGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    session_id = state_dict["session_id"]
    conversation_store = ORCHESTRATOR_COMPONENTS.conversation_store
    
    # Save session state after agent interaction
    session_state_to_save = {
        "active_agent": state_dict["active_agent"],
        "conversation_history": state_dict["conversation_history"],
        "agent_specific_state": state_dict["agent_specific_state"],
        "last_interaction_timestamp": time.time(), # Update timestamp
    }
    conversation_store.save_state(session_id, session_state_to_save)
    
    return state_dict # Return the updated state, graph will then pass to normalizer


def normalize_response_node(state: OrchestratorGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)
    
    response_payload = state_dict["orchestrator_response"]
    final_response = normalize_response(response_payload)

    # Update graph state with normalized payload so compiled graph output always
    # contains ChatResponse-compatible fields.
    state_dict["type"] = final_response["type"]
    state_dict["message"] = final_response["message"]
    state_dict["report"] = final_response["report"]
    state_dict["meta"] = final_response.get("meta", {})
    return state_dict
