# backend/agents/quiz_agent/nodes.py

import os
import sys
import json
import re
import hashlib
from typing import Dict, Any, List, Optional, cast
import openai
from openai import AsyncOpenAI

# Ensure backend directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # From backend/agents/quiz_agent to project root
sys.path.append(project_root)

# Local imports
from backend.config import get_settings
from backend.utils.logger import get_logger
# RAG functionality now provided by tools.py
from backend.agents.quiz_agent.state import QuizGraphState

# --- Constants ---
QUIZ_AGENT_TYPE = "quiz"
QUIZ_FALLBACK_REFERENCE = """
- FOB: 본선인도 조건으로, 매도인은 지정 선적항에서 물품을 본선에 적재할 때까지 책임을 집니다.
- CIF: 운임·보험료 포함 조건으로, 매도인이 보험과 운임을 부담하지만 위험은 선적 시점에 이전됩니다.
- L/C (Letter of Credit): 은행이 대금 지급을 보증하는 신용장 방식입니다.
- B/L (Bill of Lading): 선하증권으로, 운송계약 및 화물 인수 증빙 문서입니다.
- HS Code: 국제 통일상품분류체계 코드로 관세율과 통관 요건 판단에 사용됩니다.
""".strip()
DEFAULT_QUIZ_QUESTION_COUNT = 5
MAX_QUIZ_QUESTION_COUNT = 10
logger = get_logger(__name__)


def _parse_json_flexible(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    candidates: List[str] = [stripped]

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(stripped[array_start:array_end + 1].strip())

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(stripped[object_start:object_end + 1].strip())

    seen = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _clamp_question_count(value: int) -> int:
    if value < 1:
        return 1
    if value > MAX_QUIZ_QUESTION_COUNT:
        return MAX_QUIZ_QUESTION_COUNT
    return value


def _extract_requested_question_count(user_input: str, context: Dict[str, Any]) -> int:
    context_count = context.get("question_count")
    if isinstance(context_count, int):
        return _clamp_question_count(context_count)

    text = str(user_input or "")
    lowered = text.lower()

    digit_match = re.search(r"(\d{1,2})\s*(?:문제|문항|개|quiz)", lowered)
    if digit_match:
        return _clamp_question_count(int(digit_match.group(1)))

    korean_count_map = {
        "한": 1,
        "하나": 1,
        "두": 2,
        "둘": 2,
        "세": 3,
        "셋": 3,
        "네": 4,
        "넷": 4,
        "다섯": 5,
    }
    korean_match = re.search(r"(한|하나|두|둘|세|셋|네|넷|다섯)\s*(?:문제|문항|개)", text)
    if korean_match:
        return _clamp_question_count(korean_count_map[korean_match.group(1)])

    if any(token in text for token in ["더", "추가", "더줘", "더 만들어", "더 내"]):
        return DEFAULT_QUIZ_QUESTION_COUNT

    return DEFAULT_QUIZ_QUESTION_COUNT


def _build_difficulty_instruction(difficulty: str, question_count: int) -> str:
    normalized = (difficulty or "medium").lower()
    if normalized == "easy":
        return f"쉬운 난이도로 {question_count}문제를 출제하세요."
    if normalized == "hard":
        return f"어려운 난이도로 {question_count}문제를 출제하세요."
    return f"중간 난이도로 {question_count}문제를 출제하세요."


def _build_exclude_instruction(exclude_terms: Any) -> str:
    if isinstance(exclude_terms, list) and exclude_terms:
        joined = ", ".join(str(term) for term in exclude_terms)
        return f"- 다음 용어는 문제에서 제외하세요: {joined}"
    return "- 별도 제외 용어는 없습니다."


def _build_feedback_instruction(feedback: Any) -> str:
    if isinstance(feedback, str) and feedback.strip():
        return f"- 이전 피드백을 반영하세요: {feedback.strip()}"
    return "- 이전 피드백은 없습니다."


def _build_reference_data(retrieved_documents: List[Dict[str, Any]]) -> str:
    if not retrieved_documents:
        return QUIZ_FALLBACK_REFERENCE

    lines: List[str] = []
    # Use the first half of documents (up to 3) for reference
    for idx, doc in enumerate(retrieved_documents[:3], start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_dataset", "unknown")
        doc_type = metadata.get("document_type", "unknown")
        text = str(doc.get("document", "")).strip().replace("\n", " ")
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        lines.append(f"- [{idx}] ({source}/{doc_type}) {snippet}")

    return "\n".join(lines) if lines else QUIZ_FALLBACK_REFERENCE


def _build_distractor_data(retrieved_documents: List[Dict[str, Any]]) -> str:
    if not retrieved_documents or len(retrieved_documents) <= 3:
        return "(오답 생성을 위한 추가 유사 정보 없음. 일반적인 실무 오류 유형을 사용하세요.)"

    lines: List[str] = []
    # Use the second half of documents (from 4th onwards) for distractors
    for idx, doc in enumerate(retrieved_documents[3:6], start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_dataset", "unknown")
        doc_type = metadata.get("document_type", "unknown")
        text = str(doc.get("document", "")).strip().replace("\n", " ")
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        lines.append(f"- [유사-오류-{idx}] ({source}/{doc_type}) {snippet}")

    return "\n".join(lines) if lines else "(오답 생성을 위한 추가 유사 정보 없음)"


def _extract_questions_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict) and item.get("question")]

    if isinstance(payload, dict):
        if isinstance(payload.get("questions"), list):
            return [
                item
                for item in payload.get("questions", [])
                if isinstance(item, dict) and item.get("question")
            ]
        if isinstance(payload.get("answer"), list):
            return [
                item
                for item in payload.get("answer", [])
                if isinstance(item, dict) and item.get("question")
            ]
    return []


def _build_quiz_preview_message(questions: List[Dict[str, Any]]) -> str:
    if not questions:
        return "퀴즈를 생성하지 못했습니다. 다시 시도해 주세요."

    first = questions[0]
    choices = first.get("choices", [])
    option_lines = []
    if isinstance(choices, list):
        for idx, choice in enumerate(choices[:4], start=1):
            option_lines.append(f"{idx}. {choice}")

    lines = [f"퀴즈 {len(questions)}문제를 생성했습니다. 오른쪽 퀴즈 패널에서 풀어보세요."]
    lines.append(f"[퀴즈 1]\n{first.get('question')}")
    if option_lines:
        lines.append("\n".join(option_lines))
    return "\n".join(lines)


def _stable_index_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _rebalance_answer_positions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LLM이 answer=0에 쏠리는 경향을 완화하기 위해
    정답 위치를 문제별로 분산 재배치한다.
    """
    balanced: List[Dict[str, Any]] = []
    for idx, original in enumerate(questions):
        if not isinstance(original, dict):
            continue

        question = dict(original)
        choices_raw = question.get("choices")
        if not isinstance(choices_raw, list):
            balanced.append(question)
            continue

        choices = [str(choice) for choice in choices_raw]
        if len(choices) < 2:
            question["choices"] = choices
            balanced.append(question)
            continue

        answer_raw = question.get("answer")
        if isinstance(answer_raw, int):
            answer_idx = answer_raw
        elif isinstance(answer_raw, str) and answer_raw.isdigit():
            answer_idx = int(answer_raw)
        else:
            question["choices"] = choices
            balanced.append(question)
            continue

        if answer_idx < 0 or answer_idx >= len(choices):
            question["choices"] = choices
            balanced.append(question)
            continue

        seed = _stable_index_seed(str(question.get("question", "")))
        target_idx = (seed + idx) % len(choices)
        if target_idx != answer_idx:
            choices[target_idx], choices[answer_idx] = choices[answer_idx], choices[target_idx]
            answer_idx = target_idx

        question["choices"] = choices
        question["answer"] = answer_idx
        balanced.append(question)

    return balanced

# --- Prompt Loader Function ---
def _load_prompt(prompt_file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # Correct path
    prompt_path = os.path.join(project_root, 'backend', 'prompts', prompt_file_name)
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Global Components for QuizAgent ---
class QuizAgentComponents:
    def __init__(self):
        self.settings = get_settings()
        self.system_prompt = _load_prompt("quiz_prompt.txt")

        self.llm = None
        if self.settings.upstage_api_key:
            self.llm = AsyncOpenAI(
                base_url="https://api.upstage.ai/v1",
                api_key=self.settings.upstage_api_key
            )
        else:
            logger.warning("UPSTAGE_API_KEY is not set for QuizAgent. LLM calls will fail.")
        
        # Configure Langsmith tracing - if not configured by Orchestrator
        if self.settings.langsmith_tracing and self.settings.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.settings.langsmith_project
        # else: os.environ["LANGCHAIN_TRACING_V2"] is already set to "false" by orchestrator if not enabled

QUIZ_AGENT_COMPONENTS = QuizAgentComponents()


# --- Node Functions ---

def perform_rag_search_node(state: QuizGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    user_input = state_dict["user_input"]
    context = state_dict.get("context") or {}

    # Build search query
    rag_query = user_input
    if context.get("topic"):
        rag_query += f" {context['topic']}"
    if context.get("difficulty"):
        rag_query += f" {context['difficulty']}"

    # Use tool: search_trade_documents
    from backend.agents.quiz_agent.tools import search_trade_documents

    try:
        retrieved_documents = search_trade_documents.invoke(
            {
                "query": rag_query,
                "k": 6,
                "document_type": context.get("document_type"),
                "category": context.get("category"),
            }
        )
        used_rag = len(retrieved_documents) > 0
    except Exception as e:
        logger.warning("Error during quiz RAG search: %s", e)
        retrieved_documents = []
        used_rag = False

    state_dict["retrieved_documents"] = retrieved_documents
    state_dict["used_rag"] = used_rag

    # Use tool: format_quiz_context
    from backend.agents.quiz_agent.tools import format_quiz_context

    rag_context_str = format_quiz_context.invoke(
        {
            "retrieved_documents": retrieved_documents,
            "include_metadata": True,
        }
    )
    state_dict["rag_context_str"] = rag_context_str

    return state_dict


def prepare_llm_messages_node(state: QuizGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    system_prompt_template = QUIZ_AGENT_COMPONENTS.system_prompt
    user_input = state_dict["user_input"]
    context = state_dict.get("context") or {}
    rag_context_str = state_dict.get("rag_context_str", "")
    retrieved_documents = state_dict.get("retrieved_documents") or []
    question_count = _extract_requested_question_count(user_input, context)

    difficulty = str(context.get("difficulty", "medium"))
    system_prompt = (
        system_prompt_template
        .replace("{difficulty_instruction}", _build_difficulty_instruction(difficulty, question_count))
        .replace("{exclude_instruction}", _build_exclude_instruction(context.get("exclude_terms")))
        .replace("{feedback_instruction}", _build_feedback_instruction(context.get("feedback")))
        .replace("{reference_data}", _build_reference_data(retrieved_documents))
        .replace("{distractor_data}", _build_distractor_data(retrieved_documents))
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    llm_user_message_content = f"사용자 요청: {user_input}"
    if context:
        llm_user_message_content += f"\n추가 컨텍스트: {json.dumps(context, ensure_ascii=False)}"
    if rag_context_str:
        llm_user_message_content += rag_context_str
    llm_user_message_content += (
        f"\n\n반드시 무역 실무 용어 기반 4지선다 {question_count}문제를 작성하세요. "
        "출력은 JSON만 반환하고, 필요 정보 부족 같은 메시지는 출력하지 마세요."
    )
    
    messages.append({"role": "user", "content": llm_user_message_content})

    state_dict["llm_messages"] = messages
    state_dict["quiz_generation_difficulty"] = difficulty
    state_dict["quiz_question_count"] = question_count
    return state_dict


async def call_llm_and_parse_response_node(state: QuizGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    llm = QUIZ_AGENT_COMPONENTS.llm
    llm_messages = state_dict["llm_messages"]
    requested_count = int(state_dict.get("quiz_question_count") or DEFAULT_QUIZ_QUESTION_COUNT)

    llm_response_content = "LLM 호출에 실패했거나 응답이 없습니다."
    model_used = "solar-pro2"
    llm_output_details = {}
    
    if llm:
        try:
            chat_completion = await llm.chat.completions.create(
                model=model_used,
                messages=llm_messages,
                temperature=0.3
            )
            llm_response_content = chat_completion.choices[0].message.content
            state_dict["llm_raw_response_content"] = llm_response_content
            
            parsed_llm_response = _parse_json_flexible(llm_response_content)
            try:
                if parsed_llm_response is None:
                    raise json.JSONDecodeError("Unable to parse JSON payload", llm_response_content, 0)
                llm_output_details = parsed_llm_response
                questions = _extract_questions_from_payload(parsed_llm_response)
                if questions:
                    questions = _rebalance_answer_positions(questions)
                    final_response_content = _build_quiz_preview_message(questions)
                    llm_output_details = {
                        "questions": questions,
                        "generated_question_count": len(questions),
                        "requested_question_count": requested_count,
                    }
                elif isinstance(parsed_llm_response, dict):
                    error_field = parsed_llm_response.get("error")
                    if isinstance(error_field, dict):
                        final_response_content = error_field.get(
                            "message",
                            json.dumps(error_field, ensure_ascii=False),
                        )
                    elif isinstance(error_field, str):
                        final_response_content = error_field
                    elif isinstance(parsed_llm_response.get("quiz_response"), str):
                        final_response_content = parsed_llm_response["quiz_response"]
                    else:
                        final_response_content = json.dumps(parsed_llm_response, ensure_ascii=False)
                else:
                    final_response_content = llm_response_content
            except json.JSONDecodeError:
                logger.debug(
                    "QuizAgent LLM response was not valid JSON (truncated): %s",
                    llm_response_content[:200],
                )
                final_response_content = llm_response_content
                llm_output_details = {"raw_llm_response": llm_response_content}
            
        except openai.APIError as e:
            logger.warning("QuizAgent Upstage API error: %s", e)
            final_response_content = f"LLM API 호출 중 오류가 발생했습니다: {e}"
            llm_output_details = {"error": str(e)}
        except Exception as e:
            logger.warning("QuizAgent unexpected LLM error: %s", e)
            final_response_content = f"LLM 호출 중 예상치 못한 오류가 발생했습니다: {e}"
            llm_output_details = {"error": str(e)}
    else:
        final_response_content = "LLM 클라이언트가 초기화되지 않아 응답을 생성할 수 없습니다. UPSTAGE_API_KEY를 확인하세요."
        llm_output_details = {"error": "LLM client not initialized due to missing API key."}

    state_dict["final_response_content"] = final_response_content
    state_dict["llm_output_details"] = llm_output_details
    state_dict["model_used"] = model_used # Store model used for metadata
    return state_dict


def format_quiz_output_node(state: QuizGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    final_response = state_dict["final_response_content"]
    used_rag = state_dict.get("used_rag", False)
    retrieved_documents = state_dict.get("retrieved_documents", [])
    model_used = state_dict.get("model_used", "solar-pro2")
    llm_messages = state_dict.get("llm_messages", [])
    llm_output_details = state_dict.get("llm_output_details", {})
    quiz_question_count = state_dict.get("quiz_question_count", DEFAULT_QUIZ_QUESTION_COUNT)
    context = state_dict["context"]
    user_input = state_dict["user_input"]


    metadata = {
        "used_rag": used_rag,
        "documents": retrieved_documents,
        "model": model_used,
        "llm_input_prepared": llm_messages,
        "llm_output_details": llm_output_details,
        "quiz_question_count": quiz_question_count,
        "input_context": context,
        "processed_input": user_input
    }

    state_dict["agent_output_for_orchestrator"] = {
        "response": final_response,
        "agent_type": QUIZ_AGENT_TYPE,
        "metadata": metadata
    }
    return state_dict
