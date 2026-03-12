# backend/agents/email_agent/nodes.py

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional, cast
import openai
from openai import AsyncOpenAI

# Ensure backend directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # From backend/agents/email_agent to project root
sys.path.append(project_root)

# Local imports
from backend.config import get_settings
from backend.utils.logger import get_logger
# RAG and validation functionality now provided by tools.py
from backend.agents.email_agent.state import EmailGraphState

# --- Constants ---
EMAIL_AGENT_TYPE = "email"

EMAIL_REVIEW_KEYWORDS = ["리뷰", "검토", "피드백", "첨삭", "교정", "review"]
EMAIL_DRAFT_KEYWORDS = ["초안", "작성", "만들어", "draft", "write", "작성해"]
EMAIL_EDIT_FOLLOWUP_KEYWORDS = [
    "한국어",
    "영어",
    "번역",
    "다시",
    "수정",
    "고쳐",
    "톤",
    "짧게",
    "길게",
    "간단",
    "공손",
]
EMAIL_TRAILING_REVIEW_PATTERNS = [
    "리뷰해줘",
    "검토해줘",
    "검토 부탁",
    "리뷰 부탁",
    "review please",
    "please review",
]
logger = get_logger(__name__)


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


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


def _detect_email_task_type(user_input: str, context: Dict[str, Any]) -> str:
    explicit_task = str(context.get("email_task", "")).strip().lower()
    if explicit_task in {"draft", "review"}:
        return explicit_task

    normalized = user_input.lower()
    has_review_keyword = _contains_any(normalized, EMAIL_REVIEW_KEYWORDS)
    has_draft_keyword = _contains_any(normalized, EMAIL_DRAFT_KEYWORDS)

    # If both are present (e.g., "검토를 위한 이메일 초안 만들어줘"),
    # treat it as draft by default unless it is an explicit review request.
    if has_draft_keyword and has_review_keyword:
        explicit_review_patterns = [
            "리뷰해줘",
            "검토해줘",
            "리뷰 부탁",
            "검토 부탁",
            "이메일 리뷰",
            "이거 리뷰",
            "첨삭해",
            "교정해",
            "review please",
            "please review",
        ]
        if any(pattern in normalized for pattern in explicit_review_patterns):
            return "review"
        return "draft"

    if has_draft_keyword:
        return "draft"
    if has_review_keyword:
        return "review"
    return "draft"


def _detect_output_language(
    user_input: str,
    context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> str:
    explicit_language = str(context.get("language", "")).strip().lower()
    if explicit_language in {"ko", "korean", "kr", "한국어"}:
        return "ko"
    if explicit_language in {"en", "english", "영어"}:
        return "en"

    normalized = user_input.lower()
    if "한국어" in user_input or "한글" in user_input:
        return "ko"
    if "영어" in user_input or "english" in normalized:
        return "en"

    if re.search(r"[가-힣]", user_input):
        return "ko"

    # If recent conversation is mostly Korean, keep Korean output.
    recent_text = "\n".join(str(turn.get("content", "")) for turn in conversation_history[-4:])
    if re.search(r"[가-힣]", recent_text):
        return "ko"
    return "en"


def _is_follow_up_edit_request(text: str) -> bool:
    normalized = text.strip().lower()
    if len(normalized) > 100:
        return False
    return any(keyword in text for keyword in EMAIL_EDIT_FOLLOWUP_KEYWORDS)


def _build_language_instruction(output_language: str, task_type: str) -> str:
    if output_language == "ko":
        if task_type == "review":
            return "출력 언어: 반드시 한국어로 리뷰를 작성하세요."
        return "출력 언어: 반드시 한국어 이메일 본문만 작성하세요."
    if task_type == "review":
        return "Output language: English only."
    return "Output language: English email body only."


def _extract_email_body_from_text(text: str) -> Optional[str]:
    stripped = text.strip()
    if len(stripped) < 60:
        return None

    lower = stripped.lower()
    looks_like_email = any(
        marker in lower
        for marker in [
            "dear ",
            "best regards",
            "kind regards",
            "sincerely",
            "subject:",
            "제목:",
            "수신:",
            "incoterms:",
            "결제 조건:",
            "납기:",
            "수량:",
            "단가:",
        ]
    )
    if not looks_like_email and "\n" not in stripped:
        return None

    if "\n" in stripped:
        raw_lines = stripped.splitlines()
    else:
        # Many users paste templates as a single line separated by multiple spaces.
        raw_lines = re.split(r"\s{2,}", stripped)

    lines = [line.strip() for line in raw_lines if line.strip()]
    while lines:
        last_line_lower = lines[-1].lower()
        if any(pattern in last_line_lower for pattern in EMAIL_TRAILING_REVIEW_PATTERNS):
            lines.pop()
            continue
        break

    if not lines:
        return None

    candidate = "\n".join(lines).strip()
    return candidate if len(candidate) >= 60 else None


def _is_assistant_email_draft_candidate(text: str) -> bool:
    lowered = text.lower()
    strong_markers = [
        "subject:",
        "dear ",
        "best regards",
        "kind regards",
        "sincerely",
        "제목:",
        "수신:",
        "발신:",
    ]
    if any(marker in lowered for marker in strong_markers):
        return True

    has_greeting = ("안녕하세요" in text) or ("dear " in lowered)
    has_signoff = any(
        marker in lowered
        for marker in ["best regards", "kind regards", "sincerely", "감사합니다", "드림", "올림"]
    )
    has_trade_fields = any(
        marker in lowered
        for marker in [
            "incoterms",
            "결제 조건",
            "납기",
            "수량",
            "단가",
            "payment",
            "delivery",
            "quantity",
        ]
    )
    return has_greeting and (has_signoff or has_trade_fields)


def _extract_email_content(
    user_input: str,
    conversation_history: List[Dict[str, str]],
    task_type: str = "draft",
) -> Optional[str]:
    current_turn_email = _extract_email_body_from_text(user_input)
    if current_turn_email:
        return current_turn_email

    if task_type == "review":
        # 1) Prefer user-provided email text for review.
        for turn in reversed(conversation_history or []):
            role = str(turn.get("role", "")).strip().lower()
            if role not in {"user", "human"}:
                continue
            content = str(turn.get("content", "")).strip()
            candidate = _extract_email_body_from_text(content)
            if candidate:
                return candidate

        # 2) If user text is unavailable, allow assistant drafts only when they
        # clearly look like actual email content (avoid generic assistant messages).
        for turn in reversed(conversation_history or []):
            role = str(turn.get("role", "")).strip().lower()
            if role not in {"assistant", "agent", "ai"}:
                continue
            content = str(turn.get("content", "")).strip()
            if not _is_assistant_email_draft_candidate(content):
                continue
            candidate = _extract_email_body_from_text(content)
            if candidate:
                return candidate
        return None

    for turn in reversed(conversation_history or []):
        content = str(turn.get("content", "")).strip()
        candidate = _extract_email_body_from_text(content)
        if candidate:
            return candidate
    return None


def _extract_country(user_input: str, context: Dict[str, Any]) -> str:
    for key in ("recipient_country", "country", "target_country"):
        if context.get(key):
            return str(context[key]).strip()

    country_aliases = [
        ("미국", ["미국", "usa", "us", "america"]),
        ("일본", ["일본", "japan"]),
        ("중국", ["중국", "china"]),
        ("한국", ["한국", "korea"]),
        ("사우디아라비아", ["사우디", "saudi"]),
        ("독일", ["독일", "germany"]),
        ("베트남", ["베트남", "vietnam"]),
    ]
    lowered = user_input.lower()
    for country, aliases in country_aliases:
        if any(alias in lowered for alias in aliases):
            return country
    return "미지정"


def _format_retrieved_docs(
    retrieved_documents: List[Dict[str, Any]],
) -> Dict[str, str]:
    mistakes: List[str] = []
    emails: List[str] = []

    for idx, doc in enumerate(retrieved_documents or [], start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_dataset") or doc.get("source") or "unknown"
        doc_type = str(metadata.get("document_type") or doc.get("type") or "").lower()
        text = str(doc.get("document", "")).strip().replace("\n", " ")
        snippet = text[:240] + ("..." if len(text) > 240 else "")
        line = f"- [{idx}] ({source}) {snippet}"

        if "mistake" in doc_type or "error" in doc_type:
            mistakes.append(line)
        elif "email" in doc_type:
            emails.append(line)
        else:
            # Default bucket: keep usable as both style reference and caution case
            emails.append(line)

    if not mistakes:
        mistakes = ["- 관련 실수 사례를 찾지 못했습니다."]
    if not emails:
        emails = ["- 참고할 우수 이메일 사례를 찾지 못했습니다."]

    return {
        "retrieved_mistakes": "\n".join(mistakes),
        "retrieved_emails": "\n".join(emails),
    }


def _build_rule_based_review(
    email_content: str,
    recipient_country: str,
    purpose: str,
) -> str:
    from backend.agents.email_agent.tools import analyze_email_tone, detect_email_risks

    risks: List[Dict[str, Any]] = []
    tone_result: Dict[str, Any] = {}

    try:
        risks = detect_email_risks.invoke({"email_content": email_content}) or []
    except Exception as exc:
        logger.warning("Rule-based risk detection failed: %s", exc)

    try:
        tone_result = analyze_email_tone.invoke(
            {
                "email_content": email_content,
                "recipient_country": recipient_country,
                "purpose": purpose,
            }
        ) or {}
    except Exception as exc:
        logger.warning("Rule-based tone analysis failed: %s", exc)

    lines: List[str] = []
    lines.append("### 이메일 리뷰 요약")
    lines.append(f"- 검토 목적: {purpose}")
    lines.append(f"- 수신자 국가: {recipient_country}")

    if risks:
        lines.append(f"- 발견된 주요 리스크: {len(risks)}건")
        for idx, risk in enumerate(risks[:3], start=1):
            severity = str(risk.get("severity", "medium")).upper()
            risk_type = str(risk.get("type", "unknown"))
            recommendation = str(risk.get("recommendation", "표현을 구체화하세요."))
            lines.append(f"  {idx}. [{severity}] {risk_type} -> 권장: {recommendation}")
    else:
        lines.append("- 규칙 기반으로는 치명적 리스크가 감지되지 않았습니다.")

    if tone_result:
        current_tone = tone_result.get("current_tone", "unknown")
        recommended_tone = tone_result.get("recommended_tone", "professional")
        score = tone_result.get("score", "-")
        lines.append(f"- 톤 분석: 현재 {current_tone} / 권장 {recommended_tone} (점수 {score}/10)")
        summary = tone_result.get("summary")
        if summary:
            lines.append(f"- 톤 코멘트: {summary}")

    lines.append("- 보완 권장: 결제조건, Incoterms, 납기일, 수량/사양을 명시해 분쟁 가능성을 낮추세요.")
    return "\n".join(lines)

# --- Prompt Loader Function ---
def _load_prompt(prompt_file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # Correct path
    prompt_path = os.path.join(project_root, 'backend', 'prompts', prompt_file_name)
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Global Components for EmailAgent ---
class EmailAgentComponents:
    def __init__(self):
        self.settings = get_settings()
        self.system_prompt = _load_prompt("email_prompt.txt")
        self.prompt_templates = {
            "draft": _load_prompt("email/email_draft_prompt.txt"),
            "review": _load_prompt("email/email_review_prompt.txt"),
        }

        self.llm = None
        if self.settings.upstage_api_key:
            self.llm = AsyncOpenAI(
                base_url="https://api.upstage.ai/v1",
                api_key=self.settings.upstage_api_key
            )
        else:
            logger.warning("UPSTAGE_API_KEY is not set for EmailAgent. LLM calls will fail.")
        
        # Configure Langsmith tracing - if not configured by Orchestrator
        if self.settings.langsmith_tracing and self.settings.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.settings.langsmith_project
        # else: os.environ["LANGCHAIN_TRACING_V2"] is already set to "false" by orchestrator if not enabled

EMAIL_AGENT_COMPONENTS = EmailAgentComponents()


# --- Node Functions ---

def perform_rag_search_node(state: EmailGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    user_input = state_dict["user_input"]
    context = state_dict.get("context") or {}

    # Build search query
    rag_query = user_input
    if context.get("email_subject"):
        rag_query += f" {context['email_subject']}"
    if context.get("recipient_type"):
        rag_query += f" {context['recipient_type']}"

    # Use tool: search_email_references
    from backend.agents.email_agent.tools import search_email_references

    try:
        # Determine search type from mode
        task_type = _detect_email_task_type(user_input, context)
        search_type = "mistakes" if task_type == "review" else "all"

        retrieved_documents = search_email_references.invoke(
            {
                "query": rag_query,
                "k": 3,
                "search_type": search_type,
            }
        )
        used_rag = len(retrieved_documents) > 0
    except Exception as e:
        logger.warning("Error during email RAG search: %s", e)
        retrieved_documents = []
        used_rag = False

    state_dict["retrieved_documents"] = retrieved_documents
    state_dict["used_rag"] = used_rag

    # Format context string
    rag_context_str = ""
    if used_rag and retrieved_documents:
        rag_context_str = "\n--- 참조 문서 ---\n"
        for i, doc in enumerate(retrieved_documents):
            metadata = doc.get("metadata", {})
            rag_context_str += f"\n문서 {i+1} (출처: {metadata.get('source_dataset', 'unknown')} | 유형: {metadata.get('document_type', 'unknown')})\n"
            rag_context_str += f"{doc['document']}\n"

    state_dict["rag_context_str"] = rag_context_str

    return state_dict


def prepare_llm_messages_node(state: EmailGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    user_input = state_dict["user_input"]
    context = state_dict.get("context") or {}
    conversation_history = state_dict.get("conversation_history") or []
    retrieved_documents = state_dict.get("retrieved_documents") or []

    task_type = _detect_email_task_type(user_input, context)
    extracted_email_content = _extract_email_content(user_input, conversation_history, task_type)
    recipient_country = _extract_country(user_input, context)
    output_language = _detect_output_language(user_input, context, conversation_history)
    language_instruction = _build_language_instruction(output_language, task_type)
    purpose = str(
        context.get("purpose")
        or ("이메일 검토" if task_type == "review" else "이메일 초안 작성")
    ).strip()

    doc_blocks = _format_retrieved_docs(retrieved_documents)

    if task_type == "review" and not extracted_email_content:
        missing_message = (
            "검토할 이메일 본문이 필요합니다. "
            "검토할 이메일 전문을 붙여넣어 주세요."
        )
        state_dict["email_task_type"] = task_type
        state_dict["extracted_email_content"] = None
        state_dict["extracted_recipient_country"] = recipient_country
        state_dict["extracted_purpose"] = purpose
        state_dict["llm_messages"] = []
        state_dict["final_response_content"] = missing_message
        state_dict["llm_output_details"] = {
            "error": {
                "message": missing_message,
                "required_fields": ["email_content"],
            }
        }
        state_dict["model_used"] = "rule-based"
        return state_dict

    if task_type == "review":
        template = EMAIL_AGENT_COMPONENTS.prompt_templates["review"]
        system_prompt = (
            template
            .replace("{email_content}", extracted_email_content or "")
            .replace("{recipient_country}", recipient_country)
            .replace("{purpose}", purpose)
            .replace("{retrieved_mistakes}", doc_blocks["retrieved_mistakes"])
            .replace("{retrieved_emails}", doc_blocks["retrieved_emails"])
        )
        system_prompt += f"\n\n- {language_instruction}"
        if recipient_country == "미지정":
            system_prompt += (
                "\n- 수신자 국가 정보가 없으므로 특정 국가 문화권을 단정하지 말고 "
                "일반 무역 실무 기준으로만 검토하세요."
            )
        user_message = (
            "아래 이메일을 무역 실무 기준으로 검토하고 개선 포인트를 알려주세요.\n\n"
            f"{extracted_email_content}"
        )
    else:
        template = EMAIL_AGENT_COMPONENTS.prompt_templates["draft"]
        situation = str(context.get("situation") or user_input).strip()
        relationship = str(context.get("relationship") or context.get("recipient_type") or "business_partner").strip()
        system_prompt = (
            template
            .replace("{user_input}", user_input)
            .replace("{situation}", situation)
            .replace("{recipient_country}", recipient_country)
            .replace("{relationship}", relationship)
            .replace("{retrieved_emails}", doc_blocks["retrieved_emails"])
        )
        system_prompt += f"\n\n- {language_instruction}"
        if _is_follow_up_edit_request(user_input) and extracted_email_content:
            user_message = (
                f"요청사항: {user_input}\n"
                "기존 이메일 초안을 기반으로 수정/번역하세요.\n"
                f"기존 초안:\n{extracted_email_content}\n\n"
                f"수신자 국가: {recipient_country}"
            )
        else:
            user_message = (
                f"요청사항: {user_input}\n"
                f"상황: {situation}\n"
                f"수신자 국가: {recipient_country}"
            )

    state_dict["email_task_type"] = task_type
    state_dict["extracted_email_content"] = extracted_email_content
    state_dict["extracted_recipient_country"] = recipient_country
    state_dict["extracted_purpose"] = purpose
    state_dict["llm_messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return state_dict


async def call_llm_and_parse_response_node(state: EmailGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    # Early return when previous node already produced a direct response
    if state_dict.get("final_response_content"):
        if not state_dict.get("llm_output_details"):
            state_dict["llm_output_details"] = {"source": "prepare_llm_messages_node"}
        if not state_dict.get("model_used"):
            state_dict["model_used"] = "rule-based"
        return state_dict

    llm = EMAIL_AGENT_COMPONENTS.llm
    llm_messages = state_dict.get("llm_messages") or []
    task_type = str(state_dict.get("email_task_type") or "draft")
    extracted_email_content = str(state_dict.get("extracted_email_content") or "")
    recipient_country = str(state_dict.get("extracted_recipient_country") or "미지정")
    purpose = str(state_dict.get("extracted_purpose") or "이메일 작성")

    llm_response_content = "LLM 호출에 실패했거나 응답이 없습니다."
    model_used = "solar-pro2"
    llm_output_details = {}
    
    if llm and llm_messages:
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
                if isinstance(parsed_llm_response, dict):
                    error_field = parsed_llm_response.get("error")
                    if isinstance(error_field, dict):
                        final_response_content = error_field.get(
                            "message",
                            json.dumps(error_field, ensure_ascii=False),
                        )
                    elif isinstance(error_field, str):
                        final_response_content = error_field
                    elif isinstance(parsed_llm_response.get("email_response"), str):
                        final_response_content = parsed_llm_response["email_response"]
                    elif isinstance(parsed_llm_response.get("email_content"), str):
                        if task_type == "draft":
                            final_response_content = parsed_llm_response["email_content"]
                        else:
                            final_response_content = _build_rule_based_review(
                                extracted_email_content or parsed_llm_response["email_content"],
                                recipient_country,
                                purpose,
                            )
                            llm_output_details = {
                                **llm_output_details,
                                "fallback": "rule_based_review_due_to_email_content_only",
                            }
                    elif isinstance(parsed_llm_response.get("message"), str):
                        final_response_content = parsed_llm_response["message"]
                    else:
                        required_fields = parsed_llm_response.get("required_fields")
                        if isinstance(required_fields, list) and required_fields:
                            joined_fields = ", ".join(str(field) for field in required_fields)
                            final_response_content = f"필수 입력 정보가 부족합니다: {joined_fields}"
                        else:
                            final_response_content = json.dumps(parsed_llm_response, ensure_ascii=False)
                else:
                    final_response_content = llm_response_content
            except json.JSONDecodeError:
                logger.debug(
                    "EmailAgent LLM response was not valid JSON (truncated): %s",
                    llm_response_content[:200],
                )
                final_response_content = llm_response_content
                llm_output_details = {"raw_llm_response": llm_response_content}

            if (
                task_type == "review"
                and extracted_email_content
                and final_response_content.strip() == extracted_email_content.strip()
            ):
                final_response_content = _build_rule_based_review(
                    extracted_email_content,
                    recipient_country,
                    purpose,
                )
                llm_output_details = {
                    **llm_output_details,
                    "fallback": "rule_based_review_due_to_echo_response",
                }

            if task_type == "review" and recipient_country == "미지정":
                inferred_country_keywords = ["미국", "일본", "중국", "사우디", "독일", "유럽", "중동"]
                if any(keyword in final_response_content for keyword in inferred_country_keywords):
                    final_response_content = _build_rule_based_review(
                        extracted_email_content,
                        recipient_country,
                        purpose,
                    )
                    llm_output_details = {
                        **llm_output_details,
                        "fallback": "rule_based_review_due_to_unknown_country_assumption",
                    }
            
        except openai.APIError as e:
            logger.warning("EmailAgent Upstage API error: %s", e)
            if task_type == "review" and extracted_email_content:
                final_response_content = _build_rule_based_review(
                    extracted_email_content,
                    recipient_country,
                    purpose,
                )
            else:
                final_response_content = f"LLM API 호출 중 오류가 발생했습니다: {e}"
            llm_output_details = {"error": str(e)}
        except Exception as e:
            logger.warning("EmailAgent unexpected LLM error: %s", e)
            if task_type == "review" and extracted_email_content:
                final_response_content = _build_rule_based_review(
                    extracted_email_content,
                    recipient_country,
                    purpose,
                )
            else:
                final_response_content = f"LLM 호출 중 예상치 못한 오류가 발생했습니다: {e}"
            llm_output_details = {"error": str(e)}
    else:
        if task_type == "review" and extracted_email_content:
            final_response_content = _build_rule_based_review(
                extracted_email_content,
                recipient_country,
                purpose,
            )
            llm_output_details = {"fallback": "rule_based_review_without_llm"}
            model_used = "rule-based"
        else:
            final_response_content = "LLM 클라이언트가 초기화되지 않아 응답을 생성할 수 없습니다. UPSTAGE_API_KEY를 확인하세요."
            llm_output_details = {"error": "LLM client not initialized due to missing API key."}

    state_dict["final_response_content"] = final_response_content
    state_dict["llm_output_details"] = llm_output_details
    state_dict["model_used"] = model_used # Store model used for metadata
    return state_dict


def format_email_output_node(state: EmailGraphState) -> Dict[str, Any]:
    state_dict = cast(Dict[str, Any], state)

    final_response = state_dict["final_response_content"]
    used_rag = state_dict.get("used_rag", False)
    retrieved_documents = state_dict.get("retrieved_documents", [])
    model_used = state_dict.get("model_used", "solar-pro2")
    llm_messages = state_dict.get("llm_messages", [])
    llm_output_details = state_dict.get("llm_output_details", {})
    context = state_dict["context"]
    user_input = state_dict["user_input"]
    email_task_type = state_dict.get("email_task_type", "draft")


    metadata = {
        "used_rag": used_rag,
        "documents": retrieved_documents,
        "model": model_used,
        "llm_input_prepared": llm_messages,
        "llm_output_details": llm_output_details,
        "input_context": context,
        "processed_input": user_input,
        "task_type": email_task_type,
    }

    state_dict["agent_output_for_orchestrator"] = {
        "response": final_response,
        "agent_type": EMAIL_AGENT_TYPE,
        "metadata": metadata
    }
    return state_dict
