import json
import re
from typing import Any, Dict, Iterable, List, Optional


def _first_non_empty_str(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_risk_factors(raw_factors: Any) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw_factors, list):
        factor_items = [(None, factor) for factor in raw_factors]
    elif isinstance(raw_factors, dict):
        factor_items = list(raw_factors.items())
    else:
        factor_items = []

    for idx, (factor_key, factor_data) in enumerate(factor_items):
        if not isinstance(factor_data, dict):
            continue

        name_kr = _first_non_empty_str(
            factor_data.get("name_kr"),
            factor_data.get("name"),
            factor_key,
            f"factor_{idx}",
        ) or f"factor_{idx}"

        impact = factor_data.get("impact")
        likelihood = factor_data.get("likelihood")
        score = _to_float(factor_data.get("score"))
        if score is None:
            score = _to_float(factor_data.get("risk_score"))
        if score is None:
            impact_num = _to_float(impact)
            likelihood_num = _to_float(likelihood)
            if impact_num is not None and likelihood_num is not None:
                score = impact_num * likelihood_num
            else:
                score = 0.0

        normalized[name_kr] = {
            **factor_data,
            "name_kr": name_kr,
            "impact": int(impact) if isinstance(impact, int) else impact,
            "likelihood": int(likelihood) if isinstance(likelihood, int) else likelihood,
            "score": float(score),
            "reason": _first_non_empty_str(
                factor_data.get("reason"),
                factor_data.get("reasoning"),
            ) or "",
            "mitigation_suggestions": _as_string_list(
                factor_data.get("mitigation_suggestions")
            ),
        }

    return normalized


def _parse_json_flexible(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    from backend.utils.json_utils import safe_json_parse
    try:
        return safe_json_parse(text)
    except:
        return None


def _format_quiz_chat_message(payload: Any) -> Optional[str]:
    questions: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        questions = [item for item in payload if isinstance(item, dict) and item.get("question")]
    elif isinstance(payload, dict):
        if isinstance(payload.get("questions"), list):
            questions = [
                item
                for item in payload.get("questions", [])
                if isinstance(item, dict) and item.get("question")
            ]
        elif isinstance(payload.get("answer"), list):
            questions = [
                item
                for item in payload.get("answer", [])
                if isinstance(item, dict) and item.get("question")
            ]

    if not questions:
        return None

    first = questions[0]
    lines = [f"[퀴즈 1]", str(first.get("question", "")).strip()]
    choices = first.get("choices", [])
    if isinstance(choices, list):
        for idx, choice in enumerate(choices[:4], start=1):
            lines.append(f"{idx}. {choice}")
    return "\n".join(line for line in lines if line)


def _extract_quiz_questions(payload: Any) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        questions = [item for item in payload if isinstance(item, dict) and item.get("question")]
    elif isinstance(payload, dict):
        if isinstance(payload.get("questions"), list):
            questions = [
                item
                for item in payload.get("questions", [])
                if isinstance(item, dict) and item.get("question")
            ]
        elif isinstance(payload.get("answer"), list):
            questions = [
                item
                for item in payload.get("answer", [])
                if isinstance(item, dict) and item.get("question")
            ]
    return questions


def normalize_response(raw: Any) -> Dict[str, Any]:
    """
    모든 agent 응답을 통일된 ChatResponse dict로 변환
    """

    # 기본 구조 설정
    default_report = {
        "analysis_id": "N/A",
        "risk_scoring": {
            "overall_risk_level": "Unknown",
            "overall_risk_score": 0.0,
            "risk_factors": {} # app.py가 items()를 쓸 수 있게 dict로 유지
        },
        "response_summary": "요약 정보가 없습니다.",
        "suggested_actions": [],
        "similar_cases": [],
        "evidence_sources": []
    }

    # Identify if it's a report or a simple chat message
    extracted_report_dict = None
    chat_message_content = str(raw) # Default to raw as string
    chat_meta: Dict[str, Any] = {}

    if isinstance(raw, dict):
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            llm_output_details = metadata.get("llm_output_details")
            if isinstance(llm_output_details, dict):
                questions = _extract_quiz_questions(llm_output_details)
                if questions:
                    chat_meta["quiz_questions"] = questions

        if "type" in raw: # Already normalized ChatResponse
            return raw
        
        # Check for report data in 'report' key
        if isinstance(raw.get("report"), dict):
            extracted_report_dict = raw.get("report")
        # Check for report data or message in 'response' key
        elif raw.get("response") is not None:
            if isinstance(raw.get("response"), dict): # Agent returned report dict directly in 'response'
                extracted_report_dict = raw.get("response")
            elif isinstance(raw.get("response"), str):
                chat_message_content = raw.get("response")
                try: # Try to parse 'response' string as JSON report
                    parsed_str = _parse_json_flexible(chat_message_content)
                    if isinstance(parsed_str, dict) and "analysis_id" in parsed_str:
                        extracted_report_dict = parsed_str
                    elif parsed_str is not None:
                        parsed_questions = _extract_quiz_questions(parsed_str)
                        if parsed_questions:
                            chat_meta["quiz_questions"] = parsed_questions
                        quiz_message = _format_quiz_chat_message(parsed_str)
                        if quiz_message:
                            chat_message_content = quiz_message
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw.get("response"), list):
                parsed_questions = _extract_quiz_questions(raw.get("response"))
                if parsed_questions:
                    chat_meta["quiz_questions"] = parsed_questions
                quiz_message = _format_quiz_chat_message(raw.get("response"))
                if quiz_message:
                    chat_message_content = quiz_message
        # Check for message in 'message' key (fallback for some agent types)
        elif isinstance(raw.get("message"), str):
            chat_message_content = raw.get("message")
            try: # Try to parse 'message' string as JSON report
                parsed_str = _parse_json_flexible(chat_message_content)
                if isinstance(parsed_str, dict) and "analysis_id" in parsed_str:
                    extracted_report_dict = parsed_str
                elif parsed_str is not None:
                    parsed_questions = _extract_quiz_questions(parsed_str)
                    if parsed_questions:
                        chat_meta["quiz_questions"] = parsed_questions
                    quiz_message = _format_quiz_chat_message(parsed_str)
                    if quiz_message:
                        chat_message_content = quiz_message
            except json.JSONDecodeError:
                pass
        else: # If raw is a dict but no specific keys, use raw as chat message
            chat_message_content = str(raw)

    elif isinstance(raw, str): # Raw is a string
        chat_message_content = raw
        try: # Try to parse raw string as JSON report
            parsed_str = _parse_json_flexible(raw)
            if isinstance(parsed_str, dict) and "analysis_id" in parsed_str:
                extracted_report_dict = parsed_str
            elif parsed_str is not None:
                parsed_questions = _extract_quiz_questions(parsed_str)
                if parsed_questions:
                    chat_meta["quiz_questions"] = parsed_questions
                quiz_message = _format_quiz_chat_message(parsed_str)
                if quiz_message:
                    chat_message_content = quiz_message
        except json.JSONDecodeError:
            pass

    # Process if a report dict was successfully extracted
    if extracted_report_dict:
        report_source = extracted_report_dict

        # 2. 기본 필드 매핑 (신/구 리포트 스키마 모두 지원)
        summary = _first_non_empty_str(
            report_source.get("response_summary"),
            report_source.get("summary"),
            report_source.get("input_summary"),
            (report_source.get("risk_scoring") or {}).get("overall_assessment")
            if isinstance(report_source.get("risk_scoring"), dict) else None,
        )
        if summary:
            default_report["response_summary"] = summary

        analysis_id = _first_non_empty_str(
            report_source.get("analysis_id"),
            report_source.get("id"),
        )
        if analysis_id:
            default_report["analysis_id"] = analysis_id

        if isinstance(report_source.get("similar_cases"), list):
            default_report["similar_cases"] = report_source.get("similar_cases", [])

        if isinstance(report_source.get("evidence_sources"), list):
            default_report["evidence_sources"] = report_source.get("evidence_sources", [])

        # 3. 리스크 스코어링 및 요인(Factors) 정규화
        scoring_source = report_source.get("risk_scoring") or report_source
        normalized_factors: Dict[str, Dict[str, Any]] = {}
        if isinstance(scoring_source, dict):
            default_report["risk_scoring"]["overall_risk_level"] = scoring_source.get("overall_risk_level", "Unknown")

            scoring_level_factors = _normalize_risk_factors(
                scoring_source.get("risk_factors", [])
            )
            report_level_factors = _normalize_risk_factors(
                report_source.get("risk_factors", {})
            )

            # Merge factors from both shapes.
            # Priority: keep richer text fields from scoring-level factors, but allow
            # report-level fields (e.g. name_kr/score) to override when present.
            normalized_factors = dict(scoring_level_factors)
            for factor_name, report_factor in report_level_factors.items():
                existing = normalized_factors.get(factor_name, {})
                merged = {**existing, **report_factor}

                if not _first_non_empty_str(merged.get("reason")):
                    merged["reason"] = _first_non_empty_str(existing.get("reason")) or ""
                if not merged.get("mitigation_suggestions"):
                    merged["mitigation_suggestions"] = _as_string_list(
                        existing.get("mitigation_suggestions")
                    )

                merged_score = _to_float(merged.get("score"))
                existing_score = _to_float(existing.get("score"))
                if (merged_score is None or merged_score == 0.0) and existing_score is not None:
                    merged["score"] = existing_score

                normalized_factors[factor_name] = merged

            default_report["risk_scoring"]["risk_factors"] = normalized_factors

            overall_score = _to_float(scoring_source.get("overall_risk_score"))
            if overall_score is None:
                factor_scores = [
                    _to_float(factor.get("score"))
                    for factor in normalized_factors.values()
                    if isinstance(factor, dict)
                ]
                numeric_scores = [score for score in factor_scores if score is not None]
                if numeric_scores:
                    overall_score = sum(numeric_scores) / len(numeric_scores)
                else:
                    overall_score = 0.0
            default_report["risk_scoring"]["overall_risk_score"] = float(overall_score)

        # 4. 제안 조치 정규화: 신/구 스키마 + 파생값 모두 수용
        suggested_actions = _as_string_list(report_source.get("suggested_actions"))
        if not suggested_actions:
            suggested_actions = _as_string_list(report_source.get("recommendations"))

        if not suggested_actions:
            prevention_strategy = report_source.get("prevention_strategy")
            if isinstance(prevention_strategy, dict):
                suggested_actions.extend(_as_string_list(prevention_strategy.get("short_term")))
                suggested_actions.extend(_as_string_list(prevention_strategy.get("long_term")))

        control_gap_analysis = report_source.get("control_gap_analysis")
        if isinstance(control_gap_analysis, dict):
            suggested_actions.extend(_as_string_list(control_gap_analysis.get("recommendations")))

        for factor in normalized_factors.values():
            if isinstance(factor, dict):
                suggested_actions.extend(_as_string_list(factor.get("mitigation_suggestions")))

        default_report["suggested_actions"] = _dedupe_preserve_order(suggested_actions)

        # Return as report
        return {
            "type": "report",
            "message": default_report["response_summary"],
            "report": default_report,
            "meta": chat_meta
        }
    else:
        # It's a chat message
        return {
            "type": "chat",
            "message": chat_message_content,
            "report": None,
            "meta": chat_meta
        }
