from app.agents.user_input_template_agent.state import UserInputTemplateState
from app.agents.user_input_template_agent.middleware import sanitize_user_input


def user_input_template_node(
    state: UserInputTemplateState,
) -> dict:
    """
    펫보험 상품 추천을 위한 고객 입력 템플릿 노드
    - 미들웨어를 통해 자유 텍스트 필드를 정화(sanitize)한다.
    - 프롬프트 인젝션 감지 시 is_blocked 플래그를 설정한다.
    """
    result = sanitize_user_input(state)
    if result.has_changes:
        for log in result.logs:
            print(
                f"  [Guardrail] {log.field}: "
                f'"{log.original}" -> "{log.sanitized}" ({log.actions})'
            )

    output = result.state_dict

    if result.has_injection:
        blocked_fields = [log.field for log in result.logs if "injection_detected_field_nullified" in log.actions]
        output["is_blocked"] = True
        output["blocked_reason"] = (
            f"프롬프트 인젝션이 감지되어 서비스가 차단되었습니다. "
            f"(감지 필드: {', '.join(blocked_fields)})"
        )
        print(f"  [Guardrail] BLOCKED: {output['blocked_reason']}")

    return output


if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.user_input_template_agent.state import UserInputTemplateState
    from app.agents.user_input_template_agent.utils.cli import (
        create_arg_parser,
        load_state_from_yaml,
    )

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)
    result = user_input_template_node(state)
    rprint(result)
