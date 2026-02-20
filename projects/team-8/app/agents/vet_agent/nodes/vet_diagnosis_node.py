import json

from app.agents.vet_agent.model.model import llm
from app.agents.vet_agent.state import VetAgentOutputState, VetAgentState


def vet_diagnosis_node(state: VetAgentState) -> dict:
    """반려동물 취약 질병 정보를 LLM으로 분석하는 노드"""
    structured_llm = llm.with_structured_output(VetAgentOutputState)
    input_summary = json.dumps(
        state.model_dump(
            exclude={"diseases", "retry_count", "is_validated", "validation_feedback"},
            exclude_none=True,
        ),
        ensure_ascii=False,
        indent=2,
    )

    # 피드백이 있으면 프롬프트에 추가 (재시도 시)
    feedback_section = ""
    if state.validation_feedback:
        feedback_section = (
            f"\n\n이전 생성 결과에 다음 문제가 있었습니다:\n"
            f"{state.validation_feedback}\n"
            f"위 피드백을 반영하여 다시 생성해주세요."
        )

    result = structured_llm.invoke(
        f"""당신은 수의사입니다.
다음 반려동물 정보를 기반으로 해당 반려동물이 잘 걸리는 질병을 **3~5개** 추출해주세요.
(유저가 입력한 건강 정보가 있으면 우선 포함하고, 부족한 경우 품종/나이 기반 예상 질병으로 보완하세요.)

각 질병에 대해 질병명, 발병률, 발병시기를 포함해주세요.

반려동물 정보:
{input_summary}{feedback_section}"""
    )
    return {"diseases": result.diseases}


if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.vet_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, VetAgentState)
    result = vet_diagnosis_node(state)
    rprint(result)
