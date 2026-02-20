"""Orchestrator 그래프의 시나리오 기반 테스트 모듈.

실행 예시:
    uv run python -m app.agents.orchestrator.test_orchestrator --scenario router
    uv run python -m app.agents.orchestrator.test_orchestrator --scenario single
    uv run python -m app.agents.orchestrator.test_orchestrator --input path/to/custom.yaml
"""

from app.agents.orchestrator.orchestrator_graph import run_orchestration
from app.agents.user_input_template_agent.utils.cli import (
    create_arg_parser,
    make_config,
)

SCENARIOS: dict[str, list[str]] = {
    "router": [
        "app/agents/user_input_template_agent/samples/user_input_insurer_1.yaml",
        "app/agents/user_input_template_agent/samples/user_input_insurer_2.yaml",
    ],
    "single": [
        "app/agents/user_input_template_agent/samples/user_input_simple.yaml",
    ],
}


def run_scenario(scenario_name: str, config: dict) -> list[dict]:
    """시나리오에 정의된 YAML 파일들을 순차 실행합니다."""
    yaml_paths = SCENARIOS[scenario_name]
    results = []
    for i, yaml_path in enumerate(yaml_paths, 1):
        print(f"[{scenario_name} {i}/{len(yaml_paths)}] {yaml_path}")
        result = run_orchestration(yaml_path, config=config)
        results.append(result)
    return results


def create_test_arg_parser():
    """테스트 전용 CLI 파서. 공통 파서를 확장합니다."""
    parser = create_arg_parser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIOS.keys(),
        help=f"실행할 테스트 시나리오 ({', '.join(SCENARIOS.keys())})",
    )
    return parser


def main():
    args = create_test_arg_parser().parse_args()
    config = make_config(args.thread_id)

    if args.scenario:
        run_scenario(args.scenario, config)
    else:
        print(f"Testing {args.input}")
        run_orchestration(args.input, config=config)


if __name__ == "__main__":
    main()
