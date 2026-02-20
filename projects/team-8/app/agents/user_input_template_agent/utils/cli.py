import argparse
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def create_arg_parser() -> argparse.ArgumentParser:
    """펫보험 상품 추천을 위한 CLI 인자 파서를 생성합니다."""
    parser = argparse.ArgumentParser(
        description="펫보험 상품 추천을 위한 고객 입력 템플릿"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="app/agents/user_input_template_agent/samples/user_input_simple.yaml",
        help="입력 YAML 파일 경로 (기본값: app/agents/user_input_template_agent/samples/user_input_simple.yaml)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="test_user",
        help="LangGraph thread ID (기본값: test_user)",
    )
    return parser


def make_config(thread_id: str) -> dict:
    """LangGraph 실행용 config dict를 생성합니다."""
    return {"configurable": {"thread_id": thread_id}}


def load_state_from_yaml(path: str | Path, state_type: type[T]) -> T:
    """YAML 파일을 읽어 지정된 state 타입으로 변환합니다."""
    yaml_path = Path(path)
    text: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = yaml_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(
            f"Failed to decode YAML file with utf-8-sig/utf-8/cp949: {yaml_path}"
        )

    data = yaml.safe_load(text)
    state_data = data.get("state", data)
    return state_type.model_validate(state_data)
