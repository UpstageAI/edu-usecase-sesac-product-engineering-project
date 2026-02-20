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
        default="user_input_template_agent/samples/user_input_simple.yaml",
        help="입력 YAML 파일 경로 (기본값: user_input_template_agent/samples/user_input_simple.yaml)",
    )
    return parser


def load_state_from_yaml(path: str | Path, state_type: type[T]) -> T:
    """YAML 파일을 읽어 지정된 state 타입으로 변환합니다."""
    data = yaml.safe_load(Path(path).read_text())
    state_data = data.get("state", data)
    return state_type.model_validate(state_data)
