from pathlib import Path
import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

ENV_PATH = Path(__file__).parents[4] / ".env"
CONFIG_PATH = Path(__file__).parent / "model.yaml"

load_dotenv(ENV_PATH)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """model.yaml에서 모델 설정을 불러옵니다."""
    return yaml.safe_load(path.read_text())


def create_llm(config: dict | None = None):
    """설정 기반으로 LLM 인스턴스를 생성합니다."""
    if config is None:
        config = load_config()
    return init_chat_model(
        model=config["model"],
        temperature=config.get("temperature", 0.1),
    )


llm = create_llm()

if __name__ == "__main__":
    from rich import print as rprint

    config = load_config()
    rprint(config)
    rprint(llm)
