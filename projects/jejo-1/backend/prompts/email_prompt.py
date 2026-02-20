"""
Email Agent 프롬프트 로딩 유틸리티
"""
import os
from typing import Dict


def load_prompt(prompt_type: str) -> str:
    """
    프롬프트 파일 로딩

    Args:
        prompt_type: "draft" | "review" | "risk" | "tone" | "improvement"

    Returns:
        프롬프트 템플릿 (str)

    Raises:
        ValueError: 알 수 없는 prompt_type
        FileNotFoundError: 프롬프트 파일이 없을 때
    """
    # 프롬프트 파일 매핑
    prompt_files = {
        "draft": "backend/prompts/email/email_draft_prompt.txt",
        "review": "backend/prompts/email/email_review_prompt.txt",
        "risk": "backend/prompts/email/email_risk_prompt.txt",
        "tone": "backend/prompts/email/email_tone_prompt.txt",
        "improvement": "backend/prompts/email/email_improvement_prompt.txt"
    }

    if prompt_type not in prompt_files:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. "
            f"Available types: {list(prompt_files.keys())}"
        )

    file_path = prompt_files[prompt_type]

    # 프로젝트 루트 기준 경로 (backend 상위 디렉토리)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_path = os.path.join(project_root, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def load_all_prompts() -> Dict[str, str]:
    """
    모든 이메일 프롬프트 로딩

    Returns:
        {
            "draft": str,
            "review": str,
            "risk": str,
            "tone": str,
            "improvement": str
        }
    """
    return {
        "draft": load_prompt("draft"),
        "review": load_prompt("review"),
        "risk": load_prompt("risk"),
        "tone": load_prompt("tone"),
        "improvement": load_prompt("improvement")
    }


if __name__ == '__main__':
    # 테스트
    print("=== Email Prompt Loading Test ===\n")

    try:
        # 각 프롬프트 로딩 테스트
        for prompt_type in ["draft", "review", "risk", "tone", "improvement"]:
            prompt = load_prompt(prompt_type)
            print(f"✅ {prompt_type} prompt loaded: {len(prompt)} characters")
            print(f"   Preview: {prompt[:100]}...\n")

        # 전체 로딩 테스트
        all_prompts = load_all_prompts()
        print(f"✅ All prompts loaded: {len(all_prompts)} prompts")

    except Exception as e:
        print(f"❌ Error: {e}")
