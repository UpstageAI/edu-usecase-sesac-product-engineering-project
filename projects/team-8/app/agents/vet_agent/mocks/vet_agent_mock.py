from enum import Enum

from pydantic import BaseModel, Field


# 성별 Enum
class Gender(str, Enum):
    male = "male"
    female = "female"


# 질병 정보
class DiseaseInfo(BaseModel):
    name: str = Field(description="질병명")
    incidence_rate: str = Field(description="발병률")
    onset_period: str = Field(description="발병시기")


# 수의사 에이전트 목데이터
class VetAgentMockState(BaseModel):
    
    species: str = Field(description="종")
    breed: str = Field(description="품종")
    age: int = Field(description="나이")
    gender: Gender = Field(description="성별")
    weight: int = Field(description="체중(kg)")
    diseases: list[DiseaseInfo] = Field(default_factory=list, description="질병 목록")


def create_mock_vet_agent_state() -> VetAgentMockState:
    """수의사 에이전트의 목데이터를 생성합니다.

    Returns:
        VetAgentMockState: 치와와 기반 샘플 진단 결과가 포함된 목데이터
    """
    return VetAgentMockState(
        species="강아지",
        breed="치와와",
        age=10,
        gender=Gender.male,
        weight=10,
        diseases=[
            DiseaseInfo(
                name="슬개골 탈구",
                incidence_rate="높음",
                onset_period="전 연령",
            ),
            DiseaseInfo(
                name="심장판막증",
                incidence_rate="중간",
                onset_period="7세 이상",
            ),
            DiseaseInfo(
                name="치과 질환",
                incidence_rate="중간",
                onset_period="5세 이상",
            ),
            DiseaseInfo(
                name="간질",
                incidence_rate="낮음",
                onset_period="1-5세",
            ),
        ],
    )


if __name__ == "__main__":
    from rich import print as rprint

    mock_state = create_mock_vet_agent_state()
    rprint(mock_state.model_dump())
