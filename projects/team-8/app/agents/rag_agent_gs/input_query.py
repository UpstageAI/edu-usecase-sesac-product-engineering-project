#from rich import print as rprint
from mock import VetAgentMockState
from mock import create_mock_vet_agent_state

# 수의사 에이전트 목데이터 생성===============================
mock_state = create_mock_vet_agent_state()
# rprint(mock_state.model_dump())

def build_insurance_query(state: VetAgentMockState) -> str:
    # 질환 이름만 추출 (최대 3개)
    disease_names = [d.name for d in state.diseases[:3]]

    disease_text = ", ".join(disease_names)

    query = (
        f"{state.age}세 {state.breed} {state.gender.value} "
        f"{state.species}({state.weight}kg)이며 "
        f"{disease_text} 등의 질환 위험이 있는 경우 "
        f"적절한 반려동물 보험 상담을 받고 싶습니다."
    )
    # query=input("반려동물에 관한 질문을 입력하세요: ")
    return query
