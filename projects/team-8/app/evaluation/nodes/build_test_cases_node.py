"""테스트 케이스 생성 노드: 질병-약관 매핑 쌍으로 EvaluationTestCase 리스트 생성.

기존 카테시안 곱(5질병 × 25약관 = 125개) 대신,
질병별로 해당 질병에 대해 RAG가 검색한 약관만 매핑합니다.
(5질병 × 질병당 5약관 = 25개)
"""

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState
from app.evaluation.state import EvaluationTestCase

# (질병, [해당 질병의 약관 텍스트들]) 쌍 리스트
DiseasePolicyPairs = list[tuple[DiseaseInfo, list[str]]]


def build_test_cases(
    file_name: str,
    state: VetAgentState,
    disease_policy_pairs: DiseasePolicyPairs,
) -> list[EvaluationTestCase]:
    """질병-약관 매핑 쌍에 대해 EvaluationTestCase 리스트를 생성합니다.

    각 질병은 자신에게 매핑된 약관들과만 조합됩니다.
    예: 5 질병 × 질병당 5 약관 = 25개 테스트 케이스
    """
    hc = state.health_condition
    disease_history = (hc.disease_surgery_history if hc else None) or "없음"

    test_cases: list[EvaluationTestCase] = []

    for disease, policy_texts in disease_policy_pairs:
        for policy_text in policy_texts:
            test_cases.append(
                EvaluationTestCase(
                    file_name=file_name,
                    species=state.species or "미상",
                    breed=state.breed or "미상",
                    age=state.age or 0,
                    disease_surgery_history=disease_history,
                    disease_name=disease.name,
                    policy_text=policy_text,
                )
            )

    return test_cases
