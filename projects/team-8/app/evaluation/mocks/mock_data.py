"""
Mock 약관 텍스트 및 Mock 질병 목록 (API 비용 절감용).

RAG 결과 양식 설명:
  - 현재 app.agents.rag_agent 는 이 양식을 사용하지 않습니다. RAG는 LangChain Document(page_content, metadata)
    와 Qdrant dense 검색 결과를 사용하며, summary_multi에서 evaluation 메타데이터를 붙입니다.
  - 아래 RagChunkResult 형식은 RAG 담당자가 제공한 'Sparse score results - top 5 chunks' 양식에 맞춘
    것입니다. 실제 RAG 연동 시 이 구조로 응답이 오면 chunk_content 를 약관 원문으로 사용하면 됩니다.
"""

from typing import TypedDict

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState


# RAG 담당자 양식: Sparse score results - top N chunks
class RagChunkResult(TypedDict, total=False):
    """RAG 검색 결과 청크 1건. 실제 연동 시 응답 필드에 맞춰 확장 가능."""

    file: str  # 예: samsung_1_dog_anypet_4_47.pdf
    page: int
    chunk_id: str  # 예: chunk_000113
    chunk_content: str
    query_tokens: list[str]
    matched_vocab_tokens: list[str]
    unmatched_vocab_tokens: list[str]
    evaluation: tuple[float, float]  # 예: (70, 15.0)


# RAG 담당자 예시 기준 chunk_content 원문 3건 (실제 청크 내용으로 Mock 구성)
MOCK_RAG_CHUNKS: list[RagChunkResult] = [
    {
        "file": "samsung_1_dog_anypet_4_47.pdf",
        "page": 1,
        "chunk_id": "chunk_000113",
        "chunk_content": """② 제1항의 수술비용 확대보장에 대한 회사의 보장은 보험개시일로부터 30일 이내(이하"대기기간")에 발생한 질병(단, 암, 백내장, 녹내장, 심장질환, 신장질환, 방광질환 및 각종결석의 대기기간은 90일 적용)으로 인한 손해는 보상하여 드리지 않습니다. 단, 이 수술비용 확대보장 특별약관을 갱신하는 경우에는 적용하지 않습니다.""",
        "query_tokens": [
            "다리",
            "부상",
            "피부병",
            "심장",
            "판막증",
            "치과",
            "질환",
            "보장",
            "범위",
            "수술",
            "이력",
            "없",
            "경우",
            "적용",
            "가능",
            "보험",
            "약관",
        ],
        "matched_vocab_tokens": [
            "부상",
            "피부병",
            "심장",
            "치과",
            "질환",
            "보장",
            "범위",
            "수술",
            "없",
            "경우",
            "적용",
            "가능",
            "보험",
            "약관",
        ],
        "unmatched_vocab_tokens": ["다리", "판막증", "이력"],
        "evaluation": (70, 15.0),
    },
    {
        "file": "samsung_1_dog_anypet_4_47.pdf",
        "page": 1,
        "chunk_id": "chunk_000116",
        "chunk_content": """【수술비용 확대보장 보험금】 아래 ①과 ② 중 적은 금액 ① (피보험자가 부담한 수술당일치료비 - 보통약관에서 지급한 치료비보험금) × 보상비율 ② 보험증권에서 정한 1일당 수술비용 확대보장 보상한도액 ② 보험기간 중에 발생한 사고로 회사가 지급하는 수술비용 확대보장 보험금의 총 한도는 보험증권에 기재된 총보상횟수를 한도로 합니다. 제3조(준용규정) 이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다. 피부병 보장 특별약관 제1조(보상하는 손해)""",
        "query_tokens": [
            "다리",
            "부상",
            "피부병",
            "심장",
            "판막증",
            "치과",
            "질환",
            "보장",
            "범위",
            "수술",
            "이력",
            "없",
            "경우",
            "적용",
            "가능",
            "보험",
            "약관",
        ],
        "matched_vocab_tokens": [
            "부상",
            "피부병",
            "심장",
            "치과",
            "질환",
            "보장",
            "범위",
            "수술",
            "없",
            "경우",
            "적용",
            "가능",
            "보험",
            "약관",
        ],
        "unmatched_vocab_tokens": ["다리", "판막증", "이력"],
        "evaluation": (45, 15.0),
    },
    {
        "file": "meritz_1_maum_pet_12_61.pdf",
        "page": 3,
        "chunk_id": "chunk_000013",
        "chunk_content": """① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 질병 또는 상해가 발생하여 그 치료를 직접적인 목적으로 동물병원에 통원 또는 입원하여 수의사에게 치료를 받은 때에는 피보험자가 부담한 반려동물의 치료비를 이 약관에 따라 피보험자에게 치료비보험금으로 보상하여 드립니다. 단, 동물병원에서 수의사에게 수술을 받은 경우 수술당일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다.""",
        "query_tokens": [
            "다리",
            "부상",
            "피부병",
            "심장",
            "판막증",
            "치과",
            "질환",
            "보장",
            "범위",
            "수술",
            "이력",
            "없",
            "경우",
            "적용",
            "가능",
            "보험",
            "약관",
        ],
        "matched_vocab_tokens": [
            "부상",
            "치과",
            "질환",
            "보장",
            "범위",
            "수술",
            "없",
            "경우",
            "적용",
            "가능",
            "보험",
            "약관",
        ],
        "unmatched_vocab_tokens": ["다리", "피부병", "심장", "판막증", "이력"],
        "evaluation": (45, 13.0),
    },
]


def get_mock_policies() -> list[str]:
    """Mock 약관 텍스트(chunk_content) 리스트를 반환합니다 (하위 호환용)."""
    return [chunk["chunk_content"] for chunk in MOCK_RAG_CHUNKS]


# 질병-약관 매핑 타입 별칭
DiseasePolicyPairs = list[tuple[DiseaseInfo, list[str]]]


def get_mock_policies_per_disease(
    diseases: list[DiseaseInfo],
) -> DiseasePolicyPairs:
    """질병별로 동일한 Mock 약관 세트를 매핑하여 반환합니다.

    실제 RAG 대신 사용. 모든 질병에 동일한 약관 3건을 매핑합니다.
    """
    mock_texts = get_mock_policies()
    return [(disease, mock_texts) for disease in diseases]


def get_mock_rag_chunks() -> list[RagChunkResult]:
    """RAG 담당자 양식 전체를 그대로 반환합니다. 연동 시 파싱/매핑 테스트용."""
    return list(MOCK_RAG_CHUNKS)


def get_mock_diseases(state: VetAgentState) -> list[DiseaseInfo]:
    """Vet Agent 대신 사용할 Mock 질병 목록. disease_surgery_history도 포함."""
    mock_diseases = [
        DiseaseInfo(name="슬개골 탈구", incidence_rate="높음", onset_period="전 연령"),
        DiseaseInfo(
            name="치과 질환(치주염 등)", incidence_rate="중간", onset_period="5세 이상"
        ),
    ]
    hc = state.health_condition
    if hc and hc.disease_surgery_history:
        mock_diseases.append(
            DiseaseInfo(
                name=hc.disease_surgery_history,
                incidence_rate="기왕증",
                onset_period="가입 전",
            )
        )
    return mock_diseases
