"""
평가 파이프라인 흐름 정의.

YAML 로드 → 질병 추출(Vet/Mock) → 질병별 RAG 검색 → 테스트 케이스 생성
→ Judge + Evaluator 판단 → 라벨 부여. (LangGraph 미사용, 순차 호출)

핵심 변경: 질병별 개별 RAG 호출 (Method 1)
  - 질병 5개 × 질병별 top-k 약관 = 정확한 매핑
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from rich import print as rprint

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState
from app.evaluation.mocks.mock_data import (
    get_mock_diseases,
    get_mock_policies_per_disease,
)
from app.evaluation.nodes import (
    append_record_to_csv,
    build_test_cases,
    compute_label,
    evaluate_test_case,
    get_eval_csv_path,
    init_eval_csv,
    judge_predict,
    load_all_yaml_states,
)
from app.evaluation.state import EvaluationRecord, EvaluationTestCase

load_dotenv()
logger = logging.getLogger(__name__)

# ── 실행 모드 플래그 ──
USE_REAL_VET_AGENT = True  # True: 실제 Vet Agent / False: Mock 질병
USE_REAL_RAG = True  # True: 실제 RAG Agent / False: Mock 약관
YAML_LOAD_LIMIT = None  # 로드할 YAML 파일 개수 (None이면 전체)

# Rate limit 여유: API 호출 간 대기(초)
DELAY_BETWEEN_EVAL_CASES = 0.5  # Judge/Evaluator 케이스 간
DELAY_BETWEEN_RAG_DISEASES = 2.0  # 질병별 RAG 호출 간 (RAG 내부 부하 큼)

# 질병-약관 매핑 타입 별칭: [(질병, [해당 질병의 약관 텍스트들])]
DiseasePolicyPairs = list[tuple[DiseaseInfo, list[str]]]


@dataclass
class PipelineStats:
    """파이프라인 실행 후 요약 통계를 담는 클래스."""

    total_yaml_files: int = 0
    disease_counts: list[int] = field(default_factory=list)
    policy_counts_per_disease: list[int] = field(default_factory=list)
    total_test_cases: int = 0


# ── Vet Agent 연동 ──


async def run_vet_agent(state: VetAgentState) -> list[DiseaseInfo]:
    """실제 Vet Agent 그래프를 실행하여 질병 목록을 생성합니다."""
    from app.agents.vet_agent.graph import graph as vet_graph

    input_data = state.model_dump(
        include={
            "species",
            "breed",
            "age",
            "gender",
            "is_neutered",
            "weight",
            "health_condition",
            "coverage_style",
            "preferred_insurers",
        },
        exclude_none=True,
    )
    result = await vet_graph.ainvoke(input_data)
    return [DiseaseInfo.model_validate(d) for d in result.get("diseases", [])]


# ── RAG Agent 연동 (질병별 개별 호출) ──


async def run_rag_for_single_disease(
    state: VetAgentState,
    disease: DiseaseInfo,
) -> list[str]:
    """단일 질병에 대해 RAG Agent를 호출하여 해당 질병 관련 약관 텍스트를 반환합니다.

    diseases=[disease] 한 건만 넣어서 RAG를 호출하므로,
    반환되는 retrieved_documents는 해당 질병에 대한 약관만 포함합니다.
    """
    from app.agents.rag_agent.rag_graph import graph as rag_graph

    state_for_single = state.model_copy(update={"diseases": [disease]})
    input_data = state_for_single.model_dump(exclude_none=True)
    result = await rag_graph.ainvoke(input_data)

    documents: list[Document] = result.get("retrieved_documents", [])
    return [doc.page_content for doc in documents if doc.page_content]


async def run_rag_per_disease(
    state: VetAgentState,
    diseases: list[DiseaseInfo],
) -> DiseasePolicyPairs:
    """모든 질병에 대해 개별적으로 RAG를 호출하고, (질병, 약관 리스트) 쌍을 반환합니다."""
    pairs: DiseasePolicyPairs = []

    for d_idx, disease in enumerate(diseases, 1):
        if d_idx > 1 and DELAY_BETWEEN_RAG_DISEASES > 0:
            await asyncio.sleep(DELAY_BETWEEN_RAG_DISEASES)
        rprint(
            f"    RAG [{d_idx}/{len(diseases)}] '{disease.name}' 검색 중...",
            end="",
        )
        policy_texts = await run_rag_for_single_disease(state, disease)
        rprint(f" → {len(policy_texts)}개 약관")
        pairs.append((disease, policy_texts))

    return pairs


# ── Judge + Evaluator 판단 ──


async def _evaluate_test_cases(
    test_cases: list[EvaluationTestCase],
    csv_path: Path,
) -> list[EvaluationRecord]:
    """테스트 케이스 목록에 대해 Judge + Evaluator 판단을 수행하고, 1건 완료 시마다 CSV에 즉시 append합니다."""
    records: list[EvaluationRecord] = []
    label_colors = {"TP": "green", "TN": "blue", "FP": "bold red", "FN": "yellow"}

    for tc_idx, test_case in enumerate(test_cases, 1):
        rprint(
            f"    [{tc_idx}/{len(test_cases)}] "
            f"질병='{test_case.disease_name}' 판단 중...",
            end="",
        )

        judge_result = await judge_predict(test_case)
        evaluator_result = await evaluate_test_case(test_case)
        label = compute_label(judge_result, evaluator_result)

        color = label_colors.get(label, "white")
        rprint(
            f" → Judge={'O' if judge_result.is_covered else 'X'} / "
            f"Evaluator={'O' if evaluator_result.is_covered else 'X'} / "
            f"[{color}]{label}[/{color}]"
        )

        record = EvaluationRecord(
            test_case=test_case,
            judge_prediction=judge_result,
            evaluator_ground_truth=evaluator_result,
            label=label,
        )
        records.append(record)
        append_record_to_csv(record, csv_path)  # 중간 실패 대비 1건마다 즉시 저장

        # Rate limit 여유: 다음 케이스 전 대기
        if DELAY_BETWEEN_EVAL_CASES > 0:
            await asyncio.sleep(DELAY_BETWEEN_EVAL_CASES)

    return records


# ── 메인 파이프라인 ──


async def run_evaluation_pipeline() -> (
    tuple[list[EvaluationRecord], PipelineStats, Path]
):
    """LLM-as-a-Judge 평가 파이프라인 전체를 실행합니다.

    Returns:
        (EvaluationRecord 리스트, PipelineStats, CSV 경로) 튜플
        CSV는 평가 1건 완료 시마다 즉시 append되어 중간 실패 시에도 보존됨.
    """
    rprint("\n[bold cyan]═══ LLM-as-a-Judge 평가 파이프라인 시작 ═══[/bold cyan]\n")
    rag_mode = "RAG Agent (질병별 개별 호출)" if USE_REAL_RAG else "Mock"
    vet_mode = "Vet Agent" if USE_REAL_VET_AGENT else "Mock"
    rprint(f"  모드: Vet={vet_mode}, RAG={rag_mode}\n")

    rprint(f"[1/5] YAML 데이터 로드 중... (limit={YAML_LOAD_LIMIT})")
    yaml_states = load_all_yaml_states(limit=YAML_LOAD_LIMIT)
    rprint(f"  → {len(yaml_states)}개 상태 로드 완료\n")

    # 증분 저장용 CSV 초기화 (중간 실패 시 이미 평가된 건까지 보존)
    csv_path = get_eval_csv_path()
    init_eval_csv(csv_path)
    rprint(f"  [증분 저장] CSV: {csv_path}\n")

    stats = PipelineStats(total_yaml_files=len(yaml_states))
    all_records: list[EvaluationRecord] = []

    for idx, (file_name, state) in enumerate(yaml_states, 1):
        rprint(
            f"[bold]── [{idx}/{len(yaml_states)}] {file_name} "
            f"({state.breed}, {state.age}세) ──[/bold]"
        )

        # ── Step 2: 질병 목록 생성 ──
        rprint("  [2/5] 질병 목록 생성 중...")
        if USE_REAL_VET_AGENT:
            diseases = await run_vet_agent(state)
        else:
            diseases = get_mock_diseases(state)
        stats.disease_counts.append(len(diseases))
        rprint(f"  → 추출된 질병 {len(diseases)}개: {[d.name for d in diseases]}\n")

        # ── Step 3: 질병별 RAG 검색 → (질병, 약관) 매핑 ──
        if USE_REAL_RAG:
            rprint("  [3/5] 질병별 RAG Agent 검색 중...")
            disease_policy_pairs = await run_rag_per_disease(state, diseases)
        else:
            rprint("  [3/5] 질병별 약관 검색 중... (Mock)")
            disease_policy_pairs = get_mock_policies_per_disease(diseases)

        total_policies = sum(len(policies) for _, policies in disease_policy_pairs)
        for disease, policies in disease_policy_pairs:
            stats.policy_counts_per_disease.append(len(policies))
        rprint(
            f"  → 질병 {len(diseases)}개 × 질병별 약관 = 총 약관 {total_policies}개 로드\n"
        )

        # ── Step 4: 테스트 케이스 생성 (질병-약관 1:1 매핑) ──
        test_cases = build_test_cases(file_name, state, disease_policy_pairs)
        stats.total_test_cases += len(test_cases)
        rprint(f"  [4/5] 테스트 케이스 생성: {len(test_cases)}개\n")

        # ── Step 5: Judge + Evaluator 판단 ──
        rprint("  [5/5] Judge + Evaluator 판단 실행 중...")
        file_records = await _evaluate_test_cases(test_cases, csv_path)
        all_records.extend(file_records)

        rprint()

    return all_records, stats, csv_path
