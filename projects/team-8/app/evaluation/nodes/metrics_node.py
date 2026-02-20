"""
혼동 행렬(Confusion Matrix) 계산 및 결과 리포트 저장 노드.

128건 등 장시간 실행 시 토큰 제한/크래시 대비:
  - init_eval_csv()로 파이프라인 시작 시 CSV 헤더 생성
  - append_record_to_csv()로 평가 1건 완료 시마다 한 줄씩 즉시 저장
  - 중간 실패해도 이미 평가된 건까지 복구 가능
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from rich import print as rprint
from rich.table import Table

from app.evaluation.state import EvaluationRecord

if TYPE_CHECKING:
    from app.evaluation.graph import PipelineStats

# 결과 CSV 저장 디렉토리 (nodes/ 기준 상위 → evaluation/results)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# CSV 컬럼 순서 (records_to_dataframe과 동일)
CSV_COLUMNS = [
    "파일이름",
    "견묘종",
    "나이",
    "기저질환",
    "추출질병명",
    "약관원문",
    "Judge예측",
    "Judge이유",
    "Evaluator정답",
    "Evaluator이유",
    "라벨",
]


def print_dataset_summary(stats: PipelineStats, total_records: int) -> None:
    """파이프라인 실행 결과의 데이터셋 구축 요약을 출력합니다.

    "128개의 반려동물 정보 - top-k개 질병 추출 - top-k개 약관 추출로 총 N개 데이터셋을 구축하였습니다."
    형태의 로깅을 출력합니다.
    """
    avg_diseases = (
        sum(stats.disease_counts) / len(stats.disease_counts)
        if stats.disease_counts
        else 0
    )
    avg_policies_per_disease = (
        sum(stats.policy_counts_per_disease) / len(stats.policy_counts_per_disease)
        if stats.policy_counts_per_disease
        else 0
    )

    rprint()
    rprint("[bold cyan]═══ 데이터셋 구축 요약 ═══[/bold cyan]")
    rprint(
        f"  {stats.total_yaml_files}개의 반려동물 정보 "
        f"- 평균 {avg_diseases:.1f}개 질병 추출 "
        f"- 질병당 평균 {avg_policies_per_disease:.1f}개 약관 검색으로 "
        f"총 [bold]{total_records}개[/bold] 데이터셋을 구축하였습니다."
    )


def compute_and_display_metrics(
    records: list[EvaluationRecord],
    stats: PipelineStats | None = None,
) -> dict[str, int]:
    """평가 결과에서 혼동 행렬을 계산하고 Rich 테이블로 출력합니다."""
    counts: dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1

    total = len(records)

    # ── 데이터셋 구축 요약 출력 ──
    if stats is not None:
        print_dataset_summary(stats, total)

    # ── 혼동 행렬 테이블 ──
    matrix_table = Table(
        title="🔍 혼동 행렬 (Confusion Matrix)",
        show_header=True,
        header_style="bold magenta",
    )
    matrix_table.add_column("", style="bold", width=25)
    matrix_table.add_column("Evaluator: 보장(P)", justify="center", width=20)
    matrix_table.add_column("Evaluator: 면책(N)", justify="center", width=20)

    matrix_table.add_row(
        "Judge: 보장(P)",
        f"[green]TP = {counts['TP']}[/green]",
        f"[bold red]FP = {counts['FP']}[/bold red]",
    )
    matrix_table.add_row(
        "Judge: 면책(N)",
        f"[yellow]FN = {counts['FN']}[/yellow]",
        f"[blue]TN = {counts['TN']}[/blue]",
    )

    rprint()
    rprint(matrix_table)

    # ── 성능 지표 계산 ──
    accuracy = (counts["TP"] + counts["TN"]) / total if total > 0 else 0
    precision = (
        counts["TP"] / (counts["TP"] + counts["FP"])
        if (counts["TP"] + counts["FP"]) > 0
        else 0
    )
    recall = (
        counts["TP"] / (counts["TP"] + counts["FN"])
        if (counts["TP"] + counts["FN"]) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    metrics_table = Table(
        title="📊 성능 지표",
        show_header=True,
        header_style="bold cyan",
    )
    metrics_table.add_column("지표", style="bold", width=15)
    metrics_table.add_column("값", justify="center", width=15)
    metrics_table.add_row("총 테스트 수", str(total))
    metrics_table.add_row("Accuracy", f"{accuracy:.2%}")
    metrics_table.add_row("Precision", f"{precision:.2%}")
    metrics_table.add_row("Recall", f"{recall:.2%}")
    metrics_table.add_row("F1 Score", f"{f1:.2%}")

    rprint()
    rprint(metrics_table)

    # ── 요약 로그 (한 줄 요약) ──
    yaml_count = stats.total_yaml_files if stats else "?"
    rprint(
        f"\n[bold]📋 이 중 Precision {precision:.2%} / "
        f"Recall {recall:.2%} / F1 Score {f1:.2%} 입니다.[/bold]"
    )

    # ── FP 위험 케이스 상세 출력 ──
    fp_records = [r for r in records if r.label == "FP"]
    if fp_records:
        rprint()
        rprint(
            f"[bold red]⚠️  FP(위험 케이스) {len(fp_records)}건 상세 "
            f"— 보장 안 되는데 보장된다고 판단한 건[/bold red]"
        )
        fp_table = Table(
            title="⚠️ FP (False Positive) 상세",
            show_header=True,
            header_style="bold red",
        )
        fp_table.add_column("파일명", width=12)
        fp_table.add_column("품종", width=12)
        fp_table.add_column("질병명", width=20)
        fp_table.add_column("Judge 이유", width=40)
        fp_table.add_column("Evaluator 이유", width=40)
        for r in fp_records:
            fp_table.add_row(
                r.test_case.file_name,
                r.test_case.breed,
                r.test_case.disease_name,
                r.judge_prediction.reason[:80],
                r.evaluator_ground_truth.reason[:80],
            )
        rprint(fp_table)

    return counts


def _record_to_row(record: EvaluationRecord) -> dict[str, str | int]:
    """단일 EvaluationRecord를 CSV 행용 dict로 변환합니다."""
    tc = record.test_case
    jp = record.judge_prediction
    eg = record.evaluator_ground_truth
    return {
        "파일이름": tc.file_name,
        "견묘종": tc.breed,
        "나이": tc.age,
        "기저질환": tc.disease_surgery_history,
        "추출질병명": tc.disease_name,
        "약관원문": tc.policy_text[:200],
        "Judge예측": "O" if jp.is_covered else "X",
        "Judge이유": jp.reason,
        "Evaluator정답": "O" if eg.is_covered else "X",
        "Evaluator이유": eg.reason,
        "라벨": record.label,
    }


def get_eval_csv_path() -> Path:
    """이번 실행용 타임스탬프 포함 CSV 경로를 반환합니다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return RESULTS_DIR / f"eval_result_{timestamp}.csv"


def init_eval_csv(csv_path: Path) -> None:
    """CSV 파일을 생성하고 헤더 행만 씁니다. 파이프라인 시작 시 1회 호출."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_record_to_csv(record: EvaluationRecord, csv_path: Path) -> None:
    """평가 1건 완료 시마다 CSV에 한 줄을 append합니다. 중간 실패 시 이미 저장된 건까지 보존."""
    row = _record_to_row(record)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def records_to_dataframe(records: list[EvaluationRecord]) -> pd.DataFrame:
    """평가 레코드 리스트를 Pandas DataFrame으로 변환합니다."""
    rows = [_record_to_row(r) for r in records]
    return pd.DataFrame(rows)


def save_results_to_csv(
    records: list[EvaluationRecord],
    incremental_path: Path | None = None,
) -> Path:
    """평가 결과를 CSV로 저장합니다.

    incremental_path가 제공되면 이미 한 건씩 append된 파일이므로,
    덮어쓰지 않고 경로와 건수만 출력합니다.
    """
    if incremental_path is not None and incremental_path.exists():
        rprint(
            f"\n[bold green]📁 결과 저장 완료(증분 저장): {incremental_path}[/bold green]"
        )
        rprint(f"   총 {len(records)}건, 컬럼: {CSV_COLUMNS}")
        return incremental_path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = get_eval_csv_path()
    df = records_to_dataframe(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    rprint(f"\n[bold green]📁 결과 저장 완료: {csv_path}[/bold green]")
    rprint(f"   총 {len(records)}건, 컬럼: {list(df.columns)}")
    return csv_path
