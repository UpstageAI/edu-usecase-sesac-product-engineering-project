"""
기존 평가 결과 CSV에서 혼동 행렬 및 지표를 출력합니다.

중간에 실패한 평가에서 저장된 CSV로도 결과 확인 가능합니다.

실행:
    uv run python -m app.evaluation.report_from_csv [CSV경로]
    uv run python -m app.evaluation.report_from_csv  # 최신 CSV 자동 선택
"""

from pathlib import Path
import sys

import pandas as pd

from app.evaluation.nodes.metrics_node import (
    RESULTS_DIR,
    compute_and_display_metrics,
)
from app.evaluation.state import (
    EvaluationRecord,
    EvaluationTestCase,
    EvaluatorGroundTruth,
    JudgePrediction,
)


def _csv_row_to_record(row: pd.Series) -> EvaluationRecord:
    """CSV 행을 EvaluationRecord로 변환합니다."""
    tc = EvaluationTestCase(
        file_name=str(row.get("파일이름", "")),
        species="",
        breed=str(row.get("견묘종", "")),
        age=int(row.get("나이", 0)) if pd.notna(row.get("나이")) else 0,
        disease_surgery_history=str(row.get("기저질환", "없음")),
        disease_name=str(row.get("추출질병명", "")),
        policy_text=str(row.get("약관원문", "")),
    )
    jp = JudgePrediction(
        is_covered=str(row.get("Judge예측", "X")).strip().upper() == "O",
        reason=str(row.get("Judge이유", "")),
    )
    eg = EvaluatorGroundTruth(
        is_covered=str(row.get("Evaluator정답", "X")).strip().upper() == "O",
        reason=str(row.get("Evaluator이유", "")),
    )
    return EvaluationRecord(
        test_case=tc,
        judge_prediction=jp,
        evaluator_ground_truth=eg,
        label=str(row.get("라벨", "TN")),
    )


def main() -> None:
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1])
    else:
        # 최신 eval_result_*.csv 선택
        if not RESULTS_DIR.exists():
            print(f"결과 디렉토리가 없습니다: {RESULTS_DIR}")
            sys.exit(1)
        csv_files = sorted(RESULTS_DIR.glob("eval_result_*.csv"), reverse=True)
        if not csv_files:
            print(f"CSV 파일이 없습니다: {RESULTS_DIR}")
            sys.exit(1)
        csv_path = csv_files[0]
        print(f"최신 CSV 사용: {csv_path}")

    if not csv_path.exists():
        print(f"파일 없음: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    records = [_csv_row_to_record(row) for _, row in df.iterrows()]
    print(f"\n총 {len(records)}건 로드")
    compute_and_display_metrics(records, stats=None)


if __name__ == "__main__":
    main()
