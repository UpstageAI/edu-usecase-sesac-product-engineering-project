"""
평가 파이프라인 엔트리포인트.

실행 방법:
    uv run python -m app.evaluation.runner
"""

import asyncio
import logging

from rich import print as rprint

from app.evaluation.graph import run_evaluation_pipeline
from app.evaluation.nodes import compute_and_display_metrics, save_results_to_csv


async def main() -> None:
    """평가 파이프라인을 실행하고 결과를 출력/저장합니다."""
    records, stats, csv_path = await run_evaluation_pipeline()

    if not records:
        rprint("[bold red]평가 결과가 없습니다. YAML 데이터를 확인하세요.[/bold red]")
        return

    compute_and_display_metrics(records, stats=stats)
    save_results_to_csv(records, incremental_path=csv_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
