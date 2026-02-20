#!/bin/bash
# 오케스트레이터 그래프 단위 가드레일 테스트
# - 정상 입력: 전체 파이프라인 정상 수행
# - 인젝션 입력: user_input_template → END 즉시 종료

SAMPLES_DIR="app/agents/user_input_template_agent/samples"
MODULE="app.agents.orchestrator.orchestrator_graph"

echo "============================================================"
echo "  오케스트레이터 가드레일 테스트"
echo "============================================================"

echo ""
echo "--- 1. 정상 입력 (전체 파이프라인 수행) ---"
echo "[user_input_all.yaml]"
uv run python -m $MODULE --input $SAMPLES_DIR/user_input_all.yaml

echo ""
echo "--- 2. 인젝션 입력 (즉시 종료되어야 함) ---"
echo "[user_input_injection_test.yaml]"
uv run python -m $MODULE --input $SAMPLES_DIR/user_input_injection_test.yaml
