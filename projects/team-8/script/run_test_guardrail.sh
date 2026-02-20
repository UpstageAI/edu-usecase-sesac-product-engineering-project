#!/bin/bash
# 입력 Sanitization 가드레일 테스트
# - 정상 입력: 변경 없이 통과하는지 확인 (회귀 테스트)
# - 인젝션 입력: 프롬프트 인젝션 패턴이 정화되는지 확인

SAMPLES_DIR="app/agents/user_input_template_agent/samples"
MODULE="app.agents.user_input_template_agent.nodes.user_input_template_node"

echo "============================================================"
echo "  입력 Sanitization 가드레일 테스트"
echo "============================================================"

echo ""
echo "--- 1. 정상 입력 (변경 없어야 함) ---"
echo "[user_input_simple.yaml]"
uv run python -m $MODULE --input $SAMPLES_DIR/user_input_simple.yaml
echo ""
echo "[user_input_all.yaml]"
uv run python -m $MODULE --input $SAMPLES_DIR/user_input_all.yaml

echo ""
echo "--- 2. 인젝션 입력 (정화되어야 함) ---"
echo "[user_input_injection_test.yaml]"
uv run python -m $MODULE --input $SAMPLES_DIR/user_input_injection_test.yaml
