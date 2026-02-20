# pet-insurance-recommender

LangGraph 기반 멀티에이전트 반려동물 보험 추천 시스템입니다.  
오케스트레이터가 사용자 입력 정제 → 질병 추정(vet) → 약관 검색(RAG) → 상품 검증(judge) → 최종 문장 생성(composer) 순서로 파이프라인을 실행합니다.

## 주요 구성

- `app/main.py`: FastAPI 앱 (`/health`, `/test`)
- `app/agents/orchestrator/`: 전체 파이프라인 그래프
- `app/agents/user_input_template_agent/`: 입력 YAML 로딩/정제(가드레일)
- `app/agents/vet_agent/`: 반려동물 정보 기반 질병 후보 생성
- `app/agents/rag_agent/`: Qdrant 기반 약관 검색/요약
- `app/agents/judge_agent/`: 질병-약관 매칭 검증
- `app/agents/composer_agent/`: 최종 사용자 응답 생성
- `app/agents/document_parser/`: PDF 약관 파싱/청킹/태깅/벡터 적재
- `app/evaluation/`: LLM-as-a-Judge 평가 파이프라인

## 요구 사항

- Python `>=3.12`
- `uv`
- Docker (Qdrant 사용 시)
- Upstage API 키 (`UPSTAGE_API_KEY`)

## 설치

```bash
uv sync
cp .env.example .env
```

`.env`에서 최소 아래 값 확인:

- `UPSTAGE_API_KEY`
- `UPSTAGE_MODEL` (기본: `solar-pro2`)
- `QDRANT_URL` (기본: `http://localhost:6333`)
- `QDRANT_VECTOR_SIZE` (기본: `4096`)
- `TAVILY_API_KEY` (vet 검색 검증 사용 시)

## 실행

### 1) Qdrant 실행

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### 2) Orchestrator CLI 실행

```bash
# 기본 샘플 입력 실행
bash script/run_orchestrator_graph.sh

# 입력 YAML/스레드 ID 지정 실행
bash script/run_orchestrator_graph_args.sh
```

직접 실행도 가능합니다.

```bash
uv run python -m app.agents.orchestrator.orchestrator_graph \
  --input app/agents/user_input_template_agent/samples/user_input_simple.yaml \
  --thread-id test_user
```

### 3) Streamlit 데모

```bash
bash script/run_streamlit.sh
```

### 4) FastAPI 서버

```bash
uv run uvicorn app.main:app --reload
```

- Health check: `GET /health`
- 테스트 실행: `GET /test`

## 테스트 스크립트

```bash
# 라우터 시나리오 (연속 실행)
bash script/run_test_router.sh

# 단일 시나리오
bash script/run_test_single.sh

# 입력 정제 가드레일 테스트 (노드 단위)
bash script/run_test_guardrail.sh

# 오케스트레이터 가드레일 테스트 (그래프 단위)
bash script/run_test_orchestrator_guardrail.sh
```

## 약관 파서(document_parser)

약관 PDF를 페이지 단위로 분리하고 청킹/태깅 후 Qdrant에 적재할 수 있습니다.

- 입력 PDF 위치: `app/agents/document_parser/data/terms/`
- 태그 타입: `normal` 또는 `simple`
- 컬렉션명:
  - `normal` → `terms_normal_tag_dense`
  - `simple` → `terms_simple_tag_dense`

예시:

```bash
# 파싱 + 페이지 분리 + 청킹 + 태깅
uv run python -m app.agents.document_parser.dp_graph \
  --file-name meritz_1_maum_pet_1_21_22_50.pdf \
  --basic-term-start 1 --basic-term-end 21 \
  --special-term-start 22 --special-term-end 50 \
  --tag-type simple

# 위 과정 + Qdrant 적재
uv run python -m app.agents.document_parser.dp_graph \
  --file-name meritz_1_maum_pet_1_21_22_50.pdf \
  --basic-term-start 1 --basic-term-end 21 \
  --special-term-start 22 --special-term-end 50 \
  --tag-type simple --ingest
```

## 평가 파이프라인

LLM-as-a-Judge 평가 실행:

```bash
uv run python -m app.evaluation.runner
```

기존 CSV로 리포트 재출력:

```bash
# 최신 CSV 자동 선택
uv run python -m app.evaluation.report_from_csv

# 특정 CSV 지정
uv run python -m app.evaluation.report_from_csv app/evaluation/results/eval_result_YYYYMMDDHHMMSS.csv
```

상세 설명은 `app/evaluation/README.md` 참고.
