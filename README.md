# Trade Onboarding Agent (Code-Based README)

이 문서는 현재 코드 구조를 기준으로 작성되었습니다.

## 목차

- [1) 프로젝트 개요](#project-overview)
- [2) 저장소 구조](#repository-structure)
- [3) 런타임 요구사항](#runtime-requirements)
- [4) 환경 변수](#environment-variables)
- [5) 실행 방법](#run-guide)
- [6) API 요약](#api-summary)
- [7) 오케스트레이션 동작](#orchestration-flow)
- [8) RAG/인덱싱](#rag-indexing)
- [9) 세션 저장소](#session-store)
- [10) 테스트](#testing)
- [11) 로깅](#logging)

## <a id="project-overview"></a>1) 프로젝트 개요

Trade Onboarding Agent는 무역 실무 지원을 위한 멀티 에이전트 챗봇입니다.

- 백엔드: FastAPI + LangGraph 오케스트레이션
- 프론트엔드: React + Vite
- 지식 검색: ChromaDB 기반 벡터 검색(RAG)
- 주요 기능:
  - 리스크 분석(`riskmanaging`)
  - 이메일 초안/검토(`email`)
  - 퀴즈 생성/풀이(`quiz`)
  - 일반 대화(`default_chat`)

## <a id="repository-structure"></a>2) 저장소 구조

```text
trade-onboarding-agent/
├─ backend/
│  ├─ main.py                         # FastAPI 앱 진입점
│  ├─ api/routes.py                   # /api/chat
│  ├─ agents/
│  │  ├─ orchestrator/                # 세션 로딩/라우팅/에이전트 호출
│  │  ├─ email_agent/                 # 이메일 초안/리뷰
│  │  ├─ quiz_agent/                  # 퀴즈 생성/답안 처리
│  │  ├─ riskmanaging/                # 리스크 분석
│  │  └─ default_chat/                # 일반 대화
│  ├─ rag/                            # 임베딩/검색/인덱싱
│  ├─ schemas/                        # API 스키마
├─ frontend/
│  ├─ src/App.jsx                     # 채팅/퀴즈 UI
│  └─ src/lib/normalizers.js          # API 응답 정규화
├─ dataset/                           # RAG 대상 JSON 데이터
├─ tests/                             # pytest 테스트
└─ docs/README.md                     # 이 문서
```

## <a id="runtime-requirements"></a>3) 런타임 요구사항

- Python `>=3.11`
- Node.js (Vite 실행 가능 버전)
- 선택 사항:
  - Upstage API Key (`UPSTAGE_API_KEY`)
  - Redis (`USE_REDIS_SESSION=true`일 때)

## <a id="environment-variables"></a>4) 환경 변수

`.env.example`를 기준으로 `.env`를 구성합니다.

핵심 값:

- `UPSTAGE_API_KEY`: LLM/임베딩 API 키
- `EMBEDDING_PROVIDER`: `local | upstage | auto`
- `LOCAL_EMBEDDING_DIM`: 로컬 해시 임베딩 차원 (기본 4096)
- `USE_REDIS_SESSION`: 세션 저장소 Redis 사용 여부
- `AUTO_INGEST_ON_STARTUP`: 서버 시작 시 자동 인덱싱
- `REINGEST_ON_DATASET_CHANGE`: 데이터셋 변경 시 재인덱싱
- `FORCE_REINGEST_ON_STARTUP`: 강제 재인덱싱
- `CORS_ORIGINS`: 허용 Origin 목록

추가 기본값/동작은 `backend/config.py`를 기준으로 합니다.

## <a id="run-guide"></a>5) 실행 방법

### 5.1 백엔드

프로젝트 루트에서:

```bash
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

헬스 체크:

```bash
curl http://localhost:8000/health
```

### 5.2 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

기본 API 주소:

- `VITE_API_BASE_URL` 미설정 시 `http://localhost:8000/api`

## <a id="api-summary"></a>6) API 요약

### `POST /api/chat`

요청:

```json
{
  "session_id": "string",
  "message": "string",
  "context": {
    "mode": "riskmanaging|quiz|email|default_chat"
  }
}
```

`auto` 라우팅은 `context.mode`를 보내지 않는 방식으로 동작합니다.

응답(`ChatResponse`):

```json
{
  "type": "chat|report|error",
  "message": "string",
  "report": {},
  "meta": {}
}
```

## <a id="orchestration-flow"></a>7) 오케스트레이션 동작

`backend/agents/orchestrator/nodes.py` 기준:

1. 세션 상태 로드 (`conversation_history`, `active_agent`)
2. 라우팅 결정
   - `context.mode` 강제 모드 우선
   - 키워드 기반 빠른 라우팅 (risk/quiz/email)
   - 활성 에이전트 연속 처리(후속 질문, 퀴즈 답안 등)
   - 필요 시 LLM 분류(`solar-pro2`)
3. 선택 에이전트 실행
4. 세션 상태 저장
5. 응답 스키마 정규화(`backend/core/response_converter.py`)

## <a id="rag-indexing"></a>8) RAG/인덱싱

서버 시작 시 `backend/main.py`에서 컬렉션 상태를 확인하고, 설정에 따라 자동 인덱싱을 수행합니다.

수동 인덱싱:

```bash
uv run python backend/rag/ingest.py
```

강제 재인덱싱:

```bash
uv run python backend/rag/ingest.py --reset
```

## <a id="session-store"></a>9) 세션 저장소

- 기본: InMemory (`USE_REDIS_SESSION=false`)
- 프로덕션 권장: Redis (`USE_REDIS_SESSION=true`)
- 구현: `backend/agents/orchestrator/session_store.py`

## <a id="testing"></a>10) 테스트

프로젝트 루트에서:

```bash
uv run pytest
```

예시:

```bash
uv run pytest tests/test_orchestrator.py -q
uv run pytest tests/test_chatbot_flow_regression.py -q
```

## <a id="logging"></a>11) 로깅

`backend/utils/logger.py` 기준으로 `logs/`에 파일 로그를 생성합니다.

- `trade_onboarding.log`
- `trade_onboarding_error.log`
- `trade_onboarding_debug.log` (development 환경)
