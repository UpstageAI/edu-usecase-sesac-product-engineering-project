# RiskManagingAgent Memory Persistence Design & Debugging Report

이 문서는 `RiskManagingAgent`의 상태 유지를 위해 `MemorySaver` 및 `SqliteSaver`를 도입하고 디버깅한 과정을 기록합니다.

## 1. 개요 (Objective)
- `RiskManagingAgent`는 여러 턴에 걸친 대화를 통해 정보를 수집하고 리스크를 분석합니다.
- 서버 재시작이나 단일 요청 간에도 이전 대화에서 수집된 정보(`extracted_data`)와 진행 상태(`conversation_stage`)를 유지하기 위해 LangGraph의 체크포인트 기능을 도입했습니다.

## 2. 주요 해결 과제 및 디버깅 과정

### 2.1 상시(Stateless) 요청에서의 세션 유지 문제
- **현상**: `RiskManagingAgent.run` 호출 시마다 `uuid.uuid4()`로 새로운 세션 ID를 생성하여, LangGraph가 이전 체크포인트를 찾지 못하고 항상 초기 상태로 시작함.
- **해결**: 
    - 오케스트레이터(`orchestrator/nodes.py`)에서 `session_id`를 `context`에 담아 전달하도록 수정.
    - 에이전트 내에서 전달받은 `session_id`를 `thread_id`로 사용하여 체크포인트를 로드함.

### 2.2 비동기(Async) 환경에서의 SQLite 지원 문제
- **현상**: `SqliteSaver` 사용 시 `NotImplementedError: The SqliteSaver does not support async methods` 발생.
- **원인**: `ainvoke`를 사용하는 비동기 환경에서는 동기용 `SqliteSaver`를 사용할 수 없음.
- **해결**:
    - `AsyncSqliteSaver` (`langgraph.checkpoint.sqlite.aio`)로 교체.
    - `aiosqlite` 의존성 추가.
    - `AsyncSqliteSaver.from_conn_string("risk_checkpoints.db")`를 `async with` 블록 내에서 실행하여 비동기 커넥션을 안전하게 관리.

### 2.3 의존성 및 임포트 오류
- **현상**: `ModuleNotFoundError: No module named 'langgraph_checkpoint_sqlite'`.
- **해결**:
    - `uv add langgraph-checkpoint-sqlite`를 통해 `pyproject.toml` 및 `requirements.txt` 업데이트.
    - 임포트 경로를 `langgraph.checkpoint.sqlite` 표준 경로로 수정.

### 2.4 상태 병합 및 스키마 관리
- **현상**: 매번 새로운 입력이 들어올 때 기존에 추출된 데이터(`extracted_data`)가 초기화될 위험이 있음.
- **해결**:
    - `prepare_risk_state_node`에서 모든 필드를 초기화하지 않고 필요한 필드만 입력으로 받음.
    - LangGraph의 체크포인터가 `input_state`와 저장된 `checkpoint_state`를 자동으로 병합(Merge)하도록 설계.

### 2.5 JSON 파싱 안정성 확보 (Side Issue)
- **현상**: 리스크 평가 중 LLM 응답 내 제어 문자(Control Character)로 인해 `Invalid control character` JSON 파싱 에러 발생.
- **해결**:
    - `backend/utils/json_utils.py` 생성.
    - `json.loads(text, strict=False)` 옵션을 사용하는 `safe_json_parse` 유틸리티 구현 및 전방위 적용.

### 2.6 응답 중첩(Nesting) 및 문자열화 문제
- **현상**: `analysis_id`가 null인 중간 상태에서 응답이 `response='response="..." '`와 같이 여러 번 중첩되어 출력됨.
- **원인**: 
    - `format_final_output_node`에서 이전 턴의 `RiskManagingAgentResponse` 객체를 `str()`로 변환하여 새로운 응답의 `response` 필드에 담으면서 재귀적으로 중첩 발생.
    - 체크포인트에서 로드된 `dict` 형태의 응답을 `RiskManagingAgent.run`에서 다시 객체화/문자열화하며 발생.
- **해결**:
    - `nodes.py`: `agent_response_from_state`가 이미 구조화된 응답(객체 또는 dict)인 경우, `.response` 내용만 추출하여 사용하도록 로직 개선.
    - `graph.py`: 결과값이 이미 dict 형태인 경우 중복 래핑하지 않도록 처리.

## 3. 최종 아키텍처 (Final Design)

- **Persistence**: `risk_checkpoints.db` (SQLite 파일)
- **Saver**: `AsyncSqliteSaver` (비동기 지원)
- **Identifier**: 오케스트레이터 세션 ID -> LangGraph `thread_id` 매핑
- **Fallback**: DB 연결 실패 시 `MemorySaver`로 자동 전환되어 서비스 중단을 방지

## 4. 향후 고려 사항
- 대규모 동시 접속 시 SQLite의 `database is locked` 에러 발생 가능성이 있으므로, 트래픽 증가 시 `PostgresSaver`로의 전환 고려 필요.
- `risk_checkpoints.db` 파일의 주기적인 백업 또는 정제 로직 필요.
