# 리스크 관리 에이전트 업데이트 내역 (v2)

본 문서는 리스크 관리 에이전트(Risk Managing Agent)에 추가된 '감독 의사결정 계층' 및 '응답 강제 레이어'에 대한 작업 내용을 기록합니다.

## 1. 감독 의사결정 계층 (Supervisor Decision Layer) 추가
`backend/prompts/riskmanaging_prompt.txt` 파일에 관리자급 의사결정 권한과 행동 규칙을 추가했습니다.

### 주요 변경 사항:
- **운영 권한 부여**: 에이전트가 단순 조언자를 넘어 금융 손실 방지를 위한 '무역 운영 감독관'으로서의 역할을 수행하도록 설정.
- **행동 우선순위**: 리스크가 감지될 경우 설명을 제공하기 전에 즉시 작업을 중단(Stop)시키도록 강제.
- **의사결정 정책**:
    - **BLOCKED**: 손실 가능성 존재 시
    - **WARNING**: 불확실성 존재 시
    - **APPROVED**: 안전 확인 시
- **커뮤니케이션 원칙**: 인사말 생략, 결정 전 제안 금지, 사건 예방 우선 원칙 적용.

## 2. 응답 강제 레이어 (Response Enforcement Layer) 구현
`backend/agents/riskmanaging/nodes.py`에 LLM의 출력을 정해진 구조로 변환하는 후처리 로직을 통합했습니다.

### 주요 구현 내용:
- **`format_decision_report` 함수**:
    - 리스크 점수(`risk_score`)에 따른 최종 결정(DECISION) 자동 분류.
        - 8점 이상: `BLOCKED`
        - 4~7점: `WARNING`
        - 4점 미만: `APPROVED`
    - LLM의 원본 응답에서 요약, 리스크 설명, 예상 손실, 조치 사항, 증거 자료를 추출하여 구조화.
- **노드 통합**: `format_final_output_node`에서 최종 응답을 반환하기 직전에 위 함수를 호출하여 보고서 형식을 강제 적용.

### 출력 구조:
```text
[DECISION]
APPROVED | WARNING | BLOCKED

[SUMMARY]
상황 요약 (Short explanation)

[RISK]
감지된 리스크 설명

[LOSS]
발생 가능한 금전적/운영적 손실

[ACTION]
필수 이행 단계 (Mandatory steps)

[EVIDENCE]
리스크 판단 근거 (Rule or pattern)
```

## 3. 적용 파일
- `backend/prompts/riskmanaging_prompt.txt`
- `backend/agents/riskmanaging/nodes.py`
