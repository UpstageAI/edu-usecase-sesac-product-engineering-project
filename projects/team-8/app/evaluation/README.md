# LLM-as-a-Judge 평가 파이프라인

## 이 모듈은 무엇인가요?

기존 **JudgeAgent**(보험 약관 검증 에이전트)가 "이 질병이 이 약관으로 보장되는가?"를 얼마나 정확하게 판단하는지 **정량적으로 평가**하는 파이프라인입니다.

핵심 아이디어는 간단합니다:

> **더 엄격한 LLM(Evaluator)**이 내린 판단을 **정답(Ground Truth)**으로 삼고,  
> 기존 JudgeAgent의 판단과 비교하여 **혼동 행렬(Confusion Matrix)**로 성능을 측정합니다.

---

## 전체 흐름 (한눈에 보기)

```
┌─────────────────────────────────────────────────────────────────┐
│                        평가 파이프라인                            │
│                                                                 │
│  ① YAML 로드        ② Vet Agent        ③ RAG 검색              │
│  (네이버 128개)  →   (질병 추출)    →   (약관 텍스트)             │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│            ④ 테스트 케이스 생성                                   │
│              [1질병 × 1약관] 조합                                 │
│                                                                 │
│                    ↓           ↓                                │
│                                                                 │
│         ⑤-A JudgeAgent     ⑤-B Evaluator LLM                  │
│          (기존 로직)         (엄격한 정답지)                      │
│          is_covered?         is_covered?                        │
│              ↓                    ↓                             │
│                                                                 │
│                    ⑥ 비교 → 라벨 부여                            │
│                     (TP / TN / FP / FN)                         │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│          ⑦ 혼동 행렬 출력 + CSV 저장                             │
└─────────────────────────────────────────────────────────────────┘
```

### 각 단계별 설명

| 단계 | 무엇을 하나요? | 어디서? |
|------|---------------|---------|
| ① YAML 로드 | 네이버 펫보험 상담 글에서 추출한 반려동물 정보 128개를 읽어옴 | `nodes/load_data_node.py` |
| ② Vet Agent | 반려동물 정보(품종, 나이, 건강상태)를 기반으로 잘 걸리는 질병 목록 생성 | `graph.py` → `vet_agent` |
| ③ RAG 검색 | 질병에 관련된 보험 약관 텍스트를 벡터DB에서 검색 (현재 Mock 3개) | `mocks/mock_data.py` |
| ④ 테스트 케이스 | 질병 N개 × 약관 M개 = N×M개의 [질병, 약관] 조합 생성 | `nodes/build_test_cases_node.py` |
| ⑤-A Judge 판단 | 기존 JudgeAgent 프롬프트로 "이 질병이 보장되는가?" 판단 | `nodes/judge_node.py` |
| ⑤-B Evaluator 판단 | 훨씬 엄격한 프롬프트로 동일한 질문의 "정답" 생성 | `nodes/evaluator_node.py` |
| ⑥ 라벨 부여 | Judge와 Evaluator 결과를 비교하여 TP/TN/FP/FN 판정 | `nodes/judge_node.py` |
| ⑦ 결과 출력 | 혼동 행렬 + 성능 지표 터미널 출력, CSV 파일 저장 | `nodes/metrics_node.py` |

---

## 파일 구조

```
app/evaluation/
├── __init__.py
├── README.md
├── state/                        # Pydantic 데이터 모델 (상태/스키마)
│   ├── __init__.py
│   └── evaluation_state.py       # EvaluationTestCase, JudgePrediction, ...
├── nodes/
│   ├── __init__.py
│   ├── load_data_node.py         # YAML 테스트셋 로더
│   ├── build_test_cases_node.py  # [질병×약관] 조합 생성
│   ├── judge_node.py             # JudgeAgent 단건 판단 + 라벨 계산
│   ├── evaluator_node.py         # Evaluator LLM (정답지 생성)
│   └── metrics_node.py           # 혼동 행렬 + CSV 저장
├── mocks/
│   ├── __init__.py
│   └── mock_data.py              # Mock 약관/질병 (get_mock_policies, get_mock_diseases)
├── graph.py                      # 파이프라인 흐름 (run_evaluation_pipeline)
├── runner.py                     # 엔트리포인트 (asyncio.run(main))
└── results/
    └── eval_result_{timestamp}.csv
```

---

## 핵심 데이터 모델 (`state/evaluation_state.py`)

평가 파이프라인을 관통하는 4개의 Pydantic 모델입니다.

```
EvaluationTestCase          → "문제지" (질병 1개 + 약관 1개)
        │
        ├── JudgePrediction      → "학생 답안" (JudgeAgent의 판단)
        │     └── is_covered: bool, reason: str
        │
        ├── EvaluatorGroundTruth → "모범 답안" (Evaluator LLM의 판단)
        │     └── is_covered: bool, reason: str
        │
        └── EvaluationRecord     → "채점 결과" (문제 + 답안 + 정답 + 라벨)
              └── label: "TP" | "TN" | "FP" | "FN"
```

### EvaluationTestCase (문제지)

| 필드 | 설명 | 예시 |
|------|------|------|
| `file_name` | 원본 YAML 파일명 | `"34840"` |
| `species` | 종 | `"강아지"` |
| `breed` | 품종 | `"토이푸들"` |
| `age` | 나이 | `3` |
| `disease_surgery_history` | 기저질환/수술 이력 | `"슬개골 1기 진단"` |
| `disease_name` | 평가할 질병명 | `"슬개골 탈구"` |
| `policy_text` | 평가할 약관 원문 | `"제5조 보상하지 않는 손해..."` |

### JudgePrediction / EvaluatorGroundTruth (답안)

| 필드 | 설명 | 예시 |
|------|------|------|
| `is_covered` | 보장 여부 | `True` (보장) / `False` (면책) |
| `reason` | 판단 근거 | `"약관 제5조 기왕증 면책 조항에 해당..."` |

---

## 혼동 행렬 (Confusion Matrix) 이해하기

"Positive"는 **보장됨**, "Negative"는 **보장 안 됨(면책)**을 의미합니다.

```
                    ┌─── Evaluator (정답) ───┐
                    │  보장(P)     면책(N)    │
        ┌───────────┼────────────────────────┤
 Judge  │ 보장(P)   │  TP(정확)   FP(위험⚠️)  │
 (예측) │ 면책(N)   │  FN(보수적)  TN(정확)   │
        └───────────┴────────────────────────┘
```

| 라벨 | 의미 | 위험도 |
|------|------|--------|
| **TP** (True Positive) | Judge도 보장, Evaluator도 보장 → 정확한 보장 판정 | 안전 |
| **TN** (True Negative) | Judge도 면책, Evaluator도 면책 → 정확한 면책 판정 | 안전 |
| **FP** (False Positive) | Judge는 보장이라 했지만, 실제로는 면책 → **가장 위험!** | **높음** |
| **FN** (False Negative) | Judge는 면책이라 했지만, 실제로는 보장됨 → 보수적 판단 | 낮음 |

> **FP가 가장 위험한 이유**: 실제로 보장이 안 되는 질병인데 "보장됩니다"라고 사용자에게 안내하면 금전적 피해로 이어질 수 있습니다.

---

## Judge vs Evaluator 프롬프트 차이

| 구분 | JudgeAgent (학생) | Evaluator (선생님) |
|------|-------------------|-------------------|
| 역할 | 보험 약관 심사 전문가 | 금융감독원 출신 최고 전문가 (20년 경력) |
| 엄격도 | 일반적 수준 | 매우 엄격 (글자 그대로 해석) |
| 불확실할 때 | 판단에 따름 | 보수적으로 False 판정 |
| 판단 기준 | 면책→기왕증→나이→보장범위 | 면책→기왕증→보장범위→보수적 원칙 |
| 모델 | solar-pro2 | solar-pro2 |

두 LLM이 같은 모델을 쓰지만, **프롬프트의 엄격도 차이**로 판단 결과가 달라질 수 있습니다.  
이 차이를 통해 JudgeAgent 프롬프트의 개선 포인트를 발견하는 것이 목적입니다.

---

## 실행 방법

### 기본 실행 (YAML 3개, Vet Agent 실제 호출, RAG Mock)

```bash
uv run python -m app.evaluation.runner
```

### data_loader만 단독 테스트

```bash
uv run python -m app.evaluation.nodes.load_data_node
```

### 설정 변경

`graph.py` 상단의 상수로 동작을 제어할 수 있습니다:

```python
# True: 실제 Vet Agent 호출 / False: Mock 질병 사용 (API 비용 절감)
USE_REAL_VET_AGENT = True

# 로드할 YAML 파일 개수 (None이면 128개 전체 실행)
YAML_LOAD_LIMIT = 3
```

---

## 출력 예시

### 터미널 출력

```
═══ LLM-as-a-Judge 평가 파이프라인 시작 ═══

[1/5] YAML 데이터 로드 중... (limit=3)
  → 3개 상태 로드 완료

── [1/3] 34840 (토이푸들, 3세) ──
  [2/5] 질병 목록 생성 중...
  → 추출된 질병 3개: ['슬개골 탈구', '치과 질환', '피부병']
  ...
  [5/5] Judge + Evaluator 판단 실행 중...
    [1/9] 질병='슬개골 탈구' → Judge=O / Evaluator=X / FP   ← 위험!
    [2/9] 질병='슬개골 탈구' → Judge=X / Evaluator=X / TN
    ...

┌──────────────── 혼동 행렬 (Confusion Matrix) ────────────────┐
│                    │ Evaluator: 보장(P) │ Evaluator: 면책(N) │
│ Judge: 보장(P)     │    TP = 5          │    FP = 2          │
│ Judge: 면책(N)     │    FN = 1          │    TN = 10         │
└──────────────────────────────────────────────────────────────┘
```

### CSV 파일

`app/evaluation/results/eval_result_20260218193000.csv` 형태로 자동 저장됩니다.

| 파일이름 | 견묘종 | 나이 | 기저질환 | 추출질병명 | 약관원문 | Judge예측 | Judge이유 | Evaluator정답 | Evaluator이유 | 라벨 |
|---------|--------|------|---------|-----------|---------|----------|----------|-------------|-------------|------|
| 34840 | 토이푸들 | 3 | 슬개골 1기 | 슬개골 탈구 | [상품명: 메리츠...] | O | 보장 항목에... | X | 제5조 기왕증... | FP |

---

## 향후 확장 포인트

| 항목 | 현재 | 향후 |
|------|------|------|
| 약관 텍스트 | Mock 3개 고정 | 실제 RAG Agent 연동 |
| Vet Agent | 실제/Mock 선택 가능 | 항상 실제 사용 |
| YAML 수량 | 3개 슬라이싱 | 128개 전체 |
| 평가 LLM | solar-pro2 단일 | GPT-4o 등 외부 모델 교차 검증 |
| 비동기 처리 | 순차 실행 | `asyncio.gather` + Semaphore로 병렬화 |
| 결과 저장 | CSV만 | LangSmith Dataset 연동 |
