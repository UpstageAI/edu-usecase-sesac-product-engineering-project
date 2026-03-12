# Workflow Diagrams

이 문서는 현재 코드 기준의 워크플로우를 Mermaid 다이어그램으로 정리한 문서입니다.

## 목차

- [전체 시스템 워크플로우](#전체-시스템-워크플로우)
- [Orchestrator 워크플로우](#orchestrator-워크플로우)
- [QuizAgent 워크플로우](#quizagent-워크플로우)
- [EmailAgent 워크플로우](#emailagent-워크플로우)
- [RiskManagingAgent 워크플로우](#riskmanagingagent-워크플로우)
- [DefaultChatAgent 워크플로우](#defaultchatagent-워크플로우)

## 전체 시스템 워크플로우

```mermaid
flowchart LR
    U["User"] --> FE["Frontend UI<br/>frontend/src/App.jsx"]
    FE --> API["POST /api/chat<br/>backend/api/routes.py"]
    API --> OG["Orchestrator Graph"]

    OG --> Q["QuizAgent"]
    OG --> E["EmailAgent"]
    OG --> R["RiskManagingAgent"]
    OG --> D["DefaultChatAgent"]

    Q --> NR["normalize_response"]
    E --> NR
    R --> NR
    D --> NR

    NR --> RESP["ChatResponse<br/>type/message/report/meta"]
    RESP --> NORM["normalizeApiResponse<br/>frontend/src/lib/normalizers.js"]

    NORM --> UI1["Text or Error Render"]
    NORM --> UI2["Risk ReportCard Render"]
    NORM --> UI3["Quiz Panel Update"]
```

## Orchestrator 워크플로우

```mermaid
flowchart TD
    A["load_session_state_node"] --> B["detect_intent_and_route_node"]
    B --> C{"Route decision"}

    C -->|force mode| C1["set selected_agent"]
    C -->|risk keyword| C2["selected_agent=riskmanaging"]
    C -->|quiz keyword| C3["selected_agent=quiz"]
    C -->|email keyword| C4["selected_agent=email"]
    C -->|active agent| C5["keep active_agent"]
    C -->|fallback| C6["LLM intent classify"]

    C1 --> D["call_agent_node"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    C6 --> D

    D --> E["finalize_and_save_state_node"]
    E --> F["normalize_response_node"]
    F --> G["End: ChatResponse"]
```

## QuizAgent 워크플로우

```mermaid
flowchart TD
    A["QuizAgent.run"] --> B{"pending_quiz and answer input"}

    B -->|yes| C["immediate grading response"]
    C --> Z["return to Orchestrator"]

    B -->|no| D["perform_rag_search_node"]
    D --> E["prepare_llm_messages_node"]
    E --> F["call_llm_and_parse_response_node"]
    F --> G{"quiz JSON parsed"}

    G -->|yes| H["extract questions and rebalance answers"]
    G -->|no| I["keep raw or fallback response"]

    H --> J["format_quiz_output_node"]
    I --> J
    J --> Z
```

## EmailAgent 워크플로우

```mermaid
flowchart TD
    A["EmailAgent.run"] --> B["perform_rag_search_node"]
    B --> C["prepare_llm_messages_node"]

    C --> D{"review and no email body"}
    D -->|yes| E["build required_fields message"]
    E --> H["call_llm_and_parse_response_node early return"]

    D -->|no| F["LLM call and JSON parse"]
    F --> G{"fallback needed"}
    G -->|yes| R["build rule-based review"]
    G -->|no| I["use LLM output"]

    R --> J["format_email_output_node"]
    I --> J
    H --> J
    J --> Z["return to Orchestrator"]
```

fallback 예시:
- LLM API 오류
- 리뷰 응답이 원문 echo인 경우
- 수신국가 미지정인데 국가 가정 리뷰가 섞인 경우
- LLM 미초기화 + 리뷰 본문 존재

## RiskManagingAgent 워크플로우

```mermaid
flowchart TD
    A["prepare_risk_state_node"] --> B["detect_trigger_and_similarity_node"]
    B --> C{"analysis_required"}

    C -->|no| G["format_final_output_node"]
    C -->|yes| D["assess_conversation_progress_node"]

    D --> E{"analysis_ready"}
    E -->|no| G
    E -->|yes| F["perform_full_analysis_node"]

    F --> F1["RAGConnector"]
    F1 --> F2["RiskEngine"]
    F2 --> F3["ReportGenerator"]
    F3 --> G

    G --> Z["End: RiskManagingAgentResponse"]
```

## DefaultChatAgent 워크플로우

```mermaid
flowchart TD
    A["DefaultChatAgent.run"] --> B["build messages with prompt and history"]
    B --> C["LLM call solar-pro"]
    C -->|success| D["build response message"]
    C -->|failure| E["fallback error message"]
    D --> F["return response and history"]
    E --> F
```

## 코드 기준 파일

- `backend/api/routes.py`
- `backend/agents/orchestrator/graph.py`
- `backend/agents/orchestrator/nodes.py`
- `backend/agents/quiz_agent/graph.py`
- `backend/agents/quiz_agent/quiz_agent.py`
- `backend/agents/quiz_agent/nodes.py`
- `backend/agents/email_agent/graph.py`
- `backend/agents/email_agent/email_agent.py`
- `backend/agents/email_agent/nodes.py`
- `backend/agents/riskmanaging/graph.py`
- `backend/agents/riskmanaging/nodes.py`
- `backend/agents/default_chat/default_chat_agent.py`
- `frontend/src/App.jsx`
- `frontend/src/lib/normalizers.js`
