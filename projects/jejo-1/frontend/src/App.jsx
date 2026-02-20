import { useEffect, useMemo, useRef, useState } from "react";
import MarkdownMessage from "./components/MarkdownMessage";
import ReportCard from "./components/ReportCard";
import { normalizeApiResponse, shouldForceQuizMode } from "./lib/normalizers";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
const SESSION_STORAGE_KEY = "trade_onboarding_session_id";

const MODE_OPTIONS = [
  { label: "자동 (Orchestrator 판단)", value: "auto" },
  { label: "리스크 분석 강제", value: "riskmanaging" },
  { label: "퀴즈 강제", value: "quiz" },
  { label: "이메일 강제", value: "email" },
  { label: "기본 대화 강제", value: "default_chat" },
];

function createSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildQuestionKey(question) {
  const questionText = String(question?.question || "");
  const choices = Array.isArray(question?.choices) ? question.choices.map(String).join("|") : "";
  return `${questionText}::${choices}`;
}

function mergeQuizQuestions(existingQuestions, incomingQuestions) {
  const merged = [...existingQuestions];
  const seen = new Set(existingQuestions.map((question) => buildQuestionKey(question)));

  for (const incoming of incomingQuestions) {
    const key = buildQuestionKey(incoming);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push({
      ...incoming,
      id: key,
    });
  }
  return merged;
}

function QuizWorkspace({
  questions,
  answers,
  activeIndex,
  onSelectQuestion,
  onAnswer,
  onReset,
  onGenerateMore,
  onClose,
  isSending,
}) {
  const solvedCount = Object.keys(answers).length;
  const correctCount = questions.reduce((count, question) => {
    const answerState = answers[question.id];
    if (!answerState) {
      return count;
    }
    return answerState.isCorrect ? count + 1 : count;
  }, 0);
  const totalCount = questions.length;
  const safeActiveIndex =
    totalCount > 0 ? Math.min(Math.max(activeIndex, 0), totalCount - 1) : 0;
  const isCompleted = totalCount > 0 && solvedCount >= totalCount;
  const activeQuestion = totalCount > 0 ? questions[safeActiveIndex] : null;
  const answerState = activeQuestion ? answers[activeQuestion.id] : null;
  const selectedChoice = Number(answerState?.selected);
  const correctChoice = Number(activeQuestion?.answer);
  const isAnswered = Boolean(answerState);
  const isCurrentAnswerCorrect = isAnswered && Boolean(answerState?.isCorrect);
  const wrongQuestions = questions.filter((question) => {
    const answerState = answers[question.id];
    if (!answerState) {
      return false;
    }
    return !answerState.isCorrect;
  });
  const scorePercent =
    totalCount > 0 ? Math.round((correctCount / totalCount) * 100) : 0;

  return (
    <aside className="quiz-panel">
      <header className="quiz-panel-head">
        <h3>퀴즈 풀이</h3>
        <div className="quiz-panel-head-right">
          <div className="quiz-panel-actions">
            <button
              type="button"
              onClick={onGenerateMore}
              disabled={isSending}
            >
              추가 문제 생성
            </button>
            <button type="button" onClick={onReset} disabled={questions.length === 0 || isSending}>
              초기화
            </button>
          </div>
          <button type="button" className="quiz-panel-close" onClick={onClose}>
            닫기
          </button>
        </div>
      </header>
      <p className="muted">
        {questions.length === 0
          ? "퀴즈가 생성되면 이 공간에서 바로 풀이할 수 있습니다."
          : `진행: ${solvedCount}/${questions.length} | 정답: ${correctCount}`}
      </p>

      <div className="quiz-panel-body">
        {questions.length === 0 ? (
          <div className="quiz-empty">채팅에서 "무역실무 퀴즈내줘"를 입력해 시작하세요.</div>
        ) : (
          <>
            {activeQuestion ? (
              <article className="quiz-workspace-card current">
                <strong>
                  [문제 {safeActiveIndex + 1}/{totalCount}] {String(activeQuestion.question)}
                </strong>
                <p className="muted">선택 즉시 채점되고 해설이 아래에 표시됩니다.</p>
                <ul>
                  {(Array.isArray(activeQuestion.choices) ? activeQuestion.choices : []).map(
                    (choice, choiceIdx) => (
                      <li key={`${activeQuestion.id}-${choiceIdx}`}>
                        {(() => {
                          let optionClass = "quiz-option";
                          if (isAnswered && Number.isFinite(correctChoice)) {
                            if (choiceIdx === correctChoice) {
                              optionClass += " correct";
                            } else if (choiceIdx === selectedChoice) {
                              optionClass += " wrong";
                            }
                          }
                          return (
                            <button
                              type="button"
                              className={optionClass}
                              onClick={() => onAnswer(activeQuestion, choiceIdx)}
                              disabled={isSending || isAnswered}
                            >
                              {choiceIdx + 1}. {String(choice)}
                            </button>
                          );
                        })()}
                      </li>
                    )
                  )}
                </ul>

                {isAnswered ? (
                  <p className={`quiz-result ${isCurrentAnswerCorrect ? "correct" : "wrong"}`}>
                    {isCurrentAnswerCorrect
                      ? `정답입니다. (${selectedChoice + 1}번)`
                      : `오답입니다. 정답은 ${Number.isFinite(correctChoice) ? correctChoice + 1 : "-"}번입니다.`}
                  </p>
                ) : (
                  <p className="quiz-result">아직 답안을 선택하지 않았습니다.</p>
                )}

                {isAnswered && activeQuestion.explanation ? (
                  <div className="quiz-explanation">
                    <strong>해설</strong>
                    <p>{String(activeQuestion.explanation)}</p>
                  </div>
                ) : null}

                <div className="quiz-navigation">
                  <button
                    type="button"
                    onClick={() => onSelectQuestion(safeActiveIndex - 1)}
                    disabled={safeActiveIndex <= 0}
                  >
                    이전 문제
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelectQuestion(safeActiveIndex + 1)}
                    disabled={safeActiveIndex >= totalCount - 1}
                  >
                    다음 문제
                  </button>
                </div>
              </article>
            ) : null}

            {isCompleted ? (
              <section className="quiz-result-summary">
                <h4>채점 결과</h4>
                <p>
                  총 {totalCount}문제 중 {correctCount}문제 정답 ({scorePercent}%)
                </p>
                {wrongQuestions.length === 0 ? (
                  <p className="quiz-result correct">모든 문제를 맞췄습니다.</p>
                ) : (
                  <ul className="quiz-review-list">
                    {wrongQuestions.map((question, index) => {
                      const answerState = answers[question.id];
                      const selected = Number(answerState?.selected);
                      const answer = Number(question.answer);
                      const choices = Array.isArray(question.choices) ? question.choices : [];
                      const selectedText =
                        Number.isFinite(selected) &&
                        selected >= 0 &&
                        selected < choices.length
                          ? String(choices[selected])
                          : "-";
                      const answerText =
                        Number.isFinite(answer) &&
                        answer >= 0 &&
                        answer < choices.length
                          ? String(choices[answer])
                          : "-";
                      return (
                        <li key={`wrong-${question.id}`}>
                          <strong>
                            오답 {index + 1}. {String(question.question)}
                          </strong>
                          <p>
                            내 답:{" "}
                            {Number.isFinite(selected) ? `${selected + 1}번 (${selectedText})` : "-"}
                            <br />
                            정답: {Number.isFinite(answer) ? `${answer + 1}번 (${answerText})` : "-"}
                          </p>
                          {question.explanation ? (
                            <p className="muted">{String(question.explanation)}</p>
                          ) : null}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </section>
            ) : null}
          </>
        )}
      </div>
    </aside>
  );
}

function AssistantContent({ message }) {
  if (message.kind === "report") {
    return <ReportCard report={message.report} />;
  }

  if (message.kind === "quiz") {
    const count = Array.isArray(message.questions) ? message.questions.length : 0;
    return (
      <div className="quiz-notice">
        <strong>퀴즈 생성 완료 ({count}문제)</strong>
        <p className="muted">
          오른쪽 패널에서 풀이를 진행하세요. 추가 문제가 필요하면 "추가 문제"라고 입력하세요.
        </p>
      </div>
    );
  }

  if (message.kind === "email") {
    return (
      <div>
        <MarkdownMessage text={message.text} />
        <pre>{JSON.stringify(message.data, null, 2)}</pre>
      </div>
    );
  }

  if (message.kind === "json") {
    return <pre>{JSON.stringify(message.data, null, 2)}</pre>;
  }

  if (message.kind === "error") {
    return (
      <div className="error-box">
        <strong>요청 처리 오류</strong>
        <MarkdownMessage text={message.text} />
      </div>
    );
  }

  return <MarkdownMessage text={message.text} />;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [quizWorkspace, setQuizWorkspace] = useState({
    questions: [],
    answers: {},
  });
  const [quizActiveIndex, setQuizActiveIndex] = useState(0);
  const [isQuizPanelHidden, setIsQuizPanelHidden] = useState(false);
  const [mode, setMode] = useState("auto");
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [sessionId, setSessionId] = useState(() => {
    const cached = localStorage.getItem(SESSION_STORAGE_KEY);
    if (cached) {
      return cached;
    }
    const next = createSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, next);
    return next;
  });

  const listBottomRef = useRef(null);

  useEffect(() => {
    listBottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  useEffect(() => {
    setQuizActiveIndex((prev) => {
      if (quizWorkspace.questions.length === 0) {
        return 0;
      }
      return Math.min(prev, quizWorkspace.questions.length - 1);
    });
  }, [quizWorkspace.questions.length]);

  const lastAssistantMessage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === "assistant") {
        return messages[i];
      }
    }
    return null;
  }, [messages]);
  const hasQuizQuestions = quizWorkspace.questions.length > 0;
  const isQuizPanelOpen = hasQuizQuestions && !isQuizPanelHidden;

  const appendUserMessage = (text) => {
    const userMessage = {
      id: createSessionId(),
      role: "user",
      text,
    };
    setMessages((prev) => [...prev, userMessage]);
  };

  const appendAssistantMessage = (normalized) => {
    const assistantMessage = {
      id: createSessionId(),
      role: "assistant",
      ...normalized,
    };
    setMessages((prev) => [...prev, assistantMessage]);
  };

  const resetConversation = () => {
    const nextSession = createSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, nextSession);
    setSessionId(nextSession);
    setMessages([]);
    setQuizWorkspace({ questions: [], answers: {} });
    setQuizActiveIndex(0);
    setIsQuizPanelHidden(false);
    setInputValue("");
  };

  const appendQuizQuestionsToWorkspace = (questions) => {
    if (!Array.isArray(questions) || questions.length === 0) {
      return;
    }
    setIsQuizPanelHidden(false);
    setQuizWorkspace((prev) => ({
      ...prev,
      questions: mergeQuizQuestions(prev.questions, questions),
    }));
  };

  const handleWorkspaceAnswer = (question, selected) => {
    if (!question || !question.id) {
      return;
    }
    const correctAnswer = Number(question.answer);
    const isCorrect = Number.isFinite(correctAnswer) ? selected === correctAnswer : false;
    setQuizWorkspace((prev) => ({
      ...prev,
      answers: {
        ...prev.answers,
        [question.id]: {
          selected,
          isCorrect,
        },
      },
    }));
  };

  const setActiveQuizQuestion = (index) => {
    setQuizActiveIndex(() => {
      if (quizWorkspace.questions.length === 0) {
        return 0;
      }
      return Math.min(Math.max(index, 0), quizWorkspace.questions.length - 1);
    });
  };

  const resetWorkspaceProgress = () => {
    setQuizWorkspace((prev) => ({
      ...prev,
      answers: {},
    }));
    setQuizActiveIndex(0);
  };

  const sendMessage = async (rawText, options = {}) => {
    const text = String(rawText || "").trim();
    if (!text || isSending) {
      return;
    }

    appendUserMessage(text);
    setInputValue("");
    setIsSending(true);

    const payload = {
      session_id: sessionId,
      message: text,
    };

    if (options.forceQuizMode) {
      payload.context = { mode: "quiz" };
    } else if (mode !== "auto") {
      payload.context = { mode };
    } else if (shouldForceQuizMode(text, mode, lastAssistantMessage)) {
      payload.context = { mode: "quiz" };
    }

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const rawBody = await response.text();
      let data;
      try {
        data = JSON.parse(rawBody);
      } catch {
        data = { type: "chat", message: rawBody };
      }

      if (!response.ok) {
        const message =
          typeof data?.detail === "string"
            ? data.detail
            : `백엔드 오류 (${response.status})`;
        appendAssistantMessage({ kind: "error", text: message, data });
        return;
      }

      const normalized = normalizeApiResponse(data);
      appendAssistantMessage(normalized);
      if (normalized.kind === "quiz") {
        appendQuizQuestionsToWorkspace(normalized.questions || []);
      }
    } catch (error) {
      appendAssistantMessage({
        kind: "error",
        text: `서버와 통신할 수 없습니다. ${error instanceof Error ? error.message : ""}`,
      });
    } finally {
      setIsSending(false);
    }
  };

  const onSubmit = (event) => {
    event.preventDefault();
    sendMessage(inputValue);
  };

  const handleGenerateMoreQuiz = () => {
    sendMessage("추가 문제 만들어줘", { forceQuizMode: true });
  };

  const closeQuizPanel = () => {
    setIsQuizPanelHidden(true);
  };

  const reopenQuizPanel = () => {
    setIsQuizPanelHidden(false);
  };

  return (
    <div className={isQuizPanelOpen ? "page page-with-quiz" : "page"}>
      <aside className="sidebar">
        <h1>TradeOnboarding</h1>
        <p className="muted">무역 실무 온보딩 AI 코치</p>

        <label className="field-label" htmlFor="mode-select">
          라우팅 모드
        </label>
        <select
          id="mode-select"
          value={mode}
          onChange={(event) => setMode(event.target.value)}
          className="mode-select"
        >
          {MODE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <div className="session-box">
          <span>Session ID</span>
          <code>{sessionId}</code>
        </div>

        <button type="button" className="reset-btn" onClick={resetConversation}>
          새 대화 시작
        </button>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div className="chat-header-top">
            <h2>채팅</h2>
            {hasQuizQuestions && !isQuizPanelOpen ? (
              <button type="button" className="quiz-panel-toggle" onClick={reopenQuizPanel}>
                퀴즈 창 열기
              </button>
            ) : null}
          </div>
          <p className="muted">
            퀴즈/이메일/리스크 응답을 단일 UI에서 렌더링합니다.
          </p>
        </header>

        <section className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p>예시 입력: "무역실무 퀴즈내줘", "이메일 초안 영어로 만들어줘", "선적이 늦어져요"</p>
            </div>
          ) : null}

          {messages.map((message) => (
            <article
              key={message.id}
              className={message.role === "user" ? "message user" : "message assistant"}
            >
              <div className="message-label">{message.role === "user" ? "사용자" : "AI 비서"}</div>
              {message.role === "user" ? (
                <p className="multiline">{message.text}</p>
              ) : (
                <AssistantContent message={message} />
              )}
            </article>
          ))}

          {isSending ? <p className="muted">응답 생성 중...</p> : null}
          <div ref={listBottomRef} />
        </section>

        <form className="composer" onSubmit={onSubmit}>
          <input
            type="text"
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder="메시지를 입력하세요"
            disabled={isSending}
          />
          <button type="submit" disabled={isSending || !inputValue.trim()}>
            전송
          </button>
        </form>
      </main>

      {isQuizPanelOpen ? (
        <QuizWorkspace
          questions={quizWorkspace.questions}
          answers={quizWorkspace.answers}
          activeIndex={quizActiveIndex}
          onSelectQuestion={setActiveQuizQuestion}
          onAnswer={handleWorkspaceAnswer}
          onReset={resetWorkspaceProgress}
          onGenerateMore={handleGenerateMoreQuiz}
          onClose={closeQuizPanel}
          isSending={isSending}
        />
      ) : null}
    </div>
  );
}
