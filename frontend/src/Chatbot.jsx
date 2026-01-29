import { useEffect, useMemo, useState } from "react";
import "./chatbot.css";
import { useAppState } from "./appState";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const WELCOME_MESSAGE = {
  role: "assistant",
  content: "안녕하세요! 주식 투자에 대해 궁금한 점을 편하게 물어보세요.",
};

const STORAGE_KEY_CURRENT = "chat_current_session_id";

export default function Chatbot() {
  const { state, setState } = useAppState();
  const chat = state.chat;

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [stockRecs, setStockRecs] = useState([]);
  const [stockRecsLoading, setStockRecsLoading] = useState(false);
  const [stockRecsError, setStockRecsError] = useState(null);

  const currentSession = useMemo(() => {
    return chat.sessions.find((s) => s.id === chat.currentSessionId) || null;
  }, [chat.sessions, chat.currentSessionId]);

  const hasMessages = currentSession && currentSession.messages.length > 1;

  async function createBackendSession() {
    const res = await fetch(`${API_BASE}/session`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to create session");
    const data = await res.json();
    return data.session_id;
  }

  async function fetchStockRecs() {
    setStockRecsLoading(true);
    setStockRecsError(null);
    try {
      const res = await fetch(`${API_BASE}/stocks/recommendations?limit=2`);
      if (!res.ok) throw new Error("Failed to load stock recommendations");
      const data = await res.json();
      setStockRecs(data.items || []);
    } catch (e) {
      console.error(e);
      setStockRecsError("추천 종목을 불러오지 못했습니다.");
      setStockRecs([]);
    } finally {
      setStockRecsLoading(false);
    }
  }

  useEffect(() => {
    const savedCurrent = localStorage.getItem(STORAGE_KEY_CURRENT);
    if (savedCurrent && chat.sessions.some((s) => s.id === savedCurrent)) {
      if (chat.currentSessionId !== savedCurrent) {
        setState((prev) => ({
          ...prev,
          chat: { ...prev.chat, currentSessionId: savedCurrent },
        }));
      }
      return;
    }

    if (chat.sessions.length > 0) {
      if (chat.currentSessionId) {
        localStorage.setItem(STORAGE_KEY_CURRENT, chat.currentSessionId);
      }
      return;
    }

    (async () => {
      try {
        const sessionId = await createBackendSession();
        const newSession = {
          id: sessionId,
          title: "새로운 챗",
          messages: [WELCOME_MESSAGE],
          createdAt: new Date().toISOString(),
        };

        setState((prev) => ({
          ...prev,
          chat: {
            ...prev.chat,
            sessions: [newSession],
            currentSessionId: newSession.id,
          },
        }));

        localStorage.setItem(STORAGE_KEY_CURRENT, newSession.id);
      } catch (e) {
        console.error(e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchStockRecs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (chat.currentSessionId) {
      localStorage.setItem(STORAGE_KEY_CURRENT, chat.currentSessionId);
    }
  }, [chat.currentSessionId]);

  const handleSend = async (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading || !currentSession) return;

    setState((prev) => ({
      ...prev,
      chat: {
        ...prev.chat,
        sessions: prev.chat.sessions.map((s) =>
          s.id === prev.chat.currentSessionId
            ? {
                ...s,
                messages: [...s.messages, { role: "user", content: trimmed }],
                title:
                  s.messages.length === 1
                    ? trimmed.slice(0, 30) + (trimmed.length > 30 ? "..." : "")
                    : s.title,
              }
            : s
        ),
      },
    }));

    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat/${currentSession.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });
      if (!res.ok) throw new Error("Chat request failed");
      const data = await res.json();

      setState((prev) => ({
        ...prev,
        chat: {
          ...prev.chat,
          sessions: prev.chat.sessions.map((s) =>
            s.id === prev.chat.currentSessionId
              ? {
                  ...s,
                  messages: [
                    ...s.messages,
                    {
                      role: "assistant",
                      content:
                        data.answer + (data.used_db ? " (뉴스 DB 사용)" : " (일반지식 기반)"),
                    },
                  ],
                }
              : s
          ),
        },
      }));
    } catch (err) {
      console.error(err);
      setState((prev) => ({
        ...prev,
        chat: {
          ...prev.chat,
          sessions: prev.chat.sessions.map((s) =>
            s.id === prev.chat.currentSessionId
              ? {
                  ...s,
                  messages: [
                    ...s.messages,
                    { role: "assistant", content: "서버 오류가 발생했습니다." },
                  ],
                }
              : s
          ),
        },
      }));
    } finally {
      setLoading(false);
    }
  };

  const sidebarOpen = chat.sidebarOpen ?? true;

  return (
    <div className="chatbot-container">
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <button
            className="new-chat-button"
            onClick={async () => {
              const sessionId = await createBackendSession();
              const newSession = {
                id: sessionId,
                title: "새로운 챗",
                messages: [WELCOME_MESSAGE],
                createdAt: new Date().toISOString(),
              };

              setState((prev) => ({
                ...prev,
                chat: {
                  ...prev.chat,
                  sessions: [newSession, ...prev.chat.sessions],
                  currentSessionId: newSession.id,
                },
              }));

              localStorage.setItem(STORAGE_KEY_CURRENT, newSession.id);
            }}
          >
            <span className="plus-icon">+</span>
            <span className="button-text">새로운 챗</span>
          </button>
        </div>

        <div className="chat-list">
          {chat.sessions.map((session) => (
            <div
              key={session.id}
              className={`chat-item ${session.id === chat.currentSessionId ? "active" : ""}`}
              onClick={() =>
                setState((prev) => ({
                  ...prev,
                  chat: { ...prev.chat, currentSessionId: session.id },
                }))
              }
            >
              <div className="chat-icon">💬</div>
              <span className="chat-title">{session.title}</span>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-content">
        {/* 여기만 스크롤 되게 만들기 (추천+메시지가 함께 스크롤됨) */}
        <div className="scroll-panel">
          <section className="stock-recs">
            <div className="stock-recs-header">
              <div className="stock-recs-title">오늘의 추천 종목</div>
            </div>

            {stockRecsError && <div className="stock-recs-error">{stockRecsError}</div>}

            <div className="stock-recs-grid">
              {(stockRecsLoading ? [1, 2] : stockRecs).map((x, idx) => {
                if (stockRecsLoading) {
                  return (
                    <div key={idx} className="stock-card skeleton">
                      <div className="skeleton-line w60" />
                      <div className="skeleton-line w40" />
                      <div className="skeleton-line w80" />
                    </div>
                  );
                }

                return (
                  <div key={x.symbol} className="stock-card">
                    <div className="stock-top">
                      <div className="stock-symbol">{x.symbol}</div>
                      <div className="stock-market">{x.market}</div>
                    </div>

                    <div className="stock-name">{x.name}</div>

                    <div className="stock-metrics">
                      {typeof x.price === "number" && (
                        <span className="stock-price">${x.price.toFixed(2)}</span>
                      )}
                      {typeof x.change_pct === "number" && (
                        <span className={`stock-change ${x.change_pct >= 0 ? "up" : "down"}`}>
                          {x.change_pct >= 0 ? "+" : ""}
                          {x.change_pct.toFixed(2)}%
                        </span>
                      )}
                    </div>

                    <div className="stock-headline">{x.headline}</div>

                    <div className="stock-tooltip">
                      <div className="tooltip-title">추천 이유</div>
                      <div className="tooltip-body">{x.why}</div>
                      {x.risk && (
                        <>
                          <div className="tooltip-title" style={{ marginTop: 10 }}>
                            리스크
                          </div>
                          <div className="tooltip-body">{x.risk}</div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <div className={`chat-area ${hasMessages ? "top-aligned" : "center-aligned"}`}>
            {currentSession ? (
              <div className="messages-container">
                {currentSession.messages.map((m, idx) => (
                  <div
                    key={idx}
                    className={`message ${m.role === "user" ? "user-message" : "assistant-message"}`}
                  >
                    <div className="message-avatar">{m.role === "user" ? "👤" : "🤖"}</div>
                    <div className="message-bubble">
                      <div className="message-content">{m.content}</div>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="loading-container">
                    <div className="loading-dots">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ padding: 20 }}>세션을 준비 중입니다...</div>
            )}
          </div>
        </div>

        {/* 입력창은 스크롤 밖(항상 아래) */}
        <div className="input-container">
          <form onSubmit={handleSend} className="input-form">
            <input
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="궁금한 점을 물어보세요..."
              disabled={loading || !currentSession}
            />
            <button type="submit" className="send-button" disabled={loading || !input.trim() || !currentSession}>
              <span className="send-icon">↑</span>
            </button>
          </form>
          <p className="input-disclaimer">
            AI가 생성한 정보는 참고용이며, 실제 투자 결정 전 전문가와 상담하세요.
          </p>
        </div>
      </main>
    </div>
  );
}