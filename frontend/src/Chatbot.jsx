import { useEffect, useMemo, useState } from "react";
import "./chatbot.css";
import { useAppState } from "./appState";

const API_BASE = "http://localhost:8000";

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

  // 현재 선택된 세션 (id == backend session id)
  const currentSession = useMemo(() => {
    return chat.sessions.find((s) => s.id === chat.currentSessionId) || null;
  }, [chat.sessions, chat.currentSessionId]);

  const hasMessages = currentSession && currentSession.messages.length > 1;

  async function createBackendSession() {
    const res = await fetch(`${API_BASE}/session`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to create session");
    const data = await res.json();
    return data.session_id; // 이 값을 프론트 세션 id로 그대로 사용
  }

  // 앱 처음 들어왔는데 세션이 없으면 backend에서 세션 발급받아 생성
  // 새로고침/탭 이동 시 마지막 currentSessionId 복원
  useEffect(() => {
    // 1) localStorage에 저장된 currentSessionId가 있으면 복원 시도
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

    // 2) 세션이 이미 있으면(앱 상태에 남아있으면) currentSessionId만 저장
    if (chat.sessions.length > 0) {
      if (chat.currentSessionId) {
        localStorage.setItem(STORAGE_KEY_CURRENT, chat.currentSessionId);
      }
      return;
    }

    // 3) 세션이 없다면 백엔드에서 새로 발급
    (async () => {
      try {
        const sessionId = await createBackendSession();

        const newSession = {
          id: sessionId, // 백엔드 session_id와 동일하게
          title: "새로운 채팅",
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

  // currentSessionId 변경될 때마다 localStorage에 저장 (탭 이동/새로고침 유지)
  useEffect(() => {
    if (chat.currentSessionId) {
      localStorage.setItem(STORAGE_KEY_CURRENT, chat.currentSessionId);
    }
  }, [chat.currentSessionId]);

  const handleSend = async (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading || !currentSession) return;

    // 1) user message 먼저 UI에 반영
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
      // 이제부터 /chat/{session_id} 에서 session_id는 currentSession.id 하나만 사용
      const res = await fetch(`${API_BASE}/chat/${currentSession.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });
      if (!res.ok) throw new Error("Chat request failed");
      const data = await res.json();

      const answer = data.answer;
      const usedDb = data.used_db;

      // 2) assistant message 반영
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
                      content: answer + (usedDb ? " (뉴스 DB 사용)" : " (일반지식 기반)"),
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
                    { role: "assistant", content: "죄송합니다. 서버 연결에 문제가 있습니다." },
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

  const handleNewChat = async () => {
    try {
      const sessionId = await createBackendSession();
      const newSession = {
        id: sessionId, // 백엔드 session_id를 그대로
        title: "새로운 채팅",
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
    } catch (e) {
      console.error(e);
    }
  };

  const handleSelectSession = (sessionId) => {
    setState((prev) => ({
      ...prev,
      chat: { ...prev.chat, currentSessionId: sessionId },
    }));
    localStorage.setItem(STORAGE_KEY_CURRENT, sessionId);
  };

  const handleDeleteSession = (sessionId, e) => {
    e.stopPropagation();
    if (chat.sessions.length === 1) return;

    const remaining = chat.sessions.filter((s) => s.id !== sessionId);
    const nextCurrent =
      chat.currentSessionId === sessionId ? remaining[0]?.id ?? null : chat.currentSessionId;

    setState((prev) => ({
      ...prev,
      chat: { ...prev.chat, sessions: remaining, currentSessionId: nextCurrent },
    }));

    // current 삭제한 경우 localStorage도 갱신
    if (chat.currentSessionId === sessionId) {
      if (nextCurrent) localStorage.setItem(STORAGE_KEY_CURRENT, nextCurrent);
      else localStorage.removeItem(STORAGE_KEY_CURRENT);
    }
  };

  const sidebarOpen = chat.sidebarOpen ?? true;

  return (
    <div className="chatbot-container">
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <button className="new-chat-button" onClick={handleNewChat}>
            <span className="plus-icon">+</span>
            <span className="button-text">새 채팅</span>
          </button>
        </div>

        <div className="chat-list">
          {chat.sessions.map((session) => (
            <div
              key={session.id}
              className={`chat-item ${session.id === chat.currentSessionId ? "active" : ""}`}
              onClick={() => handleSelectSession(session.id)}
            >
              <div className="chat-icon">💬</div>
              <span className="chat-title">{session.title}</span>
              {chat.sessions.length > 1 && (
                <button className="delete-button" onClick={(e) => handleDeleteSession(session.id, e)}>
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>

        <button
          className="toggle-sidebar-bottom"
          onClick={() =>
            setState((prev) => ({
              ...prev,
              chat: { ...prev.chat, sidebarOpen: false },
            }))
          }
        >
          <span>◀</span>
        </button>
      </aside>

      <main className="main-content">
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
