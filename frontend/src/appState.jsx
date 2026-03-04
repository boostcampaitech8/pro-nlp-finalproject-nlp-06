import { createContext, useContext, useEffect, useMemo, useState } from "react";

const AppStateContext = createContext(null);

// localStorage key
const STORAGE_KEY = "stockq_app_state_v1";

// 초기값
const defaultState = {
  chat: {
    sessions: [], // [{ id, backendSessionId, title, messages, createdAt }]
    currentSessionId: null,
    sidebarOpen: true,
  },
  news: {
    // 필요하면 news도 여기에 저장 가능
    items: [],
    lastFetchedAt: null,
  },
};

// localStorage 복구
function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState;
    const parsed = JSON.parse(raw);

    // 기본 구조 + 저장된 값 merge
    return {
      ...defaultState,
      ...parsed,
      chat: { ...defaultState.chat, ...(parsed.chat || {}) },
      news: { ...defaultState.news, ...(parsed.news || {}) },
    };
  } catch {
    return defaultState;
  }
}

export function AppStateProvider({ children }) {
  const [state, setState] = useState(() => loadState());

  // state 바뀔 때마다 localStorage 저장 => 새로고침해도 유지
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      console.error("Failed to save state to localStorage:", e);
    }
  }, [state]);

  const value = useMemo(() => ({ state, setState }), [state]);
  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
