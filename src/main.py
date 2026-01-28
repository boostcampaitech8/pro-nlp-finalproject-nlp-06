from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 상대경로 import
from .model_ollama import RagNewsChatService
from redis_dir.redis_storage import RedisSessionStore

app = FastAPI()

# ----------------------------
# CORS
# ----------------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# PROJECT_ROOT 자동 계산
# ----------------------------
THIS_FILE = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = THIS_FILE.parents[1]  # project/

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
)

# ----------------------------
# 공통 경로 (pipeline / DAG / FastAPI 통일)
# ----------------------------
CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_news"))
)
CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION", "naver_finance_news_chunks"
)

# ----------------------------
# Ollama
# ----------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ----------------------------
# Retrieval 튜닝
# ----------------------------
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "48"))
TOP_ARTICLES = int(os.getenv("TOP_ARTICLES", "5"))
MAX_CHUNKS_PER_ARTICLE = int(os.getenv("MAX_CHUNKS_PER_ARTICLE", "3"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.7"))
MIN_DOCS_AFTER_FILTER = int(os.getenv("MIN_DOCS_AFTER_FILTER", "12"))
ENABLE_QUERY_REFINE = os.getenv("ENABLE_QUERY_REFINE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ----------------------------
# RAG 서비스 (lazy init)
# ----------------------------
_rag_service: Optional[RagNewsChatService] = None


def get_rag_service() -> RagNewsChatService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagNewsChatService(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            llm_model=OLLAMA_LLM_MODEL,
            embedding_model=OLLAMA_EMBED_MODEL,
            ollama_base_url=OLLAMA_BASE_URL,
            retrieval_k=RETRIEVAL_K,
            top_articles=TOP_ARTICLES,
            max_chunks_per_article=MAX_CHUNKS_PER_ARTICLE,
            max_distance=MAX_DISTANCE,
            min_docs_after_filter=MIN_DOCS_AFTER_FILTER,
            enable_query_refine=ENABLE_QUERY_REFINE,
            debug=DEBUG,
        )
    return _rag_service

# ----------------------------
# Redis Session Store (lazy init)
# ----------------------------
_store: Optional[RedisSessionStore] = None


def get_store() -> RedisSessionStore:
    global _store
    if _store is None:
        _store = RedisSessionStore()
        # 서버 시작 시 Redis 연결 문제를 빨리 감지하고 싶으면 ping 체크:
        try:
            _store.ping()
        except Exception as e:
            raise RuntimeError(f"Redis connection failed: {e}") from e
    return _store



# ----------------------------
# Request / Response
# ----------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    used_db: bool

class SessionChatResponse(BaseModel):
    session_id: str
    answer: str
    used_db: bool


class RecentMessagesResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "chroma_dir": str(CHROMA_DIR),
        "chroma_collection": CHROMA_COLLECTION,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_llm_model": OLLAMA_LLM_MODEL,
        "ollama_embed_model": OLLAMA_EMBED_MODEL,
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": os.getenv("REDIS_PORT", "6379"),
    }


#기존 
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    rag = get_rag_service()
    answer, used_db = rag.answer(request.message)
    return ChatResponse(answer=answer, used_db=used_db)

#새 채팅방(세션) 만들기
@app.post("/session")
def create_session():
    session_id = str(uuid4())
    store = get_store()
    store.create_session(session_id)
    return {"session_id": session_id}



# 세션 기반 채팅: "최근 5개 히스토리"를 프롬프트에 포함
@app.post("/chat/{session_id}", response_model=SessionChatResponse)
def chat_with_session(session_id: str, request: ChatRequest):
    store = get_store()
    rag = get_rag_service()

    # 0) 과거 대화 최근 5개를 먼저 가져오기 (현재 user 메시지 저장 전에!)
    history = store.get_last_n(session_id, n=10, chronological=True)

    # 1) 사용자 메시지 저장
    store.add_message(session_id, "user", request.message)

    # 2) 히스토리 포함하여 답변 생성
    answer, used_db = rag.answer_with_history(request.message, history)

    # 3) 어시스턴트 메시지 저장
    store.add_message(session_id, "assistant", answer)

    return SessionChatResponse(session_id=session_id, answer=answer, used_db=used_db)




#최근 N개 메시지 조회 (기본 5개)
@app.get("/chat/{session_id}/recent", response_model=RecentMessagesResponse)
def recent_messages(session_id: str, limit: int = 10):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    store = get_store()
    msgs = store.get_last_n(session_id, n=limit, chronological=True)
    return RecentMessagesResponse(session_id=session_id, messages=msgs)