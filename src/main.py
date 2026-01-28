from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4 

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 상대경로 import
from .model_ollama import RagNewsChatService
from redis_dir.redis_storage import RedisSessionStore
from .chroma_store import get_collection

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



class NewsItem(BaseModel):
    title: str
    press: str
    date: str
    date_iso: str
    date_ts: int
    link: str
    preview_lines: List[str]
    summary: str


class NewsListResponse(BaseModel):
    items: List[NewsItem]



def _first_3_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:3]


def fetch_latest_news(limit: int = 20) -> List[Dict[str, Any]]:
    # Chroma collection 열기
    client, col = get_collection(
        persist_dir=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_embed_model=OLLAMA_EMBED_MODEL,
    )

    # Chroma 버전마다 limit/offset 지원이 다를 수 있어서 방어적으로 처리
    try:
        got = col.get(include=["metadatas", "documents"], limit=5000)
    except TypeError:
        got = col.get(include=["metadatas", "documents"])

    metadatas = got.get("metadatas") or []
    documents = got.get("documents") or []

    # link 기준으로 기사 단위 묶기
    by_link: Dict[str, Dict[str, Any]] = {}

    for meta, doc in zip(metadatas, documents):
        if not meta:
            continue
        link = (meta.get("link") or "").strip()
        if not link:
            continue

        # 기본 기사 정보
        item = by_link.get(link)
        if item is None:
            item = {
                "title": meta.get("title", ""),
                "press": meta.get("press", ""),
                "date": meta.get("date", ""),
                "date_iso": meta.get("date_iso", ""),
                "date_ts": int(meta.get("date_ts", 0) or 0),
                "link": link,
                "summary": meta.get("summary", ""),
                "preview_doc": "",
                "preview_lines": [],
                "best_chunk_index": 10**9,
            }
            by_link[link] = item

        # chunk_index==0(혹은 가장 작은 chunk_index)을 미리보기로 사용
        chunk_index = meta.get("chunk_index")
        try:
            chunk_index = int(chunk_index) if chunk_index is not None else 10**9
        except Exception:
            chunk_index = 10**9

        if chunk_index < item["best_chunk_index"]:
            item["best_chunk_index"] = chunk_index
            item["preview_doc"] = doc or ""

    # preview_lines 만들고 정렬/슬라이스
    items = []
    for link, item in by_link.items():
        preview_lines = _first_3_lines(item.get("preview_doc", ""))
        items.append(
            {
                "title": item.get("title", ""),
                "press": item.get("press", ""),
                "date": item.get("date", ""),
                "date_iso": item.get("date_iso", ""),
                "date_ts": int(item.get("date_ts", 0) or 0),
                "link": item.get("link", ""),
                "preview_lines": preview_lines,
                "summary": item.get("summary", ""),
            }
        )

    items.sort(key=lambda x: x.get("date_ts", 0), reverse=True)
    return items[:limit]



class StockRecOut(BaseModel):
    symbol: str                 # 예: AAPL
    name: str                   # 예: Apple Inc.
    market: Optional[str] = None  # 예: NASDAQ
    price: Optional[float] = None # 예시
    change_pct: Optional[float] = None # 예시
    headline: str               # 카드에 보이는 한 줄 추천 문구
    why: str                    # hover 시 노출되는 상세 이유
    risk: Optional[str] = None  # 리스크 한줄(선택)

class StockRecListOut(BaseModel):
    items: List[StockRecOut]


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



@app.get("/news/latest", response_model=NewsListResponse)
def latest_news(limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    items = fetch_latest_news(limit=limit)
    return {"items": items}



@app.get("/stocks/recommendations", response_model=StockRecListOut)
def get_stock_recommendations(limit: int = Query(2, ge=1, le=10)):
    # 지금은 예시 2개 하드코딩.
    # 나중에 실제 추천으로 교체하면 됨.
    sample = [
        StockRecOut(
            symbol="AAPL",
            name="Apple Inc.",
            market="NASDAQ",
            price=198.12,
            change_pct=1.24,
            headline="현금흐름/자사주 매입 기반의 방어적 빅테크",
            why=(
                "• 실적 변동성이 상대적으로 낮고, 서비스 매출 비중이 커서 수익 구조가 안정적입니다.\n"
                "• 강한 현금흐름을 바탕으로 자사주 매입/배당을 지속해 주주환원 여력이 큽니다.\n"
                "• 단기 변동성(금리/기술주 조정)에도 포트폴리오 코어로 편입하기 좋습니다."
            ),
            risk="밸류에이션(멀티플) 부담 시 조정 폭이 커질 수 있음",
        ),
        StockRecOut(
            symbol="MSFT",
            name="Microsoft",
            market="NASDAQ",
            price=431.55,
            change_pct=0.78,
            headline="클라우드 + AI 수요의 구조적 수혜",
            why=(
                "• Azure 성장과 기업용 소프트웨어 구독 매출로 장기 성장성이 탄탄합니다.\n"
                "• AI 도입(업무 자동화/코파일럿) 확산이 매출 업사이드로 연결될 가능성이 있습니다.\n"
                "• 경기 둔화 국면에서도 엔터프라이즈 락인 효과가 강합니다."
            ),
            risk="클라우드 성장률 둔화/경쟁 심화 시 모멘텀 약화 가능",
        ),
    ]
    return {"items": sample[:limit]}