from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4 

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Agent.py import
from .Agent import app as agent_app, AgentState

# 상대경로 import
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
    os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "Chroma_db" / "News_chroma_db"))
)
CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION", "naver_finance_news_chunks"
)

# ----------------------------
# vLLM 설정 (Ollama 대신)
# ----------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "skt/A.X-4.0-Light")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "vllm-key")

# ----------------------------
# HuggingFace Embedding 설정
# ----------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")

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
    category: str = "unknown"
    sub_category: str = ""

class SessionChatResponse(BaseModel):
    session_id: str
    answer: str
    category: str = "unknown"
    sub_category: str = ""


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
    # [수정] 함수 정의에 있는 이름(embedding_model_name)과 정확히 맞춥니다.
    client, col = get_collection(
        persist_dir=Path("/data/ephemeral/home/pro-nlp-finalproject-nlp-06/Chroma_db/News_chroma_db"),
        collection_name=CHROMA_COLLECTION,
        embedding_model_name=EMBEDDING_MODEL  # hf_embed_model이 아니라 이 이름이어야 합니다!
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
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": os.getenv("REDIS_PORT", "6379"),
    }


# Agent.py 사용하는 채팅
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Agent.py의 LangGraph를 실행하여 사용자 질문에 답변
    """
    # 초기 상태 설정
    state: AgentState = {
        "query": request.message,
        "category": "",
        "sub_category": "",
        "debate_history": [],
        "debate_count": 0,
        "response": "",
    }
    
    try:
        # Agent 그래프 실행
        result = agent_app.invoke(state)
        
        # 응답 반환
        return ChatResponse(
            answer=result.get("response", "응답을 생성할 수 없습니다."),
            category=result.get("category", "unknown"),
            sub_category=result.get("sub_category", "")
        )
    
    except Exception as e:
        print(f"[ERROR] Agent 실행 중 오류 발생: {e}")
        return ChatResponse(
            answer=f"오류가 발생했습니다: {str(e)}",
            category="error",
            sub_category=""
        )


#새 채팅방(세션) 만들기
@app.post("/session")
def create_session():
    session_id = str(uuid4())
    store = get_store()
    store.create_session(session_id)
    return {"session_id": session_id}



# 세션 기반 채팅: Agent.py 사용
@app.post("/chat/{session_id}", response_model=SessionChatResponse)
def chat_with_session(session_id: str, request: ChatRequest):
    store = get_store()

    history = store.get_last_n(session_id, n=3, chronological=True)

    # 0) 과거 대화 최근 10개를 먼저 가져오기
    if history:
        # 모델이 이해하기 쉽게 역할을 명시해서 합쳐줍니다.
        formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        full_query = f"[이전 대화 내역]\n{formatted_history}\n\n[현재 질문]\n{request.message}"
    else:
        full_query = request.message

    # 2) 사용자 메시지 저장 (순수한 질문만 저장해야 다음 턴에 안 꼬입니다)
    store.add_message(session_id, "user", request.message)

    # 2) Agent 상태 설정 (히스토리 포함)
    state: AgentState = {
        "query": full_query,
        "category": "",
        "sub_category": "",
        "debate_history": history,  # 과거 대화 포함
        "debate_count": 0,
        "response": "",
    }

    try:
        # Agent 그래프 실행
        result = agent_app.invoke(state)
        
        answer = result.get("response", "응답을 생성할 수 없습니다.")
        category = result.get("category", "unknown")
        sub_category = result.get("sub_category", "")
        
        # 3) 어시스턴트 메시지 저장
        store.add_message(session_id, "assistant", answer)
        
        return SessionChatResponse(
            session_id=session_id,
            answer=answer,
            category=category,
            sub_category=sub_category
        )
    
    except Exception as e:
        print(f"[ERROR] Agent 실행 중 오류 발생: {e}")
        error_msg = f"오류가 발생했습니다: {str(e)}"
        store.add_message(session_id, "assistant", error_msg)
        
        return SessionChatResponse(
            session_id=session_id,
            answer=error_msg,
            category="error",
            sub_category=""
        )




#최근 N개 메시지 조회 (기본 10개)
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