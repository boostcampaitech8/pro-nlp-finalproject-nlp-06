from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Agent.py import
from .Agent import app as agent_app, AgentState

# 상대경로 import
from redis_dir.redis_storage import RedisSessionStore
from .chroma_store import get_collection
from .chroma_store import get_collection_no_embed

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

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "Chroma_db"))).expanduser().resolve()
CHROMA_NEWS_DIR = Path(os.getenv("CHROMA_NEWS_DIR", str(CHROMA_DIR / "News_chroma_db"))).expanduser().resolve()

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
# (NEW) TFT Model Directory
# ----------------------------
TFT_PATH = Path(
    os.getenv("TFT_RESULT_DIR", str(PROJECT_ROOT / "tft"))
)
ANALYSIS_FILE_PATH = TFT_PATH / "result" / "inference_results.json"

# ----------------------------
# Redis Session Store (lazy init)
# ----------------------------
_store: Optional[RedisSessionStore] = None


def get_store() -> RedisSessionStore:
    global _store
    if _store is None:
        _store = RedisSessionStore()
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
    print("NEWS API using persist_dir=", CHROMA_NEWS_DIR, "collection=", CHROMA_COLLECTION)

    client, col = get_collection_no_embed(
        persist_dir=CHROMA_NEWS_DIR,
        collection_name=CHROMA_COLLECTION,
    )

    try:
        got = col.get(include=["metadatas", "documents"], limit=5000)
    except TypeError:
        got = col.get(include=["metadatas", "documents"])

    metadatas = got.get("metadatas") or []
    documents = got.get("documents") or []

    by_link: Dict[str, Dict[str, Any]] = {}

    for meta, doc in zip(metadatas, documents):
        if not meta:
            continue
        link = (meta.get("link") or "").strip()
        if not link:
            continue

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

        chunk_index = meta.get("chunk_index")
        try:
            chunk_index = int(chunk_index) if chunk_index is not None else 10**9
        except Exception:
            chunk_index = 10**9

        if chunk_index < item["best_chunk_index"]:
            item["best_chunk_index"] = chunk_index
            item["preview_doc"] = doc or ""

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
    market: str | None = None  # 예: NASDAQ
    # price: float | None = None # 예시
    prev_close: int | None = None
    current_price: int | None = None
    predicted_price: int | None = None
    change_pct: float | None = None # 예시
    headline: str               # 카드에 보이는 한 줄 추천 문구
    why: str                    # hover 시 노출되는 상세 이유
    risk: str | None = None  # 리스크 한줄(선택)

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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    state: AgentState = {
        "query": request.message,
        "category": "",
        "sub_category": "",
        "debate_history": [],
        "debate_count": 0,
        "response": "",
    }

    try:
        result = agent_app.invoke(state)
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


# 새 채팅방(세션) 만들기
@app.post("/session")
def create_session():
    session_id = str(uuid4())
    store = get_store()
    store.create_session(session_id)
    return {"session_id": session_id}


# 세션 삭제(프론트 X 버튼에서 호출)
@app.delete("/session/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(session_id: str):
    store = get_store()
    deleted = store.delete_session(session_id)
    store.delete_session(session_id)  # 없어도 그냥 패스
    return {"ok": True, "session_id": session_id}


@app.post("/chat/{session_id}", response_model=SessionChatResponse)
def chat_with_session(session_id: str, request: ChatRequest):
    store = get_store()

    history = store.get_last_n(session_id, n=3, chronological=True)

    if history:
        formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        full_query = f"[이전 대화 내역]\n{formatted_history}\n\n[현재 질문]\n{request.message}"
    else:
        full_query = request.message

    store.add_message(session_id, "user", request.message)

    state: AgentState = {
        "query": full_query,
        "category": "",
        "sub_category": "",
        "debate_history": history,
        "debate_count": 0,
        "response": "",
    }

    try:
        result = agent_app.invoke(state)

        answer = result.get("response", "응답을 생성할 수 없습니다.")
        category = result.get("category", "unknown")
        sub_category = result.get("sub_category", "")

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

# ------------------------------
# (NEW) TFT Helper Functions
# ------------------------------

def load_tft_result(file_path: Path) -> list[StockRecOut]:
    """
    JSON 파일을 읽어서 StockRecOut Object의 list로 반환
    """
    print(file_path)
    if not file_path.exists():
        print("no file exists")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            print(f"path: {file_path}")
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] Error decoding JSON from {file_path}")
        return None

def stock_recommendation() -> list[StockRecOut]:
    """
    load_tft_result()를 호출하여 추천 종목 리스트 반환
    """
    data = load_tft_result(ANALYSIS_FILE_PATH)

    if not data:
        return []
    
    raw_recs = data.get("recommendations", {})
    raw_results = data.get("results", [])

    results_by_code = {}
    for item in raw_results:
        code = (item.get("code") or "").strip()
        if code:
            results_by_code[code] = item
    
    price_map = {
        item.get("code"): item.get("base_close", 0.0) for item in raw_results
    }

    processed_list = []

    for strategy_key, rec_data in raw_recs.items():
        # code = rec_data.get("code")
        code = (rec_data.get("code") or "").strip()
        if not code:
            continue

        # 1. Headline 생성
        headline_text = strategy_key.replace("_", " ").title()

        # 2. Risk Value
        risk_val = rec_data.get("risk_spread")
        if risk_val is not None:
            try:
                risk_text = f"변동성 지표: {risk_val:.2f}"
            except (TypeError, ValueError):
                risk_text = "시장 변동성에 유의 필요"
        else:
            risk_text = "시장 변동성에 유의 필요"
        # risk_text = f"변동성 지표: {risk_val:.2f}" if risk_val else "시장 변동성에 유의 필요"

        result = results_by_code.get(code) or {}

        # 3. Interpretability
        top_vars = result.get("top_variables") or []
        var_parts = []

        for v in top_vars[:3]:
            name = v.get("name")
            weight = v.get("weight")
            if not name or weight is None:
                continue
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            var_parts.append(f"{name}({w:.3f})")
        
        if var_parts:
            why_text = "주요 영향 변수: " + ", ".join(var_parts)
        else:
            metric = rec_data.get("metric")
            why_text = f"Metric: {metric}" if metric else "모델 추론 근거 요약 정보가 부족합니다."

        # 4. Prices
        base_close = result.get("base_close", 0.0) or 0.0
        prev_close: int | None

        if base_close is None:
            prev_close = None
        else:
            try:
                prev_close = int(round(float(base_close)))
            except (TypeError, ValueError):
                prev_close = None
        
        predicted_price: int | None = None
        forecasts = result.get("forecasts") or []
        if forecasts:
            first = forecasts[0] or {}
            p = first.get("price")
            if p is not None:
                try:
                    predicted_price = int(round(float(p)))
                except (TypeError, ValueError):
                    predicted_price = None

        name = (rec_data.get("name") or result.get("name") or code)

        stock = StockRecOut(
            symbol=code,
            name=name,
            market="KRX",
            # price=float(base_close) if base_close is not None else 0.0,
            prev_close=prev_close,
            current_price=None,
            predicted_price=predicted_price,
            change_pct=rec_data.get("expected_return", 0.0),
            headline=headline_text,
            why=why_text,
            risk=risk_text
        )
        processed_list.append(stock)
    
    print(processed_list)

    return processed_list

@app.get("/stocks/recommendations", response_model=StockRecListOut)
def get_stock_recommendations(limit: int = Query(2, ge=1, le=10)):
    rec_items = stock_recommendation()
    return {"items": rec_items[:limit]}
