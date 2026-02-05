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

# TFT
from tft.tft_feature_map import FEATURE_META, STRATEGY_TEXT_MAP

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


class ForecastOut(BaseModel):
    horizon_days: int
    dates: list[str]
    q10: list[int]
    q50: list[int]
    q90: list[int]

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
    forecast: ForecastOut | None = None

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

def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None
    
def _extract_forecast(result: dict[str, Any], horizon_days: int = 3) -> ForecastOut | None:
    raw = result.get("forecasts") or []
    if len(raw) < horizon_days:
        return None

    dates: list[str] = []
    q10: list[int] = []
    q50: list[int] = []
    q90: list[int] = []

    for row in raw[:horizon_days]:
        d = str(row.get("date") or "").strip()
        lo = _to_int(row.get("price_lower"))
        med = _to_int(row.get("price"))
        hi = _to_int(row.get("price_upper"))

        if not d or lo is None or med is None or hi is None:
            return None

        dates.append(d)
        q10.append(lo)
        q50.append(med)
        q90.append(hi)

    return ForecastOut(
        horizon_days=horizon_days,
        dates=dates,
        q10=q10,
        q50=q50,
        q90=q90,
    )

def _build_why_text(
        top_vars: list[dict],
        strategy_key: str,
        metric: str | None = None,
        top_k: int = 3
) -> str:
    """
    top_variables 바탕으로 추천 이유 문구 생성
    """
    parsed = []
    for v in top_vars or []:
        name = (v.get("name") or "").strip()
        weight = v.get("weight")
        if not name or weight is None:
            continue
        parsed.append((name, weight))
    
    selection_sentence = STRATEGY_TEXT_MAP.get(strategy_key)
    if not selection_sentence:
        selection_sentence = metric or "모델 예측 점수를 바탕으로 선정된 종목입니다."

    if not parsed:
        return (
            f"선정 이유: {selection_sentence}\n"
            "변수 중요도 정보가 충분하지 않아, 상세 영향 변수는 제공되지 않았습니다."
        )
    
    parsed.sort(key=lambda x: abs(x[1]), reverse=True)
    topk = parsed[:top_k]
    denom = sum(abs(w) for _, w in topk) or 1.0

    lines = []
    lines.append(f"선정 이유: {selection_sentence}\n")
    lines.append("예측에 크게 반영된 변수(상위)")
    for raw_name, w in topk:
        meta = FEATURE_META.get(raw_name, {})
        label = meta.get("label", raw_name)
        desc = meta.get("desc", "모델 예측에 반영된 입력 변수")
        share = abs(w) / denom * 100.0
        lines.append(f"- {label} ({share:.1f}%): {desc}")
    # lines.append("※ 중요도는 예측에 대한 상대적 기여도이며, 가격 방향의 인과를 직접 의미하지 않습니다.")

    return "\n".join(lines)

def _build_risk_text(
    prev_close: int | None,
    forecast_obj,   # ForecastOut | None
    risk_spread_raw,
) -> str:
    """
    리스크를 '예측구간 폭 + 하방 시나리오'로 설명
    """
    # forecast 기반 계산 우선
    try:
        if (
            forecast_obj is not None
            and forecast_obj.q10
            and forecast_obj.q50
            and forecast_obj.q90
            and len(forecast_obj.q10) >= 3
            and len(forecast_obj.q50) >= 3
            and len(forecast_obj.q90) >= 3
        ):
            q10_d3 = float(forecast_obj.q10[2])
            q50_d3 = float(forecast_obj.q50[2])
            q90_d3 = float(forecast_obj.q90[2])

            if q50_d3 > 0:
                band_pct = (q90_d3 - q10_d3) / q50_d3 * 100.0
            else:
                band_pct = None

            downside_pct = None
            if prev_close and prev_close > 0:
                downside_pct = (q10_d3 / float(prev_close) - 1.0) * 100.0

            if band_pct is None:
                grade = "판단 어려움"
            elif band_pct < 4:
                grade = "낮은 편"
            elif band_pct < 8:
                grade = "보통"
            else:
                grade = "큰 편"
            
            parts = []
            parts.append(
                f"D+3 예상 가격 범위는 약 {int(round(q10_d3)):,}원 ~ {int(round(q90_d3)):,}원입니다."
            )
            parts.append(
                f"중앙값은 {int(round(q50_d3)):,}원이며, 예측 폭은 "
                f"{band_pct:.2f}%로 변동성은 {grade}입니다." if band_pct is not None
                else "예측 폭 정보를 계산하기 어려워 변동성 판단이 제한적입니다."
            )

            if downside_pct is not None:
                if downside_pct < 0:
                    parts.append(
                        f"보수적으로 보면 전일 종가 대비 최대 {abs(downside_pct):.2f}% "
                        "하락 가능성도 염두에 두는 것이 좋습니다."
                    )
                else:
                    parts.append(
                        "보수적 시나리오(q10)에서도 전일 종가 대비 상승으로 예측됩니다."
                    )
            return " ".join(parts)
    except Exception:
        pass

    # fallback: 기존 risk_spread 사용
    try:
        if risk_spread_raw is not None:
            rs = float(risk_spread_raw)
            if rs < 2:
                grade = "낮은 편"
            elif rs < 5:
                grade = "보통"
            else:
                grade = "큰 편"
            return (
                f"내부 리스크 지표는 {rs:.2f}%이며, 변동성은 {grade}입니다. "
                "값이 클수록 예측 범위가 넓어 실제 결과의 흔들림이 커질 수 있습니다."
            )
    except (TypeError, ValueError):
        pass

    return "예측구간 기반 리스크 정보가 부족합니다."


def _build_headline(strategy_key: str, expected_return_raw: Any) -> str:
    strategy = strategy_key.replace("_", " ").title()
    er = expected_return_raw
    if er is None:
        return strategy
    return f"{strategy} · 3일 기대수익 {er:+.2f}%"

def load_tft_result(file_path: Path) -> dict[str, Any] | None:
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

        # 3. Prices
        base_close = result.get("base_close", 0.0) or 0.0
        prev_close: int | None

        if base_close is None:
            prev_close = None
        else:
            try:
                prev_close = int(round(float(base_close)))
            except (TypeError, ValueError):
                prev_close = None

        # 5. Horizon Days
        horizon_days = int(
            rec_data.get("horizon_days")
            or data.get("horizon_days")
            or 3
        )
        predicted_price: int | None = None
        forecast_obj = _extract_forecast(result, horizon_days=horizon_days)
        predicted_price = forecast_obj.q50[0] if forecast_obj and forecast_obj.q50 else None

        name = (rec_data.get("name") or result.get("name") or code)

        # 4. Interpretability
        top_vars = result.get("top_variables") or []

        # 4-1. why_text
        metric = rec_data.get("metric")
        why_text = _build_why_text(
            top_vars=top_vars,
            strategy_key=strategy_key,
            metric=metric,
            top_k=3
        )

        # 4-2. risk_text
        risk_text = _build_risk_text(
            prev_close=prev_close,
            forecast_obj=forecast_obj,
            risk_spread_raw=rec_data.get("risk_spread")
        )        

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
            risk=risk_text,
            forecast=forecast_obj
        )
        processed_list.append(stock)
    
    print(processed_list)

    return processed_list

@app.get("/stocks/recommendations", response_model=StockRecListOut)
def get_stock_recommendations(limit: int = Query(2, ge=1, le=10)):
    rec_items = stock_recommendation()
    return {"items": rec_items[:limit]}
