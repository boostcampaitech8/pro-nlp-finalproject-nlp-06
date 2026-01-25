from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 절대경로 import
try:
    from .model_agent import agent
    print("[SUCCESS] Agent import 성공!")
except Exception as e:
    print(f"[ERROR] Agent import 실패: {e}")
    raise

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
# Request / Response
# ----------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    category: str
    sub_category: str = ""

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    """서버 상태 및 설정 확인"""
    return {
        "message": "Backend server is running with FinancialAgent!",
        "status": "ok",
        "config": {
            "vllm_url": agent.vllm_base_url,
            "model": agent.vllm_model,
            "embedding": agent.embedding_model,
            "embedding_device": agent.embedding_device,
            "chroma_dir": str(agent.chroma_base_dir),
            "k_values": agent.k_values,
        }
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """채팅 엔드포인트"""
    try:
        print(f"\n[REQUEST] User: {request.message}")
        
        # Agent 실행
        result = agent.invoke(request.message)
        
        print(f"[RESPONSE] Category: {result.get('category')}")
        print(f"[RESPONSE] Answer: {result.get('response')[:100]}...")
        
        return ChatResponse(
            answer=result.get("response", "응답을 생성할 수 없습니다."),
            category=result.get("category", "unknown"),
            sub_category=result.get("sub_category", "")
        )
    
    except Exception as e:
        print(f"\n[ERROR] Agent 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            answer=f"오류가 발생했습니다: {str(e)}",
            category="error",
            sub_category=""
        )