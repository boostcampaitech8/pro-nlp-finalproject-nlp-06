from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_ollama import RagNewsChatService  


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
# 경로/환경변수 통일 (pipeline/DAG와 동일)
# ----------------------------
# Airflow가 어디서 돌든, FastAPI가 어디서 돌든
# "항상 같은 Chroma 폴더"를 보게 만드는 게 핵심
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/data/ephemeral/home/project")

CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(PROJECT_ROOT, "chroma_news"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "naver_finance_news_chunks")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# (선택) Retrieval 튜닝 env로 조절 가능하게
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "48"))
TOP_ARTICLES = int(os.getenv("TOP_ARTICLES", "5"))
MAX_CHUNKS_PER_ARTICLE = int(os.getenv("MAX_CHUNKS_PER_ARTICLE", "3"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.7"))
MIN_DOCS_AFTER_FILTER = int(os.getenv("MIN_DOCS_AFTER_FILTER", "12"))
ENABLE_QUERY_REFINE = os.getenv("ENABLE_QUERY_REFINE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ----------------------------
# RAG 서비스 인스턴스 (서버 시작 시 1회 로딩)
# ----------------------------
rag_service = RagNewsChatService(
    collection_name=CHROMA_COLLECTION,
    persist_directory=CHROMA_DIR,
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


# ----------------------------
# Request/Response Models
# ----------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    used_db: bool


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "Backend server is running!",
        "chroma_dir": CHROMA_DIR,
        "chroma_collection": CHROMA_COLLECTION,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_llm_model": OLLAMA_LLM_MODEL,
        "ollama_embed_model": OLLAMA_EMBED_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer, used_db = rag_service.answer(request.message)
    return ChatResponse(answer=answer, used_db=used_db)
