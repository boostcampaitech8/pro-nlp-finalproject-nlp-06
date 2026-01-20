from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_ollama import RagNewsChatService

app = FastAPI()

# CORS 설정
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

# 파이프라인과 동일하게 맞추기
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_news")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "naver_finance_news_chunks")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# RAG 서비스 인스턴스 생성 (서버 시작 시 1회 로딩)
rag_service = RagNewsChatService(
    collection_name=CHROMA_COLLECTION,    
    persist_directory=CHROMA_DIR,
    llm_model=OLLAMA_LLM_MODEL,
    embedding_model=OLLAMA_EMBED_MODEL,
    ollama_base_url=OLLAMA_BASE_URL,
)

# 요청 형식
class ChatRequest(BaseModel):
    message: str

# 응답 형식
class ChatResponse(BaseModel):
    answer: str
    used_db: bool

@app.get("/")
def root():
    return {"message": "Backend server is running!"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer, used_db = rag_service.answer(request.message)
    return ChatResponse(answer=answer, used_db=used_db)
