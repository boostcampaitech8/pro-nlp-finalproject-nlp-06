from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Agent.py에서 필요한 함수와 그래프를 import
from .Agent import app as agent_app, AgentState

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

# 요청 형식
class ChatRequest(BaseModel):
    message: str

# 응답 형식
class ChatResponse(BaseModel):
    answer: str
    category: str
    sub_category: str = ""

@app.get("/")
def root():
    return {"message": "Backend server is running with Agent!"}

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