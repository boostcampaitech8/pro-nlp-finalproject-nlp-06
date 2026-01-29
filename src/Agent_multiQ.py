import operator
import os
import ast
from typing import Annotated, List, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# 1. 상태 정의 (State)
class AgentState(TypedDict):
    query: str
    categories: List[str]      # 다중 선택을 위한 리스트
    sub_categories: List[str]  # 다중 선택을 위한 리스트
    # 각 노드의 결과물을 통합하기 위한 리듀서 (리스트 합치기)
    results: Annotated[List[str], operator.add] 
    debate_history: Annotated[List[str], operator.add]
    debate_count: int
    response: str

# LLM 설정
llm = ChatOpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
    model=os.getenv("VLLM_MODEL", "skt/A.X-4.0-Light"),
    api_key=os.getenv("VLLM_API_KEY", "vllm-key")
)

# --- ChromaDB 및 임베딩 설정 (기존 코드 유지) ---
_vectorstore_cache = {}
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'device': 'cuda', 'batch_size': 32}
        )
    return _embeddings

def get_vectorstore(db_name: str):
    if db_name not in _vectorstore_cache:
        embeddings = get_embeddings()
        db_path = f"./Chroma_db/{db_name}"
        _vectorstore_cache[db_name] = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return _vectorstore_cache[db_name]

def search_db(db_name: str, query: str, k: int = 3):
    try:
        vectorstore = get_vectorstore(db_name)
        results = vectorstore.similarity_search(query, k=k)
        formatted_context = []
        for doc in results:
            if "Vocab" in db_name:
                formatted_context.append(f"용어: {doc.page_content}\n설명: {doc.metadata.get('description', '')}")
            else:
                formatted_context.append(f"제목: {doc.metadata.get('title', '')}\n내용: {doc.page_content}")
        return formatted_context
    except Exception as e:
        print(f"[ERROR] DB 검색 실패: {e}")
        return []

# --- 노드 정의 ---

def main_router(state: AgentState):
    print("\n[ROUTER] main_router 실행 중 (다중 선택 가능)...")
    prompt = f"""
질문: {state['query']}
아래 카테고리 중 질문과 관련된 모든 항목을 리스트 형식으로 골라주세요. 
(예: ['news', 'report'], ['vocab'], ['news', 'prediction'])
- vocab (경제/통계 용어)
- report (산업, 경제, 종목, 시황 리포트)
- news (뉴스 분석)
- prediction (주가 예측)
- chat (일반 대화)
Assistant (리스트 형식으로만 출력):"""
    res = llm.invoke(prompt).content.strip()
    try:
        categories = ast.literal_eval(res)
    except:
        categories = ["chat"]
    return {"categories": categories}

def vocab_node(state: AgentState):
    print("\n[NODE] vocab_node 실행...")
    context = search_db("Vocab_chroma_db", state['query'], k=1)
    res = llm.invoke(f"문맥: {context}\n질문: {state['query']} 설명.").content
    return {"results": [f"[용어 사전 결과]\n{res}"]}

def news_node(state: AgentState):
    print("\n[NODE] news_node 실행...")
    context = search_db("News_chroma_db", state['query'], k=3)
    res = llm.invoke(f"뉴스: {context}\n질문: {state['query']} 분석. 출처/날짜 포함.").content
    return {"results": [f"[뉴스 분석 결과]\n{res}"]}

def report_router_node(state: AgentState):
    print("\n[ROUTER] report_router_node 실행 (다중 선택)...")
    prompt = f"""
질문: {state['query']}
아래 카테고리 중 질문과 관련된 모든 항목을 리스트 형식으로 골라주세요. 
(예: ['news', 'report'], ['vocab'], ['news', 'prediction'])
- stock (종목, 또는 회사, 회사 리포트, 종목 리포트)
- industry (산업, 특정 산업 동향, 산업 리포트)
- market (시황, 현재 시장 상황, 시황 리포트)
- economy (경제, 현재 경제 상황, 경제 리포트)
출력은 반드시 리스트 형태로 출력하세요.
Assistant:
"""
    res = llm.invoke(prompt).content.strip()
    try:
        sub_categories = ast.literal_eval(res)
    except:
        sub_categories = ["stock"]
    return {"sub_categories": sub_categories}

# 각 리포트 노드들 (결과를 results 리스트에 추가)
def stock_report_node(state: AgentState):
    context = search_db("Company_report_chroma_db", state['query'], k=3)
    res = llm.invoke(f"종목 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[종목 리포트 분석]\n{res}"]}

def industry_report_node(state: AgentState):
    context = search_db("Industry_report_chroma_db", state['query'], k=3)
    res = llm.invoke(f"산업 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[산업 리포트 분석]\n{res}"]}

def market_report_node(state: AgentState):
    context = search_db("MarketConditions_report_chroma_db", state['query'], k=3)
    res = llm.invoke(f"시황 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[시황 리포트 분석]\n{res}"]}

def economy_report_node(state: AgentState):
    context = search_db("Economy_report_chroma_db", state['query'], k=3)
    res = llm.invoke(f"경제 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[경제 리포트 분석]\n{res}"]}

# 예측 에이전트 (단기/장기 출력 반영)
def short_term_agent(state: AgentState):
    print("\n[AGENT] short_term_agent 실행 중...")
    context = search_db("News_chroma_db", state['query'], 3)
    res = llm.invoke(f"단기 예측 전문가로서 분석하세요. 질문: {state['query']}\n정보: {context}").content
    print(f"  > 단기 전망: {res[:50]}...") # 콘솔 출력
    return {"debate_history": [f"[단기 전망]\n{res}"]}

def long_term_agent(state: AgentState):
    print("\n[AGENT] long_term_agent 실행 중...")
    context = search_db("Industry_report_chroma_db", state['query'], 1)
    res = llm.invoke(f"장기 예측 전문가로서 분석하세요. 질문: {state['query']}\n정보: {context}").content
    print(f"  > 장기 전망: {res[:50]}...") # 콘솔 출력
    return {"debate_history": [f"[장기 전망]\n{res}"], "debate_count": state["debate_count"] + 1}

def finalize_prediction(state: AgentState):
    print("\n[NODE] finalize_prediction 실행...")
    history = "\n\n".join(state["debate_history"])
    # 최종 결과물 리스트에 추가
    return {"results": [f"[주가 예측 토론 합계]\n{history}"]}

def chat_node(state: AgentState):
    res = llm.invoke(state['query']).content
    return {"results": [f"[일반 답변]\n{res}"]}

# [추가] 최종 답변 통합 노드 (Fan-in)
def final_aggregator(state: AgentState):
    print("\n[NODE] final_aggregator 실행 (결과 통합)...")
    combined_context = "\n\n".join(state["results"])
    prompt = f"""
질문: {state['query']}
아래의 여러 분석 결과들을 바탕으로 사용자를 위한 최종 답변을 친절하고 구조적으로 작성해주세요.
모든 출처와 날짜 정보를 포함시켜야 합니다.

분석 결과들:
{combined_context}
"""
    final_res = llm.invoke(prompt).content
    return {"response": final_res}

# --- 그래프 구성 ---

workflow = StateGraph(AgentState)

workflow.add_node("main_router", main_router)
workflow.add_node("vocab", vocab_node)
workflow.add_node("news", news_node)
workflow.add_node("report_router_node", report_router_node)
workflow.add_node("stock_report", stock_report_node)
workflow.add_node("industry_report", industry_report_node)
workflow.add_node("market_report", market_report_node)
workflow.add_node("economy_report", economy_report_node)
workflow.add_node("short_term_agent", short_term_agent)
workflow.add_node("long_term_agent", long_term_agent)
workflow.add_node("finalize_prediction", finalize_prediction)
workflow.add_node("chat", chat_node)
workflow.add_node("final_aggregator", final_aggregator)

workflow.add_edge(START, "main_router")

# --- 다중 라우팅 로직 (Fan-out) ---

def main_routing_logic(state: AgentState):
    return state["categories"]

workflow.add_conditional_edges(
    "main_router",
    main_routing_logic,
    {
        "vocab": "vocab",
        "news": "news",
        "report": "report_router_node",
        "prediction": "short_term_agent",
        "chat": "chat"
    }
)

def report_routing_logic(state: AgentState):
    return state["sub_categories"]

workflow.add_conditional_edges(
    "report_router_node",
    report_routing_logic,
    {
        "stock": "stock_report",
        "industry": "industry_report",
        "market": "market_report",
        "economy": "economy_report"
    }
)

# 예측 루프
def debate_routing_logic(state: AgentState):
    if state["debate_count"] >= 1: # 1회 토론 후 종료 (단기/장기 각각 1번씩)
        return "finalize_prediction"
    return "short_term_agent"

workflow.add_edge("short_term_agent", "long_term_agent")
workflow.add_conditional_edges("long_term_agent", debate_routing_logic)

# 모든 노드의 끝을 final_aggregator로 연결 (Fan-in)
workflow.add_edge("vocab", "final_aggregator")
workflow.add_edge("news", "final_aggregator")
workflow.add_edge("stock_report", "final_aggregator")
workflow.add_edge("industry_report", "final_aggregator")
workflow.add_edge("market_report", "final_aggregator")
workflow.add_edge("economy_report", "final_aggregator")
workflow.add_edge("finalize_prediction", "final_aggregator")
workflow.add_edge("chat", "final_aggregator")

workflow.add_edge("final_aggregator", END)

app = workflow.compile()

# --- 실행부 (기존 유지) ---
def run_chatbot():
    print("=" * 50)
    print("금융 에이전트 챗봇 (다중 분석 모드)")
    print("=" * 50)
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]: break
        state = {
            "query": f"[{datetime.now().strftime('%Y-%m-%d')}] {user_input}",
            "categories": [], "sub_categories": [], "results": [],
            "debate_history": [], "debate_count": 0, "response": "",
        }
        try:
            result = app.invoke(state)
            print("\n" + "=" * 50 + "\nAssistant:\n" + result.get("response", "") + "\n" + "=" * 50)
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    run_chatbot()