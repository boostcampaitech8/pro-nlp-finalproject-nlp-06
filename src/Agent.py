# --- 그래프 구성 ---
import operator
from typing import Annotated, List, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_naver import ChatClovaX
from datetime import datetime

from dotenv import load_dotenv

load_dotenv() # .env 파일 로드


# 1. 상태 정의 (State)
class AgentState(TypedDict):
    query: str
    category: str
    sub_category: str
    debate_history: Annotated[List[str], operator.add]
    debate_count: int
    response: str


# LLM 설정 (vLLM 서빙 모델 연동)
router_llm = ChatOpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
    model=os.getenv("VLLM_MODEL", "skt/A.X-4.0-Light"),
    api_key=os.getenv("VLLM_API_KEY", "vllm-key")
)

CLOVA_STUDIO_API_KEY = os.getenv("CLOVA_STUDIO_API_KEY")
answer_llm = ChatClovaX(
    model="HCX-007",
    api_key=CLOVA_STUDIO_API_KEY,
    max_tokens= 16384
)


# --- ChromaDB 전역 캐싱 (메모리 절약) ---

_vectorstore_cache = {}
_embeddings = None  # 임베딩 모델도 한 번만 로드

def get_embeddings():
    """임베딩 모델을 GPU에서 한 번만 로드"""
    global _embeddings
    if _embeddings is None:
        print(f"[INFO] 임베딩 모델 로드 중 (GPU 사용)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda'},  # GPU 사용
            encode_kwargs={'device': 'cuda', 'batch_size': 32}
        )
    return _embeddings

def get_vectorstore(db_name: str, collection_name: str = None):
    """ChromaDB 인스턴스를 캐싱하여 재사용 (뉴스 DB 예외 처리 포함)"""
    
    # 1. 뉴스 DB인 경우 컬렉션 이름을 자동으로 설정
    if collection_name is None:
        if db_name == "News_chroma_db":
            collection_name = "naver_finance_news_chunks"
        else:
            collection_name = "langchain"

    # 2. 캐시 키를 (DB이름, 컬렉션이름) 조합으로 만들어 충돌 방지
    cache_key = (db_name, collection_name)

    if cache_key not in _vectorstore_cache:
        print(f"[INFO] ChromaDB 로드 중: {db_name} (Collection: {collection_name})")
        
        embeddings = get_embeddings()
        
        # [주의] 아까 우리를 괴롭혔던 경로! 절대 경로로 하는 것이 가장 안전합니다.
        project_root = "/data/ephemeral/home/pro-nlp-finalproject-nlp-06"
        db_path = os.path.join(project_root, "Chroma_db", db_name)

        _vectorstore_cache[cache_key] = Chroma(
            persist_directory=db_path, 
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
    return _vectorstore_cache[cache_key]


# --- 검색 함수 (개선) ---
def search_db(db_name: str, query: str, k: int = 3):
    """캐싱된 vectorstore를 사용하여 검색"""
    try:
        vectorstore = get_vectorstore(db_name)
        results = vectorstore.similarity_search(query, k=k)
        
        # 검색 성공 시 결과 출력
        if results:
            print(f"\n[SEARCH SUCCESS] DB: {db_name}, 검색어: '{query}', 결과 개수: {len(results)}")
            print("-" * 80)
            
            for idx, doc in enumerate(results, 1):  # 검색된 개수만큼만
                if "Vocab" in db_name:
                    term = doc.page_content
                    description = doc.metadata.get("description", "설명 정보 없음")
                    print(f"[{idx}] 용어: {term}")
                    print(f"    설명: {description[:100]}")  # 100자까지만
                else:
                    title = doc.metadata.get("title", "제목 없음")
                    content = doc.page_content
                    print(f"[{idx}] 제목: {title}")
                    print(f"    내용: {content[:100]}")  # 100자까지만
                print()
            
            print("-" * 80)
        
        # 포맷팅된 컨텍스트 생성
        formatted_context = []
        for doc in results:
            if "Vocab" in db_name:
                term = doc.page_content
                description = doc.metadata.get("description", "설명 정보 없음")
                formatted_context.append(f"용어: {term}\n설명: {description}")
            else:
                title = doc.metadata.get("title", "제목 없음")
                content = doc.page_content
                formatted_context.append(f"제목: {title}\n내용: {content}")
        
        return formatted_context
    
    except Exception as e:
        print(f"[ERROR] DB 검색 실패 ({db_name}): {e}")
        return []


# --- 노드 정의 ---

def main_router(state: AgentState):
    print("\n[ROUTER] main_router 실행 중...")
    prompt = f"""
질문: {state['query']}
아래 중 하나만 정확히 선택:
- vocab (경제/통계 용어 질문)
- report (산업, 경제, 종목(기업), 시황에 대한 질문 또는 리포트 분석)
- news (뉴스 분석, 최근 뉴스 질문. 최근 상황에 대한 질문, 리포트 보다는 뉴스를 참조하면 더 정확한 정보를 찾을 수 있을 때)
- prediction (기업 주가 예측)
- chat (일반 대화, 인사말, 농담, 가벼운 대화 등)
출력은 vocab, report, news, prediction, chat 5가지 중 하나로만 반드시 출력해.
Assistant:
"""
    category = router_llm.invoke(prompt).content.strip().lower()
    print(f"[ROUTER] 선택된 카테고리: {category}")
    return {"category": category}


def vocab_node(state: AgentState):
    print("\n[NODE] vocab_node 실행 중...")
    context = search_db("Vocab_chroma_db", state['query'], k=1)
    res = answer_llm.invoke(f"문맥: {context}\n질문: {state['query']}에 대해 설명해줘.").content
    return {"response": res}


def news_node(state: AgentState):
    print("\n[NODE] news_node 실행 중...")
    context = search_db("News_chroma_db", state['query'], k=10)
    res = answer_llm.invoke(f"뉴스: {context}\n질문: {state['query']} 뉴스를 보고 질문에 대해 답변해줘.").content
    return {"response": res}


def report_router_node(state: AgentState):
    print("\n[ROUTER] report_router_node 실행 중...")
    prompt = f"""
질문: {state['query']}
질문에 답하기 위해 제공할 수 있는 정보는 다음과 같다.
- stock (종목, 또는 회사)
- industry (산업, 특정 산업 동향)
- market (시황, 현재 시장 상황)
- economy (경제, 현재 경제 상황)
출력은 stock, industry, market, economy 4가지 중 하나로만 반드시 출력해.
Assistant:
"""
    sub_category = router_llm.invoke(prompt).content.strip().lower()
    print(f"[ROUTER] 선택된 서브카테고리: {sub_category}")
    return {"sub_category": sub_category}


def stock_report_node(state: AgentState):
    print("\n[NODE] stock_report_node 실행 중...")
    context = search_db("Company_report_chroma_db", state['query'], k=3)
    res = answer_llm.invoke(f"종목(회사) 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘").content
    return {"response": res}


def industry_report_node(state: AgentState):
    print("\n[NODE] industry_report_node 실행 중...")
    context = search_db("Industry_report_chroma_db", state['query'], k=3)
    res = answer_llm.invoke(f"산업 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
    return {"response": res}


def market_report_node(state: AgentState):
    print("\n[NODE] market_report_node 실행 중...")
    context = search_db("MarketConditions_report_chroma_db", state['query'], k=3)
    res = answer_llm.invoke(f"시황 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
    return {"response": res}


def economy_report_node(state: AgentState):
    print("\n[NODE] economy_report_node 실행 중...")
    context = search_db("Economy_report_chroma_db", state['query'], k=3)
    res = answer_llm.invoke(f"경제 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
    return {"response": res}


def short_term_agent(state: AgentState):
    print("\n[AGENT] short_term_agent 실행 중...")
    context = (
        search_db("MarketConditions_report_chroma_db", state['query'], 1) +
        search_db("Company_report_chroma_db", state['query'], 1) +
        search_db("News_chroma_db", state['query'], 1)
    )
    history = "\n".join(state["debate_history"])
    res = router_llm.invoke(
        f"당신은 단기 주식 예측 전문가입니다. 질문에 대해 생각한 뒤 단기 전망을 제시하세요. \n질문: {state['query']}\n문맥: {context}\n이전 토론:\n{history}\n단기 전망 제시"
    ).content
    print(res)
    return {
        "debate_history": [f"단기: {res}"],
    }


def long_term_agent(state: AgentState):
    print("\n[AGENT] long_term_agent 실행 중...")
    context = (
        search_db("Industry_report_chroma_db", state['query'], 1) +
        search_db("Economy_report_chroma_db", state['query'], 1)
    )
    history = "\n".join(state["debate_history"])
    res = router_llm.invoke(
        f"당신은 장기 주식 예측 전문가입니다. 질문에 대해 생각한 뒤 장기 전망을 제시하세요.\n질문: {state['query']}\n문맥: {context}\n이전 토론:\n{history}\n장기 관점 제시."
    ).content
    print(res)
    return {
        "debate_history": [f"장기: {res}"],
        "debate_count": state["debate_count"] + 1
    }


def finalize_prediction(state: AgentState):
    print("\n[NODE] finalize_prediction 실행 중...")
    history = "\n".join(state["debate_history"])
    res = answer_llm.invoke(f"당신은 주식 예측 전문가입니다. 단기예측 전문가와 장기 예측 전문가의 토론을 보고 최종 결론을 도출하세요.:\n{history}").content
    return {"response": res}


def chat_node(state: AgentState):
    print("\n[NODE] chat_node 실행 중...")
    res = answer_llm.invoke(state['query']).content
    return {"response": res}


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

# START -> main_router (고정)
workflow.add_edge(START, "main_router")


# --- 라우팅 로직 ---

def main_routing_logic(state: AgentState) -> Literal[
    "vocab", "news", "report_router_node", "short_term_agent", "chat"
]:
    cat = state["category"]
    if "vocab" in cat: 
        print(f"[ROUTING] main_router -> vocab")
        return "vocab"
    if "report" in cat: 
        print(f"[ROUTING] main_router -> report_router_node")
        return "report_router_node"
    if "news" in cat: 
        print(f"[ROUTING] main_router -> news")
        return "news"
    if "prediction" in cat: 
        print(f"[ROUTING] main_router -> short_term_agent")
        return "short_term_agent"
    print(f"[ROUTING] main_router -> chat (기본값)")
    return "chat"


# main_router에서만 분기
workflow.add_conditional_edges(
    "main_router",
    main_routing_logic,
    {
        "vocab": "vocab",
        "news": "news",
        "report_router_node": "report_router_node",
        "short_term_agent": "short_term_agent",
        "chat": "chat",
    }
)


def report_routing_logic(state: AgentState) -> Literal[
    "stock_report", "industry_report", "market_report", "economy_report"
]:
    sub = state["sub_category"]
    if "stock" in sub: 
        print(f"[ROUTING] report_router -> stock_report")
        return "stock_report"
    if "industry" in sub: 
        print(f"[ROUTING] report_router -> industry_report")
        return "industry_report"
    if "market" in sub: 
        print(f"[ROUTING] report_router -> market_report")
        return "market_report"
    print(f"[ROUTING] report_router -> economy_report (기본값)")
    return "economy_report"


workflow.add_conditional_edges(
    "report_router_node",
    report_routing_logic,
    {
        "stock_report": "stock_report",
        "industry_report": "industry_report",
        "market_report": "market_report",
        "economy_report": "economy_report",
    }
)


def debate_routing_logic(state: AgentState) -> Literal[
    "short_term_agent", "finalize_prediction"
]:
    if state["debate_count"] >= 2:
        print(f"[ROUTING] long_term_agent -> finalize_prediction (토론 횟수: {state['debate_count']})")
        return "finalize_prediction"
    print(f"[ROUTING] long_term_agent -> short_term_agent (토론 계속, 현재 횟수: {state['debate_count']})")
    return "short_term_agent"


workflow.add_edge("short_term_agent", "long_term_agent")
workflow.add_conditional_edges(
    "long_term_agent",
    debate_routing_logic,
    {
        "short_term_agent": "short_term_agent",
        "finalize_prediction": "finalize_prediction",
    }
)

# --- 종료 엣지 ---
workflow.add_edge("vocab", END)
workflow.add_edge("news", END)
workflow.add_edge("stock_report", END)
workflow.add_edge("industry_report", END)
workflow.add_edge("market_report", END)
workflow.add_edge("economy_report", END)
workflow.add_edge("finalize_prediction", END)
workflow.add_edge("chat", END)

app = workflow.compile()

def run_chatbot():
    print("=" * 50)
    print("금융 에이전트 챗봇 시작 (종료: exit)")
    print("=" * 50)

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\n종료합니다.")
            break

        today_str = datetime.now().strftime("%Y-%m-%d")
        query_with_date = f"[오늘 날짜 : {today_str}] {user_input}"
        
        print(f"[INFO] 날짜가 추가된 쿼리: {query_with_date}")  # ✅ 디버깅용 출력

        # 초기 상태
        state = {
            "query": user_input,
            "category": "",
            "sub_category": "",
            "debate_history": [],
            "debate_count": 0,
            "response": "",
        }

        # 그래프 실행
        try:
            result = app.invoke(state)
            print("\n" + "=" * 50)
            print("Assistant:")
            print(result.get("response", "응답을 생성할 수 없습니다."))
            print("=" * 50)
        except Exception as e:
            print(f"\n[ERROR] 그래프 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    run_chatbot()