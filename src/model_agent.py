# model_ollama.py
from __future__ import annotations

import operator
import os
from typing import Annotated, List, Literal, TypedDict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime  # 파일 상단에 이 임포트가 있는지 확인하세요!


load_dotenv()

def _kw_to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

class AgentState(TypedDict):
    query: str
    category: str
    sub_category: str
    debate_history: Annotated[List[str], operator.add]
    debate_count: int
    response: str

class FinancialAgent:
    """
    금융 챗봇 에이전트 (Langgraph)
    """

    """
    뉴스 Chroma DB + Ollama(로컬 LLM/임베딩)를 이용해 질문에 답변하는 RAG 서비스
    - DB는 chunk 단위로 저장되어 있음
    - 검색 결과(chunk)를 기사(link) 단위로 묶어서 컨텍스트를 구성
    - 관련 없는 뉴스가 섞이는 문제를 줄이기 위해:
      1) similarity_search_with_score + 임계치 필터
      2) link(기사) 단위로 best_score 기반 랭킹
      3) (선택) 질문을 검색용 키워드로 정제(refine_search_query)
    """

    def __init__(
        self,
        # 파이프라인과 맞추기 (중요)
        collection_name: str = "naver_finance_news_chunks",
        # Chroma db 경로 수정
        chroma_base_dir: str = "./Chroma_db",
        persist_directory: str = "./Chroma_db/News_chroma_db",
        # vllm 모델, 임베딩 모델 값
        vllm_model: str = "skt/A.X-4.0-Light",
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        embedding_device: str = "cuda",  # "cuda" 또는 "cpu"
        embedding_batch_size: int = 32,

        vllm_base_url: str = "http://127.0.0.1:8001/v1",
        vllm_api_key: str = None,
        # 뉴스 Retrieval 설정값(기존)
        retrieval_k: int = 48,
        top_articles: int = 5,
        max_chunks_per_article: int = 3,
        # Chroma 점수는 보통 "distance(낮을수록 유사)"인 경우가 많습니다.
        # 아래 값은 기본 안전값이고, 로그 보고 튜닝하세요.
        max_distance: float = 0.7,
        min_docs_after_filter: int = 12,
        enable_query_refine: bool = False,

        # Langgraph Vocab, News, Report top k 설정값
        vocab_k: int = 1,
        news_k: int = 3,
        stock_report_k: int = 3,
        industry_report_k: int = 3,
        market_report_k: int = 3,
        economy_report_k: int = 3,
        
        # Langgraph Prediction 에이전트용 top k 값
        short_term_market_k: int = 1,
        short_term_company_k: int = 1,
        short_term_news_k: int = 1,
        long_term_industry_k: int = 1,
        long_term_economy_k: int = 1,

        debug: bool = True,
    ):
        self.vllm_base_url = vllm_base_url 
        self.vllm_model = vllm_model
        self.embedding_model = embedding_model
        self.vllm_api_key = vllm_api_key or os.getenv("VLLM_API_KEY", "vllm-key")

        self.retrieval_k = retrieval_k
        self.top_articles = top_articles
        self.max_chunks_per_article = max_chunks_per_article
        self.max_distance = max_distance
        self.min_docs_after_filter = min_docs_after_filter
        self.enable_query_refine = enable_query_refine
        self.debug = debug

        # 1) 임베딩 준비 (Ollama Embeddings)
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size

        self.chroma_base_dir = chroma_base_dir
        
        # 검색 k 값 저장
        self.k_values = {
            "vocab": vocab_k,
            "news": news_k,
            "stock_report": stock_report_k,
            "industry_report": industry_report_k,
            "market_report": market_report_k,
            "economy_report": economy_report_k,
            "short_term_market": short_term_market_k,
            "short_term_company": short_term_company_k,
            "short_term_news": short_term_news_k,
            "long_term_industry": long_term_industry_k,
            "long_term_economy": long_term_economy_k,
        }


        # 2) 벡터 DB 준비
        self.news_db: Optional[Chroma] = None
        if os.path.isdir(persist_directory):
            try:
                self.news_db = Chroma(
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                    embedding_function=self.embedding,
                )
                if self.debug:
                    try:
                        print("[Chroma] persist_directory(abs):", os.path.abspath(persist_directory))
                        print("[Chroma] collection_name:", collection_name)
                        print("[Chroma] count:", self.news_db._collection.count())
                    except Exception as e:
                        print("[Chroma] count check failed:", e)
            except Exception as e:
                print(f"[WARN] Chroma init failed -> DB disabled: {e}")
                self.news_db = None
        else:
            print(f"[WARN] persist_directory not found: {persist_directory} -> DB disabled")

        # 3) LLM 준비
        self.llm = ChatOpenAI(
            base_url=self.vllm_base_url,
            model=self.vllm_model,
            api_key=self.vllm_api_key
        )
        self._vectorstore_cache = {}
        self._embeddings = None
        
        # 그래프 빌드
        self.app = self._build_graph()

        if self.debug:
            print(f"[FinancialAgent] 초기화 완료")
            print(f"  - vLLM: {self.vllm_base_url}")
            print(f"  - Model: {self.vllm_model}")
            print(f"  - Embedding: {self.embedding_model} (device: {self.embedding_device})")
            print(f"  - ChromaDB: {self.chroma_base_dir}")
            print(f"  - K values: {self.k_values}")



    def get_embeddings(self):
            """임베딩 모델을 한 번만 로드 (싱글톤)"""
            if self._embeddings is None:
                if self.debug:
                    print(f"[INFO] 임베딩 모델 로드 중: {self.embedding_model}")
                
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': self.embedding_device},
                    encode_kwargs={'device': self.embedding_device, 'batch_size': self.embedding_batch_size}
                )
            return self._embeddings
        
    def get_vectorstore(self, db_name: str):
            """ChromaDB 인스턴스를 캐싱하여 재사용"""
            if db_name not in self._vectorstore_cache:
                embeddings = self.get_embeddings()
                db_path = os.path.join(self.chroma_base_dir, db_name)
                
                # ✅ FIX: collection_name을 sqlite3에서 확인한 'langchain'으로 강제 고정
                self._vectorstore_cache[db_name] = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_name="langchain"  # <--- 여기서 'langchain'을 써야 데이터를 읽어옵니다!
                )
                
                if self.debug:
                    count = self._vectorstore_cache[db_name]._collection.count()
                    print(f"[DEBUG] DB: {db_name} | 데이터 개수: {count}개 로드 완료")
                    
            return self._vectorstore_cache[db_name]
        
    def search_db(self, db_name: str, query: str, k: int = 3):
        """캐싱된 vectorstore를 사용하여 검색"""
        try:
            vectorstore = self.get_vectorstore(db_name)
            results = vectorstore.similarity_search(query, k=k)
            
            if self.debug and results:
                print(f"\n[SEARCH SUCCESS] DB: {db_name}, 검색어: '{query}', K: {k}, 결과: {len(results)}")
                print("-" * 80)
                
                for idx, doc in enumerate(results, 1):
                    if "Vocab" in db_name:
                        term = doc.page_content
                        description = doc.metadata.get("description", "설명 정보 없음")
                        print(f"[{idx}] 용어: {term}")
                        print(f"    설명: {description[:100]}")
                    else:
                        title = doc.metadata.get("title", "제목 없음")
                        content = doc.page_content
                        print(f"[{idx}] 제목: {title}")
                        print(f"    내용: {content[:100]}")
                    print()
                
                print("-" * 80)
            
            # 포맷팅
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


    def main_router(self, state: AgentState):
        if self.debug:
            print("\n[ROUTER] main_router 실행 중...")
        
        prompt = f"""
    질문: {state['query']}
    아래 중 하나만 정확히 선택:
    - vocab (경제/통계 용어 질문)
    - report (산업, 경제, 종목(기업), 시황에 대한 질문 또는 리포트 분석)
    - news (뉴스 분석, 최근 뉴스 질문)
    - prediction (기업 주가 예측)
    - chat (일반 대화, 인사말, 농담 등)
    출력은 vocab, report, news, prediction, chat 5가지 중 하나로만 반드시 출력해.
    Assistant:
    """
        category = self.llm.invoke(prompt).content.strip().lower()
        
        if self.debug:
            print(f"[ROUTER] 선택된 카테고리: {category}")
        
        return {"category": category}

    def vocab_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] vocab_node 실행 중...")
        
        context = self.search_db("Vocab_chroma_db", state['query'], k=self.k_values["vocab"])
        res = self.llm.invoke(f"문맥: {context}\n질문: {state['query']}문맥을 보고 질문에 대해 답변해줘.").content
        return {"response": res}

    def news_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] news_node 실행 중...")
        
        context = self.search_db("News_chroma_db", state['query'], k=self.k_values["news"])
        res = self.llm.invoke(f"뉴스: {context}\n질문: {state['query']} 뉴스를 보고 질문에 대해 분석해줘.").content
        return {"response": res}

    def report_router_node(self, state: AgentState):
        if self.debug:
            print("\n[ROUTER] report_router_node 실행 중...")
        
        prompt = f"""
    질문: {state['query']}
    아래 중 하나만 정확히 선택:
    - stock (종목, 또는 회사, 회사 리포트, 종목 리포트)
    - industry (산업, 특정 산업 동향, 산업 리포트)
    - market (시황, 현재 시장 상황, 시황 리포트)
    - economy (경제, 현재 경제 상황, 경제 리포트)
    출력은 stock, industry, market, economy 4가지 중 하나로만 반드시 출력해.
    Assistant:
    """
        sub_category = self.llm.invoke(prompt).content.strip().lower()
    
        if self.debug:
            print(f"[ROUTER] 선택된 서브카테고리: {sub_category}")
        
        return {"sub_category": sub_category}

    def stock_report_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] stock_report_node 실행 중...")
        
        context = self.search_db("Company_report_chroma_db", state['query'], k=self.k_values["stock_report"])
        res = self.llm.invoke(f"종목(회사) 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
        return {"response": res}

    def industry_report_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] industry_report_node 실행 중...")
        
        context = self.search_db("Industry_report_chroma_db", state['query'], k=self.k_values["industry_report"])
        res = self.llm.invoke(f"산업 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
        return {"response": res}

    def market_report_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] market_report_node 실행 중...")
        
        context = self.search_db("MarketConditions_report_chroma_db", state['query'], k=self.k_values["market_report"])
        res = self.llm.invoke(f"시황 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
        return {"response": res}

    def economy_report_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] economy_report_node 실행 중...")
        
        context = self.search_db("Economy_report_chroma_db", state['query'], k=self.k_values["economy_report"])
        res = self.llm.invoke(f"경제 리포트: {context}\n질문: {state['query']} 리포트를 보고 질문에 대해 답변해줘.").content
        return {"response": res}

    def short_term_agent(self, state: AgentState):
        if self.debug:
            print("\n[AGENT] short_term_agent 실행 중...")
        
        context = (
            self.search_db("MarketConditions_report_chroma_db", state['query'], k=self.k_values["short_term_market"]) +
            self.search_db("Company_report_chroma_db", state['query'], k=self.k_values["short_term_company"]) +
            self.search_db("News_chroma_db", state['query'], k=self.k_values["short_term_news"])
        )
        history = "\n".join(state["debate_history"])
        res = self.llm.invoke(
            f"당신은 단기 주식 예측 전문가입니다. 질문에 대해 생각한 뒤 단기 전망을 제시하세요. \n질문: {state['query']}\n문맥: {context}\n이전 토론:\n{history}\n단기 전망 제시."
        ).content
        
        return {"debate_history": [f"단기: {res}"]}

    def long_term_agent(self, state: AgentState):
        if self.debug:
            print("\n[AGENT] long_term_agent 실행 중...")
        
        context = (
            self.search_db("Industry_report_chroma_db", state['query'], k=self.k_values["long_term_industry"]) +
            self.search_db("Economy_report_chroma_db", state['query'], k=self.k_values["long_term_economy"])
        )
        history = "\n".join(state["debate_history"])
        res = self.llm.invoke(
            f"당신은 장기 주식 예측 전문가입니다. 질문에 대해 생각한 뒤 장기 전망을 제시하세요.\n질문: {state['query']}\n문맥: {context}\n이전 토론:\n{history}\n장기 관점 제시."
        ).content
        
        return {
            "debate_history": [f"장기: {res}"],
            "debate_count": state["debate_count"] + 1
        }

    def finalize_prediction(self, state: AgentState):
        if self.debug:
            print("\n[NODE] finalize_prediction 실행 중...")
        
        history = "\n".join(state["debate_history"])
        res = self.llm.invoke(f"질문에 대한 토론 요약 및 최종 주가 예측 결론 도출 \n질문: {state['query']}:\n 토론 내용: {history}").content
        return {"response": res}

    def chat_node(self, state: AgentState):
        if self.debug:
            print("\n[NODE] chat_node 실행 중...")
        
        res = self.llm.invoke(state['query']).content
        return {"response": res}

    # ========================================
    # 라우팅 로직
    # ========================================

    def main_routing_logic(self, state: AgentState) -> Literal[
        "vocab", "news", "report_router_node", "short_term_agent", "chat"
    ]:
        cat = state["category"]
        if "vocab" in cat:
            if self.debug:
                print(f"[ROUTING] main_router -> vocab")
            return "vocab"
        if "report" in cat:
            if self.debug:
                print(f"[ROUTING] main_router -> report_router_node")
            return "report_router_node"
        if "news" in cat:
            if self.debug:
                print(f"[ROUTING] main_router -> news")
            return "news"
        if "prediction" in cat:
            if self.debug:
                print(f"[ROUTING] main_router -> short_term_agent")
            return "short_term_agent"
        
        if self.debug:
            print(f"[ROUTING] main_router -> chat (기본값)")
        return "chat"

    def report_routing_logic(self, state: AgentState) -> Literal[
        "stock_report", "industry_report", "market_report", "economy_report"
    ]:
        sub = state["sub_category"]
        if "stock" in sub:
            if self.debug:
                print(f"[ROUTING] report_router -> stock_report")
            return "stock_report"
        if "industry" in sub:
            if self.debug:
                print(f"[ROUTING] report_router -> industry_report")
            return "industry_report"
        if "market" in sub:
            if self.debug:
                print(f"[ROUTING] report_router -> market_report")
            return "market_report"
        
        if self.debug:
            print(f"[ROUTING] report_router -> economy_report (기본값)")
        return "economy_report"

    def debate_routing_logic(self, state: AgentState) -> Literal[
        "short_term_agent", "finalize_prediction"
    ]:
        if state["debate_count"] >= 2:
            if self.debug:
                print(f"[ROUTING] long_term_agent -> finalize_prediction (토론 횟수: {state['debate_count']})")
            return "finalize_prediction"
        
        if self.debug:
            print(f"[ROUTING] long_term_agent -> short_term_agent (토론 계속, 현재 횟수: {state['debate_count']})")
        return "short_term_agent"

# ========================================
# 그래프 빌드
# ========================================

    def _build_graph(self):
        """LangGraph 빌드"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("main_router", self.main_router)
        workflow.add_node("vocab", self.vocab_node)
        workflow.add_node("news", self.news_node)
        workflow.add_node("report_router_node", self.report_router_node)
        workflow.add_node("stock_report", self.stock_report_node)
        workflow.add_node("industry_report", self.industry_report_node)
        workflow.add_node("market_report", self.market_report_node)
        workflow.add_node("economy_report", self.economy_report_node)
        workflow.add_node("short_term_agent", self.short_term_agent)
        workflow.add_node("long_term_agent", self.long_term_agent)
        workflow.add_node("finalize_prediction", self.finalize_prediction)
        workflow.add_node("chat", self.chat_node)
        
        # 엣지 추가
        workflow.add_edge(START, "main_router")
        
        workflow.add_conditional_edges(
            "main_router",
            self.main_routing_logic,
            {
                "vocab": "vocab",
                "news": "news",
                "report_router_node": "report_router_node",
                "short_term_agent": "short_term_agent",
                "chat": "chat",
            }
        )
    
        workflow.add_conditional_edges(
            "report_router_node",
            self.report_routing_logic,
            {
                "stock_report": "stock_report",
                "industry_report": "industry_report",
                "market_report": "market_report",
                "economy_report": "economy_report",
            }
        )
        
        workflow.add_edge("short_term_agent", "long_term_agent")
        workflow.add_conditional_edges(
            "long_term_agent",
            self.debate_routing_logic,
            {
                "short_term_agent": "short_term_agent",
                "finalize_prediction": "finalize_prediction",
            }
        )
    
    # 종료 엣지
        workflow.add_edge("vocab", END)
        workflow.add_edge("news", END)
        workflow.add_edge("stock_report", END)
        workflow.add_edge("industry_report", END)
        workflow.add_edge("market_report", END)
        workflow.add_edge("economy_report", END)
        workflow.add_edge("finalize_prediction", END)
        workflow.add_edge("chat", END)
        
        return workflow.compile()

# ========================================
# 실행 메서드
# ========================================


    def invoke(self, query: str) -> dict:
        # 1. 현재 날짜 구하기 (예: 2026-01-26)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 2. 날짜와 기존 쿼리 합치기
        # 예: "2026-01-26: 삼성전자 주가 전망 알려줘"
        refined_query = f"[{today_str}] {query}"
        
        # 3. state에 넣기
        state = {
            "query": refined_query,  # 이제 모든 노드가 날짜가 붙은 쿼리를 봅니다.
            "category": "",
            "sub_category": "",
            "debate_history": [],
            "debate_count": 0,
            "response": "",
        }
        
        if self.debug:
            print(f"[Invoke] 날짜 포함 쿼리: {refined_query}")
            
        result = self.app.invoke(state)
        return result

    def run_chatbot(self):
        """대화형 챗봇 실행 (CLI)"""
        print("=" * 50)
        print("금융 에이전트 챗봇 시작 (종료: exit)")
        print("=" * 50)
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n종료합니다.")
                break
            
            try:
                result = self.invoke(user_input)
                print("\n" + "=" * 50)
                print("Assistant:")
                print(result.get("response", "응답을 생성할 수 없습니다."))
                print("=" * 50)
            except Exception as e:
                print(f"\n[ERROR] 그래프 실행 중 오류 발생: {e}")


# ========================================
# FastAPI 연동용 전역 인스턴스
# ========================================

# 기본 설정으로 전역 인스턴스 생성
agent = FinancialAgent(
    # vLLM은 환경변수에서 자동 로드
    embedding_device="cuda",  # CPU 서버면 "cpu"로 변경
    chroma_base_dir=os.getenv("CHROMA_DIR", "./Chroma_db"),
    
    # 검색 k 값 커스터마이징 (필요 시 조정)
    vocab_k=1,
    news_k=3,
    stock_report_k=3,
    industry_report_k=3,
    market_report_k=3,
    economy_report_k=3,
    
    short_term_market_k=1,
    short_term_company_k=1,
    short_term_news_k=1,
    long_term_industry_k=1,
    long_term_economy_k=1,
    
    debug=True,
)

# FastAPI에서 사용할 app (하위 호환성)
app = agent.app


# ========================================
# CLI 실행
# ========================================

if __name__ == "__main__":
    agent.run_chatbot()


