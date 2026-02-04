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
import ast
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv() # .env 파일 로드

# ------------------------------------------------------------
# 상대경로(프로젝트 기준)로 DB 경로 통일
# - src/Agent.py 기준으로 project/ 찾기
# - main.py / pipeline.py 와 같은 규칙(= PROJECT_ROOT, CHROMA_DIR env 우선)
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = THIS_FILE.parents[1]  # project/
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "Chroma_db"))).resolve()


# 1. 상태 정의 (State)
class AgentState(TypedDict):
    query: str
    category: str              # 단일 선택으로 변경
    rag_categories: List[str]  # RAG router에서 선택된 카테고리들
    # 각 노드의 결과물을 통합하기 위한 리듀서 (리스트 합치기)
    results: Annotated[List[str], operator.add] 
    debate_history: Annotated[List[str], operator.add]
    debate_count: int
    response: str
    # Prediction 관련 추가
    target_companies: List[str]  # 추출된 기업명 리스트
    tft_data: List[dict]   


# LLM 설정 (vLLM 서빙 모델 연동)
router_llm = ChatOpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
    model=os.getenv("VLLM_MODEL", "skt/A.X-4.0-Light"),
    api_key=os.getenv("VLLM_API_KEY", "vllm-key"),
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
            model_name=os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask"),
            model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cuda")},  # 기본 cuda
            encode_kwargs={
                "device": os.getenv("EMBEDDING_DEVICE", "cuda"),
                "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            },
        )
    return _embeddings


def get_vectorstore(db_name: str, collection_name: str = None):
    """ChromaDB 인스턴스를 캐싱하여 재사용 (뉴스 DB 예외 처리 포함)"""

    # 1. 뉴스 DB인 경우 컬렉션 이름을 자동으로 설정
    if collection_name is None:
        if db_name == "News_chroma_db":
            collection_name = os.getenv("CHROMA_NEWS_COLLECTION", "naver_finance_news_chunks")
        else:
            collection_name = "langchain"

    # 2. 캐시 키를 (DB이름, 컬렉션이름) 조합으로 만들어 충돌 방지
    cache_key = (db_name, collection_name)

    if cache_key not in _vectorstore_cache:
        print(f"[INFO] ChromaDB 로드 중: {db_name} (Collection: {collection_name})")

        embeddings = get_embeddings()

        # 절대경로 하드코딩 제거 → PROJECT_ROOT/CHROMA_DIR 기준 상대경로로 통일
        # project_root = "/data/ephemeral/home/pro-nlp-finalproject-nlp-06"
        # db_path = os.path.join(project_root, "Chroma_db", db_name)
        db_path = (CHROMA_DIR / db_name).resolve()

        # 디버그 찍고 싶으면 켜도 됨
        print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)
        print("[DEBUG] CHROMA_DIR:", CHROMA_DIR)
        print("[DEBUG] db_path:", str(db_path))

        _vectorstore_cache[cache_key] = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name=collection_name,
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
    print("\n[ROUTER] main_router 실행 중 (단일 선택)...")
    prompt = f"""
당신은 경제, 금융 챗봇을 위한 router입니다.
사용자의 입력에 대해 다음 3가지 중 하나로 분류하세요.

[chat] : 일반 대화, 인삿말, 농담 등 
[rag] : 검색 엔진. 뉴스와 경제관련 리포트, 용어 참조가 필요한 질문
[prediction] : 주가 예측, 산업 예측 등 예측에 대한 질문

사용자: {state['query']}

출력은 chat, rag, prediction 중 하나만 출력하세요. (리스트가 아닌 단일 문자열)
Assistant:"""
    
    res = router_llm.invoke(prompt).content.strip().lower()
    
    # 결과 정제
    if "chat" in res:
        category = "chat"
    elif "rag" in res:
        category = "rag"
    elif "prediction" in res:
        category = "prediction"
    else:
        category = "chat"  # 기본값
    
    print(f"[ROUTER] 선택된 카테고리: {category}")
    return {"category": category}

def rag_router_node(state: AgentState):
    """뉴스, 리포트, 용어를 통합한 RAG Router"""
    print("\n[ROUTER] rag_router_node 실행 (다중 선택)...")
    prompt = f"""
당신은 경제, 금융 챗봇을 위한 router입니다.
당신은 6가지 데이터베이스를 사용할 수 있습니다.

- vocab (경제/통계 용어 사전)
- news (최신 뉴스)
- stock (종목/회사 리포트)
- industry (산업 리포트)
- market (시황 리포트)
- economy (경제 리포트)

위 데이터베이스 중 질문과 관련된 모든 항목을 리스트 형식으로 골라주세요. 
(예: ['news', 'stock'], ['vocab', 'industry'], ['news', 'market', 'economy'])

질문: {state['query']}

출력은 반드시 6개 데이터베이스 중 필요한 데이터베이스 (한개, 또는 다중)를 골라 리스트 형태로 출력하세요.
Assistant:
"""
    res = router_llm.invoke(prompt).content.strip()
    try:
        rag_categories = ast.literal_eval(res)
    except:
        rag_categories = ["news"]
    
    print(f"[ROUTER] 선택된 RAG 카테고리: {rag_categories}")
    return {"rag_categories": rag_categories}


# === Prediction 관련 함수들 ===

def extract_tft_data(target_companies, inference_json):
    """
    target_companies: ['유한양행', '고려아연'] 형태의 리스트
    inference_json: TFT 추론 결과 데이터
    """
    extracted_results = []
    
    # results 리스트에서 대상 기업만 필터링
    for item in inference_json.get("results", []):
        if item["name"] in target_companies:
            # 첫 번째 날(D+1) 예측 데이터 추출
            first_day = item["forecasts"][0]
            
            data = {
                "name": item["name"],
                "base_close": item["base_close"],
                "prediction": {
                    "date": first_day["date"],
                    "price": first_day["price"],
                    "pct_change": first_day["pct_change"],
                    "lower": first_day["price_lower"],
                    "upper": first_day["price_upper"]
                },
                "variables": item["top_variables"],
                "attention": item["top_attention"]
            }
            extracted_results.append(data)
            
    return extracted_results

def build_quant_prompt(extracted_data):
    """Quant Agent용 프롬프트 생성"""
    context_text = ""
    for stock in extracted_data:
        var_text = ", ".join([f"{v['name']}({v['weight']:.2%})" for v in stock['variables']])
        attn_text = ", ".join([f"{a['date']}(중요도 {a['weight']:.2%})" for a in stock['attention']])
        
        context_text += f"""
[종목명: {stock['name']}]
- 기준가: {stock['base_close']:,}원
- {stock['prediction']['date']} 예측 종가: {stock['prediction']['price']:,}원 ({stock['prediction']['pct_change']}% 변동 예상)
- 신뢰 구간: {stock['prediction']['lower']:,}원 ~ {stock['prediction']['upper']:,}원
- 주요 영향 변수: {var_text}
- 모델이 참고한 과거 유사 시점: {attn_text}
---"""
    
    full_prompt = f"""
너는 데이터와 수치를 기반으로 시장을 분석하는 냉철한 **Quant Agent(수치 분석 전문가)**야.
제공된 [TFT 모델 예측 데이터]를 바탕으로 해당 종목의 단기 향방을 분석해줘.

### [제공된 예측 데이터]
{context_text}

### [분석 요청 사항]
1. **수치 해석**: 모델이 예측한 상승/하락 폭을 명확히 전달하고, 예측의 신뢰 구간(Upper/Lower)을 통해 변동성 위험을 진단해줘.
2. **근거 분석**: '주요 영향 변수' 중에서 가장 가중치가 높은 항목이 무엇인지, 그리고 '과거 유사 시점'의 사례가 현재 상황에 어떤 의미를 주는지 설명해줘.
3. **결론**: 데이터만 보았을 때 내일의 투자 매력도를 5점 만점으로 평가해줘.

**주의: 뉴스나 소문이 아닌, 오직 위 수치 데이터에만 근거해서 논리적으로 말해.**
"""
    return full_prompt



# RAG 노드들
def vocab_node(state: AgentState):
    print("\n[NODE] vocab_node 실행...")
    context = search_db("Vocab_chroma_db", state['query'], k=1)
    res = router_llm.invoke(f"문맥: {context}\n질문: {state['query']} 설명.").content
    return {"results": [f"[용어 사전 결과]\n{res}"]}

def news_node(state: AgentState):
    print("\n[NODE] news_node 실행...")
    context = search_db("News_chroma_db", state['query'], k=10)
    res = router_llm.invoke(f"뉴스: {context}\n질문: {state['query']} 분석. 출처/날짜 포함.").content
    return {"results": [f"[뉴스 분석 결과]\n{res}"]}

def stock_report_node(state: AgentState):
    print("\n[NODE] stock_report_node 실행...")
    context = search_db("Company_report_chroma_db", state['query'], k=3)
    res = router_llm.invoke(f"종목 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[종목 리포트 분석]\n{res}"]}

def industry_report_node(state: AgentState):
    print("\n[NODE] industry_report_node 실행...")
    context = search_db("Industry_report_chroma_db", state['query'], k=3)
    res = router_llm.invoke(f"산업 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[산업 리포트 분석]\n{res}"]}

def market_report_node(state: AgentState):
    print("\n[NODE] market_report_node 실행...")
    context = search_db("MarketConditions_report_chroma_db", state['query'], k=3)
    res = router_llm.invoke(f"시황 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[시황 리포트 분석]\n{res}"]}

def economy_report_node(state: AgentState):
    print("\n[NODE] economy_report_node 실행...")
    context = search_db("Economy_report_chroma_db", state['query'], k=3)
    res = router_llm.invoke(f"경제 리포트 기반 분석: {context}\n질문: {state['query']}").content
    return {"results": [f"[경제 리포트 분석]\n{res}"]}

# 예측 에이전트 
def extract_companies_node(state: AgentState):
    """1단계: 쿼리에서 기업명 추출"""
    print("\n[NODE] extract_companies_node 실행...")
    
    prompt = f"""
질문에서 언급된 기업명을 모두 추출하여 리스트 형태로 출력하세요.

질문: {state['query']}

예시:
- 질문: "삼성전자와 SK하이닉스 주가 예측해줘" → ['삼성전자', 'SK하이닉스']
- 질문: "현대차 내일 얼마나 오를까?" → ['현대차']

출력은 반드시 리스트 형태로만 출력하세요.
Assistant:"""
    
    res = router_llm.invoke(prompt).content.strip()
    try:
        target_companies = ast.literal_eval(res)
    except:
        target_companies = []
    
    print(f"[추출된 기업명]: {target_companies}")
    
    # TFT 추론 결과 로드 (실제 경로로 수정 필요)
    tft_json_path = os.getenv("TFT_INFERENCE_JSON", "./inference_results.json")
    try:
        with open(tft_json_path, 'r', encoding='utf-8') as f:
            inference_json = json.load(f)
        
        # TFT 데이터 추출
        tft_data = extract_tft_data(target_companies, inference_json)
        print(f"[TFT 데이터 추출 완료]: {len(tft_data)}개 기업")
        
        return {
            "target_companies": target_companies,
            "tft_data": tft_data
        }
    except Exception as e:
        print(f"[ERROR] TFT 데이터 로드 실패: {e}")
        return {
            "target_companies": target_companies,
            "tft_data": []
        }

def quant_agent_node(state: AgentState):
    """2단계: Quant Agent - TFT 데이터 분석"""
    print("\n[AGENT] quant_agent 실행...")
    
    if not state["tft_data"]:
        return {"debate_history": ["[Quant Agent] TFT 데이터를 찾을 수 없습니다."]}
    
    # Quant 프롬프트 생성
    prompt = build_quant_prompt(state["tft_data"])
    
    # Router LLM으로 분석
    res = router_llm.invoke(prompt).content
    print(f"  > Quant 분석: {res[:100]}...")
    
    return {"debate_history": [f"[Quant Agent - 수치 분석]\n{res}"]}

def research_agent_node(state: AgentState):
    """3단계: Research Agent - 리포트 기반 중단기 예측"""
    print("\n[AGENT] research_agent 실행...")
    
    companies_str = ", ".join(state["target_companies"])
    
    # Company, Industry, News 리포트 검색
    company_context = search_db("Company_report_chroma_db", state['query'], k=3)
    industry_context = search_db("Industry_report_chroma_db", state['query'], k=3)
    news_context = search_db("News_chroma_db", state['query'], k=5)
    
    prompt = f"""
너는 기업 분석 리포트와 뉴스를 통해 중단기 전망을 제시하는 **Research Agent**야.

### [분석 대상]
기업: {companies_str}

### [참고 자료]
**기업 리포트:**
{company_context}

**산업 리포트:**
{industry_context}

**최신 뉴스:**
{news_context}

### [분석 요청]
1. 해당 기업의 최근 실적, 신규 사업, 경쟁 상황을 리포트와 뉴스를 통해 파악해줘.
2. 향후 3~6개월 관점에서 기업의 성장 가능성과 리스크를 평가해줘.
3. 투자자 관점에서 중단기 매수/보유/매도 의견을 제시해줘.

질문: {state['query']}
"""
    
    res = router_llm.invoke(prompt).content
    print(f"  > Research 분석: {res[:100]}...")
    
    return {"debate_history": [f"[Research Agent - 중단기 전망]\n{res}"]}

def macro_agent_node(state: AgentState):
    """4단계: Macro Agent - 거시경제 관점 분석"""
    print("\n[AGENT] macro_agent 실행...")
    
    # Market Conditions, Economy 리포트 검색
    market_context = search_db("MarketConditions_report_chroma_db", state['query'], k=3)
    economy_context = search_db("Economy_report_chroma_db", state['query'], k=3)
    
    prompt = f"""
너는 거시경제와 시장 전반을 분석하는 **Macro Agent**야.

### [참고 자료]
**시황 리포트:**
{market_context}

**경제 리포트:**
{economy_context}

### [분석 요청]
1. 현재 금리, 환율, 원자재 가격 등 거시경제 지표가 해당 종목/산업에 미치는 영향을 분석해줘.
2. 글로벌 경기 흐름과 국내 증시 전망을 종합해줘.
3. 거시적 관점에서 투자 시 유의해야 할 리스크 요인을 제시해줘.

질문: {state['query']}
"""
    
    res = router_llm.invoke(prompt).content
    print(f"  > Macro 분석: {res[:100]}...")
    
    return {"debate_history": [f"[Macro Agent - 거시경제 관점]\n{res}"]}

def finalize_prediction(state: AgentState):
    """5단계: 그래프 생성 및 최종 답변 통합"""
    print("\n[NODE] finalize_prediction_with_graph 실행...")
    
    # 1. 그래프 생성

    
    # 2. Quant Agent 결과 추출
    quant_result = ""
    for history in state["debate_history"]:
        if "[Quant Agent" in history:
            quant_result = history
            break
    
    # 3. Research + Macro 결과 통합
    research_macro_results = []
    for history in state["debate_history"]:
        if "[Research Agent" in history or "[Macro Agent" in history:
            research_macro_results.append(history)
    
    combined_analysis = "\n\n".join(research_macro_results)
    
    # 4. 최종 답변 생성 (Answer LLM 사용)
    prompt = f"""
당신은 주가 예측 전문가입니다. 다음 분석 결과들을 바탕으로 최종 답변을 작성하세요.

### [Quant Agent의 다음날 주가 예측]
{quant_result}


### [Research Agent와 Macro Agent의 종합 분석]
{combined_analysis}

### [최종 답변 형식]
다음과 같은 구조로 답변을 작성하세요:

**1. Quant Agent의 다음날 주가 예측**
- 예측 가격과 이유를 요약
- 그래프 참조 안내

**2. 종합 투자 의견**
- Research Agent와 Macro Agent의 분석을 통합
- 중단기 관점과 거시경제 관점을 모두 고려한 최종 의견 제시

질문: {state['query']}
"""
    
    final_response = answer_llm.invoke(prompt).content

    # Agent.py의 finalize_prediction_with_graph에 추가

    return {
        "response": final_response,
    }


def chat_node(state: AgentState):
    """일반 대화 - 바로 최종 답변 반환 (aggregator 거치지 않음)"""
    print("\n[NODE] chat_node 실행...")
    res = answer_llm.invoke(state['query']).content
    return {"response": res}  # results가 아닌 response에 직접 저장

# [추가] 최종 답변 통합 노드 (Fan-in)
def final_aggregator(state: AgentState):
    print("\n[NODE] final_aggregator 실행 (결과 통합)...")
    combined_context = "\n\n".join(state["results"])
    prompt = f"""
질문: {state['query']}

아래의 여러 분석 결과들을 바탕으로 사용자를 위한 최종 답변을 초보자도 이해하기 쉽게 구조적으로 작성해주세요.
참조한 모든 뉴스와 리포트의 제목과 게시 날짜를 포함시켜야 합니다.

분석 결과들:
{combined_context}
"""
    final_res = answer_llm.invoke(prompt).content
    return {"response": final_res}

# --- 그래프 구성 ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("main_router", main_router)
workflow.add_node("rag_router_node", rag_router_node)
workflow.add_node("vocab", vocab_node)
workflow.add_node("news", news_node)
workflow.add_node("stock_report", stock_report_node)
workflow.add_node("industry_report", industry_report_node)
workflow.add_node("market_report", market_report_node)
workflow.add_node("economy_report", economy_report_node)
workflow.add_node("chat", chat_node)
workflow.add_node("final_aggregator", final_aggregator)

workflow.add_node("extract_companies", extract_companies_node)
workflow.add_node("quant_agent", quant_agent_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("macro_agent", macro_agent_node)
workflow.add_node("finalize_prediction_with_graph", finalize_prediction)

workflow.add_edge(START, "main_router")

# --- 단일 라우팅 로직 ---
def main_routing_logic(state: AgentState) -> Literal["rag_router_node", "extract_companies", "chat"]:
    category = state["category"]
    if category == "rag":
        return "rag_router_node"
    elif category == "prediction":
        return "extract_companies"  # prediction 시작점 변경
    else:  # chat
        return "chat"

workflow.add_conditional_edges(
    "main_router",
    main_routing_logic,
    {
        "rag_router_node": "rag_router_node",
        "extract_companies": "extract_companies",
        "chat": "chat"
    }
)

# RAG 다중 라우팅
def rag_routing_logic(state: AgentState):
    return state["rag_categories"]

workflow.add_conditional_edges(
    "rag_router_node",
    rag_routing_logic,
    {
        "vocab": "vocab",
        "news": "news",
        "stock": "stock_report",
        "industry": "industry_report",
        "market": "market_report",
        "economy": "economy_report"
    }
)

# 예측 루프

workflow.add_edge("extract_companies", "quant_agent")
workflow.add_edge("quant_agent", "research_agent")
workflow.add_edge("research_agent", "macro_agent")
workflow.add_edge("macro_agent", "finalize_prediction_with_graph")

# RAG 노드들은 final_aggregator로 연결 (Fan-in)
workflow.add_edge("vocab", "final_aggregator")
workflow.add_edge("news", "final_aggregator")
workflow.add_edge("stock_report", "final_aggregator")
workflow.add_edge("industry_report", "final_aggregator")
workflow.add_edge("market_report", "final_aggregator")
workflow.add_edge("economy_report", "final_aggregator")

# Prediction과 chat은 바로 END로
workflow.add_edge("finalize_prediction_with_graph", END)
workflow.add_edge("chat", END)

# aggregator는 END로
workflow.add_edge("final_aggregator", END)


app = workflow.compile()

# --- 실행부 (기존 유지) ---
def run_chatbot():
    print("=" * 50)
    print("금융 에이전트 챗봇 (다중 분석 모드)")
    print("=" * 50)
    
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]: 
            print("\n종료합니다.")
            break
        
        state = {
            "query": f"오늘 날짜 : [{datetime.now().strftime('%Y-%m-%d')}] {user_input}",
            "category": "",  # 단일 선택으로 변경
            "rag_categories": [],
            "results": [],
            "debate_history": [], 
            "debate_count": 0, 
            "response": "",
            "target_companies": [],  # ✅ 추가!
            "tft_data": [],  # ✅ 추가!
        }
        
        try:
            result = app.invoke(state)
            print("\n" + "=" * 50 + "\nAssistant:\n" + result.get("response", "") + "\n" + "=" * 50)
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_chatbot()