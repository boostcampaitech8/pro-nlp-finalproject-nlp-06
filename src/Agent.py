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
from langchain_core.messages import SystemMessage, HumanMessage

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
    print("\n[ROUTER] main_router 실행 중...")
    system_prompt = "당신은 경제, 금융 챗봇을 위한 router입니다. 사용자의 입력에 대해 다음 3가지 중 하나로 분류하세요. [chat] : 일반 대화, 인삿말, 농담 등 [rag] : 검색 엔진. 뉴스와 경제관련 리포트, 용어 참조가 필요한 질문 [prediction] : 주가 예측, 산업 예측 등 예측에 대한 질문"
    user_prompt = f"사용자: {state['query']}\n\n출력은 chat, rag, prediction 중 하나만 출력하세요. (리스트가 아닌 단일 문자열)\nAssistant:"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    res = router_llm.invoke(messages).content.strip().lower()
    
    category = "rag" if "rag" in res else "prediction" if "prediction" in res else "chat"
    print(f"[ROUTER] 선택된 카테고리: {category}")
    return {"category": category}

def rag_router_node(state: AgentState):
    print("\n[ROUTER] rag_router_node 실행...")
    system_prompt = "당신은 경제, 금융 챗봇을 위한 router입니다. 당신은 6가지 데이터베이스를 사용할 수 있습니다. - vocab (경제/통계 용어 사전), news (최신 뉴스), stock (종목/회사 리포트), industry (산업 리포트), market (시황 리포트), economy (경제 리포트)"
    user_prompt = f"위 데이터베이스 중 질문과 관련된 모든 항목을 리스트 형식으로 골라주세요. (예: ['news', 'stock'], ['vocab', 'industry'], ['news', 'market', 'economy'])\n\n질문: {state['query']}\n\n출력은 반드시 6개 데이터베이스 중 필요한 데이터베이스 (한개, 또는 다중)를 골라 리스트 형태로 출력하세요.\nAssistant:"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    res = router_llm.invoke(messages).content.strip()
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
    context = search_db("Vocab_chroma_db", state['query'], k=1)
    system_prompt = "제공된 문맥을 바탕으로 경제 용어를 설명하세요."
    user_prompt = f"문맥: {context}\n질문: {state['query']} 설명."
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"results": [f"[용어 사전 결과]\n{res}"]}

def news_node(state: AgentState):
    context = search_db("News_chroma_db", state['query'], k=10)
    system_prompt = "제공된 뉴스 데이터를 바탕으로 질문을 분석하세요. 반드시 출처와 개시날짜를 포함해야 합니다."
    user_prompt = f"뉴스: {context}\n질문: {state['query']} 분석. 출처/날짜 포함."
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"results": [f"[뉴스 분석 결과]\n{res}"]}

def stock_report_node(state: AgentState):
    context = search_db("Company_report_chroma_db", state['query'], k=3)
    system_prompt = "종목 리포트 데이터를 기반으로 심층 분석을 수행하세요.반드시 출처와 개시날짜를 포함해야 합니다."
    user_prompt = f"종목 리포트 기반 분석: {context}\n질문: {state['query']}"
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"results": [f"[종목 리포트 분석]\n{res}"]}

def industry_report_node(state: AgentState):
    context = search_db("Industry_report_chroma_db", state['query'], k=3)
    system_prompt = "산업 리포트 데이터를 기반으로 산업 동향을 분석하세요.반드시 출처와 개시날짜를 포함해야 합니다."
    user_prompt = f"산업 리포트 기반 분석: {context}\n질문: {state['query']}"
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"results": [f"[산업 리포트 분석]\n{res}"]}

def market_report_node(state: AgentState):
    context = search_db("MarketConditions_report_chroma_db", state['query'], k=3)
    system_prompt = "시황 리포트를 바탕으로 현재 시장 상황을 분석하세요.반드시 출처와 개시날짜를 포함해야 합니다."
    user_prompt = f"시황 리포트 기반 분석: {context}\n질문: {state['query']}"
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"results": [f"[시황 리포트 분석]\n{res}"]}

def economy_report_node(state: AgentState):
    context = search_db("Economy_report_chroma_db", state['query'], k=3)
    system_prompt = "경제 리포트를 기반으로 거시 경제 흐름을 분석하세요.반드시 출처와 개시날짜를 포함해야 합니다."
    user_prompt = f"경제 리포트 기반 분석: {context}\n질문: {state['query']}"
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"results": [f"[경제 리포트 분석]\n{res}"]}

# 예측 에이전트 
def extract_companies_node(state: AgentState):
    system_prompt = """당신은 한국 금융 텍스트에서 기업명을 정확하게 찾아내는 **'엔티티 추출 전문가'**입니다. 다음 지침에 따라 질문에서 기업명만 정확히 추출하세요.

1. **영문 대문자**: SK, LG, KT, POSCO 등 영문이 포함된 기업은 대문자 표기를 우선시합니다.
2. **산업군 키워드**: 이름 뒤에 '전자', '생명', '화학', '바이오', '증권' 등이 붙어 있다면 하나의 기업명으로 묶어서 추출하세요.
3. **불필요한 요소 제거**: 기업명 뒤에 붙은 조사(가, 는, 의, 에 등)와 문장 부호는 반드시 제거하고 순수 기업명만 리스트에 담으세요.
4. **결과 형식**: 오직 파이썬 리스트 형태(예: ['기업A', '기업B'])로만 출력하며, 설명이나 추가 텍스트는 절대 포함하지 마세요."""
    user_prompt = f"질문: {state['query']}\n\n예시:\n- 질문: '삼성전자와 SK하이닉스 주가 예측해줘' → ['삼성전자', 'SK하이닉스']\n- 질문: '현대차 내일 얼마나 오를까?' → ['현대차']\n\n출력은 반드시 리스트 형태로만 출력하세요.\nAssistant:"
    
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content.strip()
    try:
        target_companies = ast.literal_eval(res)
    except:
        target_companies = []
    
    tft_json_path = os.getenv("TFT_INFERENCE_JSON", "./inference_results.json")
    try:
        with open(tft_json_path, 'r', encoding='utf-8') as f:
            inference_json = json.load(f)
        extracted_data = []
        for item in inference_json.get("results", []):
            if item["name"] in target_companies:
                first_day = item["forecasts"][0]
                extracted_data.append({"name": item["name"], "base_close": item["base_close"], "prediction": {"date": first_day["date"], "price": first_day["price"], "pct_change": first_day["pct_change"], "lower": first_day["price_lower"], "upper": first_day["price_upper"]}, "variables": item["top_variables"], "attention": item["top_attention"]})
        return {"target_companies": target_companies, "tft_data": extracted_data}
    except:
        return {"target_companies": target_companies, "tft_data": []}

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
    companies_str = ", ".join(state["target_companies"])
    company_context = search_db("Company_report_chroma_db", state['query'], k=3)
    industry_context = search_db("Industry_report_chroma_db", state['query'], k=3)
    news_context = search_db("News_chroma_db", state['query'], k=5)
    
    system_prompt = "너는 기업 분석 리포트와 뉴스를 통해 중단기 전망을 제시하는 **Research Agent**야."
    user_prompt = f"### [분석 대상]\n기업: {companies_str}\n\n### [참고 자료]\n기업 리포트: {company_context}\n산업 리포트: {industry_context}\n최신 뉴스: {news_context}\n\n### [분석 요청]\n1. 최근 실적, 신규 사업, 경쟁 상황 파악\n2. 향후 3~6개월 관점 성장 가능성과 리스크 평가\n3. 중단기 매수/보유/매도 의견 제시\n\n질문: {state['query']}"
    
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"debate_history": [f"[Research Agent - 중단기 전망]\n{res}"]}

def macro_agent_node(state: AgentState):
    market_context = search_db("MarketConditions_report_chroma_db", state['query'], k=3)
    economy_context = search_db("Economy_report_chroma_db", state['query'], k=3)
    
    system_prompt = "너는 거시경제와 시장 전반을 분석하는 **Macro Agent**야."
    user_prompt = f"### [참고 자료]\n시황 리포트: {market_context}\n경제 리포트: {economy_context}\n\n### [분석 요청]\n1. 거시경제 지표가 해당 종목/산업에 미치는 영향 분석\n2. 글로벌 경기 흐름과 국내 증시 전망 종합\n3. 거시적 관점 리스크 요인 제시\n\n질문: {state['query']}"
    
    res = router_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"debate_history": [f"[Macro Agent - 거시경제 관점]\n{res}"]}

def finalize_prediction(state: AgentState):
    quant_result = next((h for h in state["debate_history"] if "[Quant Agent" in h), "")
    combined_analysis = "\n\n".join([h for h in state["debate_history"] if "[Research Agent" in h or "[Macro Agent" in h])
    
    system_prompt = """당신은 복잡한 주식 분석 데이터를 주식, 금융, 투자 초보자도 이해할 수 있게 쉽게 풀어서 설명해주는 **'친절한 투자 멘토'**입니다. 

제공되는 수치(Quant), 기업 정보(Research), 시장 상황(Macro) 데이터를 바탕으로 최종 답변을 작성할 때, 다음의 **'초보자 배려 원칙'**을 반드시 지키세요.

1. **전문 용어 금지 및 풀이**: 
   - '퀀트(Quant)', '거시 경제(Macro)', '신뢰 구간' 같은 어려운 용어는 가급적 피하거나, 사용해야 한다면 반드시 쉬운 비유를 덧붙이세요.
   - 문장은 짧게 끊어서 쓰고, "해요"체나 "습니다"체를 사용하여 부드럽게 대화하듯 작성하세요.

2. **분석 결과의 친절한 요약**: 
   - [1. 수학 모델이 본 내일의 날씨(예측)]: Quant 데이터를 바탕으로 내일 주가가 오를지 내릴지, 그 확률과 예상 가격을 아주 쉽게 설명하세요.
   - [2. 전문가들이 보는 이 기업의 스토리]: 기업 리포트(Research)와 세상 뉴스(Macro)를 합쳐서, 이 회사가 지금 어떤 상황인지 뉴스나 책을 읽어주듯 설명하세요.

3. **시각적 구조화**: 
   - 중요한 내용은 **굵게** 표시하고, 불렛포인트(•)를 사용하여 한눈에 들어오게 만드세요.
   - 마지막에는 반드시 한 줄로 "그래서 내일은 어떻게 하면 좋을지"에 대한 조언을 덧붙이세요.

4. **객관성 유지**: 초보자가 오해하지 않도록, 투자의 책임은 본인에게 있으며 분석 결과는 참고용이라는 점을 따뜻하게 언급하세요."""
    user_prompt = f"### [Quant Agent의 다음날 주가 예측]\n{quant_result}\n\n### [Research Agent와 Macro Agent의 종합 분석]\n{combined_analysis}\n\n### [최종 답변 형식]\n1. Quant Agent 예측 요약\n2. 종합 투자 의견 (중단기 및 거시경제 관점 통합)\n\n질문: {state['query']}"
    
    res = answer_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"response": res}


def chat_node(state: AgentState):
    """일반 대화 - 바로 최종 답변 반환 (aggregator 거치지 않음)"""
    print("\n[NODE] chat_node 실행...")
    res = answer_llm.invoke(state['query']).content
    return {"response": res}  # results가 아닌 response에 직접 저장

# [추가] 최종 답변 통합 노드 (Fan-in)
def final_aggregator(state: AgentState):
    combined_context = "\n\n".join(state["results"])
    system_prompt = """당신은 복잡한 금융 및 경제 분석 결과를 통합하여 일반인도 쉽게 이해할 수 있도록 전달하는 **'금융 지식 가이드'**입니다. 

당신의 목표는 다양한 분석 데이터를 논리적이고 구조적으로 재구성하여 사용자에게 최적의 답변을 제공하는 것입니다. 답변 시 아래 지침을 반드시 준수하세요.

1. **초보자 맞춤형 설명**: 경제 전문 용어를 피하거나, 사용이 불가피할 경우 반드시 쉬운 비유를 들어 설명하세요. 문장은 짧고 간결하게 작성하여 가독성을 높입니다.
2. **구조적 답변**: 정보를 단순히 나열하지 말고 '주요 현황', '전문가 분석 요약', '향후 전망' 등 논리적인 섹션으로 나누어 답변하세요.
3. **출처 정보의 완전성**: 분석에 활용된 모든 뉴스 기사와 리포트의 **[제목]**과 **[게시 날짜]**를 본문 하단이나 인용구에 반드시 명시하여 정보의 신뢰성을 확보하세요.
4. **객관성 유지**: 여러 분석 결과가 충돌할 경우 이를 가감 없이 보여주되, 사용자가 판단을 내릴 수 있도록 중립적인 입장에서 통합하세요."""
    user_prompt = f"질문: {state['query']}\n\n분석 결과들:\n{combined_context}"
    
    res = answer_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    return {"response": res}

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