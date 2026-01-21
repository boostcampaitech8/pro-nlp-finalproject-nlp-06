# model_ollama.py
from __future__ import annotations

import os
import re
from typing import Optional, Tuple, List, Dict, Any, Tuple as TypingTuple

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document


def _kw_to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


class RagNewsChatService:
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
        persist_directory: str = "./chroma_news",
        llm_model: str = "llama3",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        # --- Retrieval tuning knobs ---
        retrieval_k: int = 48,
        top_articles: int = 5,
        max_chunks_per_article: int = 3,
        # Chroma 점수는 보통 "distance(낮을수록 유사)"인 경우가 많습니다.
        # 아래 값은 기본 안전값이고, 로그 보고 튜닝하세요.
        max_distance: float = 0.7,
        min_docs_after_filter: int = 12,
        enable_query_refine: bool = False,
        debug: bool = True,
    ):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model

        self.retrieval_k = retrieval_k
        self.top_articles = top_articles
        self.max_chunks_per_article = max_chunks_per_article
        self.max_distance = max_distance
        self.min_docs_after_filter = min_docs_after_filter
        self.enable_query_refine = enable_query_refine
        self.debug = debug

        # 1) 임베딩 준비 (Ollama Embeddings)
        self.embedding = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
        )

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
        self.router_llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.0,
        )
        self.answer_llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.2,
        )

    # ---------------- Routing ----------------
    def should_use_news_db(self, question: str) -> bool:
        routing_prompt = f"""
당신은 뉴스 사용 여부를 판단하는 라우팅 어시스턴트입니다.

아래 질문이 '최근 금융/증시/경제 상황, 특정 뉴스 기사, 오늘/최근 무슨 일이 있었는지'와 같이
뉴스 기사 내용에 의존해서 답해야 하는 질문이라면 "USE_DB"라고만 출력하세요.

반대로, 일반적인 개념 설명(예: PER가 뭐야?, 채권이 뭐야?)처럼
뉴스를 몰라도 답변할 수 있는 질문이라면 "NO_DB"라고만 출력하세요.

정확히 USE_DB 또는 NO_DB 중 하나만 출력하세요. 다른 말은 절대 하지 마세요.

[질문]
{question}
"""
        res = self.router_llm.invoke(routing_prompt)
        decision = (getattr(res, "content", "") or "").strip().upper()
        # 방어적 파싱
        m = re.search(r"\b(USE_DB|NO_DB)\b", decision)
        final = (m.group(1) if m else "NO_DB")
        if self.debug:
            print("[라우팅 결과]:", decision, "->", final)
        return final == "USE_DB"

    # ---------------- Query refinement ----------------
    def refine_search_query(self, question: str) -> str:
        """
        사용자 질문을 벡터검색에 더 적합한 '키워드 나열'로 정제.
        정제가 실패/불안정하면 원 질문을 그대로 사용.
        """
        prompt = f"""
너는 뉴스 벡터DB 검색 쿼리를 만드는 도우미야.
아래 질문에서 '검색에 도움되는 핵심 키워드/고유명사'만 3~7개 뽑아서
쉼표로만 출력해. 다른 말 금지.

[질문]
{question}
"""
        try:
            res = self.router_llm.invoke(prompt)
            s = (getattr(res, "content", "") or "").strip()
        except Exception:
            return question

        # 너무 길거나 포맷이 이상하면 원문 유지
        if len(s) > 120:
            return question
        # 최소한 쉼표가 있어야 키워드라고 판단
        if "," not in s:
            return question

        # 정제 결과가 공백/쓰레기면 원문
        kws = [x.strip() for x in s.split(",") if x.strip()]
        if not (3 <= len(kws) <= 10):
            return question

        refined = ", ".join(kws[:7])
        if self.debug:
            print("[Refine] question ->", question)
            print("[Refine] search_q ->", refined)
        return refined

    # ---------------- Retrieval helpers ----------------
    def _filter_pairs_by_score(
        self,
        pairs: List[TypingTuple[Document, float]],
    ) -> List[TypingTuple[Document, float]]:
        """
        Chroma 점수가 'distance(낮을수록 유사)'라고 가정하고 max_distance로 필터.
        환경에 따라 score 의미가 다를 수 있으니 debug 로그로 튜닝.
        """
        cleaned = [(d, s) for d, s in pairs if s is not None]
        if not cleaned:
            return []

        # 1차: 임계치 필터
        filtered = [(d, s) for d, s in cleaned if s <= self.max_distance]

        # 너무 적으면 안전장치: 상위 일부 사용
        if len(filtered) < self.min_docs_after_filter:
            filtered = cleaned[: max(self.min_docs_after_filter, 12)]

        return filtered

    # ---------- 핵심: (doc, score) 검색결과를 기사(link) 단위로 묶고, best_score로 랭킹 ----------
    def _group_pairs_to_articles(
        self,
        pairs: List[TypingTuple[Document, float]],
        top_articles: int = 3,
    ) -> List[Dict[str, Any]]:
        by_link: Dict[str, Dict[str, Any]] = {}

        for d, score in pairs:
            m = d.metadata or {}
            link = (m.get("link") or "").strip()
            if not link:
                continue

            if link not in by_link:
                by_link[link] = {
                    "link": link,
                    "title": m.get("title", ""),
                    "press": m.get("press", ""),
                    "date": m.get("date", ""),
                    "summary": m.get("summary", ""),
                    "keywords": _kw_to_list(m.get("keywords", "")),
                    "chunks": [],
                    "best_score": score,
                }
            else:
                # 대표 점수는 최소 distance로
                by_link[link]["best_score"] = min(by_link[link]["best_score"], score)

            by_link[link]["chunks"].append(
                {
                    "chunk_index": m.get("chunk_index"),
                    "chunk_total": m.get("chunk_total"),
                    "text": d.page_content,
                    "score": score,
                }
            )

        # 각 기사 내 chunk를 chunk_index 기준 정렬
        for a in by_link.values():
            a["chunks"].sort(key=lambda x: (x["chunk_index"] is None, x["chunk_index"]))

        # 기사 랭킹: best_score 낮은 순(더 유사) + tie-breaker로 chunk 수(많은게 더 풍부)
        ranked = sorted(
            by_link.values(),
            key=lambda a: (a.get("best_score", 1e9), -len(a.get("chunks", []))),
        )

        return ranked[:top_articles]

    def _build_news_prompt(self, question: str, articles: List[Dict[str, Any]]) -> str:
        context_parts = []
        for a in articles:
            chunks_text = "\n\n".join(
                [
                    c["text"]
                    for c in a["chunks"][: self.max_chunks_per_article]
                    if (c.get("text") or "").strip()
                ]
            )

            context_parts.append(
                f"""
------------------------------
[기사 제목] {a.get('title')}
[언론사] {a.get('press')}
[날짜] {a.get('date')}
[링크] {a.get('link')}
[요약] {a.get('summary')}
[키워드] {", ".join(a.get("keywords", []))}
[검색점수] {a.get("best_score")}

[기사 내용(발췌)]
{chunks_text}
""".strip()
            )

        context = "\n\n".join(context_parts).strip()

        prompt = f"""
[Role]
너는 '초보 주식 투자자(금융 지식이 거의 없는 사회초년생)'를 돕는 투자 학습/이해 중심 챗봇이야.
사용자가 뉴스를 이해할 수 있게 '아주 친절하고 쉽게' 설명해줘.

[Hard Rules]
- 반드시 아래 [Context] 기사 내용에 기반해 답변해. (추측/상상 금지)
- 기사에 없는 숫자, 사실, 원인 단정은 하지 말고 "기사에서 확인되지 않는다"라고 말해.
- 투자 판단(매수/매도 지시)처럼 들리지 않게 표현해. (조언이 아니라 이해 도움)
- 이해를 돕기 위해 비유/예시를 쓰되, 사실처럼 단정하지 말고 "예를 들면"으로 표현해.
- 초보가 읽기 쉽게 문장 짧게, 어려운 표현 금지.
- 반드시 한국어로만 답변하세요.
- 머리말(예: "Here is ...", "요약:") 없이 바로 답변을 시작하세요.

[Context]
{context}

[User Question]
{question}

[Response Format]
1) 한 줄 요약: (초등학생도 이해할 말로 1문장)
2) 핵심 내용 3가지: (불릿 3개, 쉬운 표현)
3) 왜 중요한가?: (초보 관점에서 '내 계좌에 어떤 영향 가능성?'을 일반론으로)
4) 체크포인트: (추가로 확인하면 좋은 것 2~3개. 기사 기반으로만)
5) 출처: (활용한 기사 제목 + 언론사만 간단히 나열)
"""
        return prompt

    def _build_general_prompt(self, question: str) -> str:
        prompt = f"""
[Role]
너는 '초보 주식 투자자(금융 지식이 거의 없는 사회초년생)'를 돕는 친절한 금융 설명 챗봇이야.

[Hard Rules]
- 뉴스/특정 기사/특정 종목의 최신 이슈를 "봤다"라고 가정하지 말 것.
- 특정 종목/코인의 매수/매도 지시 금지. (이해를 돕는 설명만)
- 모르면 모른다고 말하고, 사용자가 확인할 포인트를 제시해.
- 반드시 한국어로만 답변하세요.
- 머리말 없이 바로 답변하세요.

[User Question]
{question}

[Response Format]
1) 한 줄 결론: (최대한 쉬운 말 1문장)
2) 개념 설명: (초보 눈높이로 4~6문장)
3) 예시: (가상의 간단한 숫자 예시 1개)
4) 주의할 점: (초보가 흔히 하는 오해/리스크 2~3개)
5) 다음 질문 추천: (사용자가 더 물어보면 좋은 질문 2개)
"""
        return prompt

    # ---------------- Main entry ----------------
    def answer(self, question: str) -> Tuple[str, bool]:
        use_db = self.should_use_news_db(question)

        used_db = False
        articles: List[Dict[str, Any]] = []

        if use_db and self.news_db is not None:
            try:
                search_q = question
                if self.enable_query_refine:
                    search_q = self.refine_search_query(question)

                # 점수까지 가져오기
                raw_pairs: List[TypingTuple[Document, float]] = self.news_db.similarity_search_with_score(
                    search_q, k=self.retrieval_k
                )

                if self.debug:
                    top_scores = [round(s, 4) for _d, s in raw_pairs[:10] if s is not None]
                    print("[DB] retrieved_pairs:", len(raw_pairs))
                    print("[DB] top_scores:", top_scores)

                # 임계치 필터
                pairs = self._filter_pairs_by_score(raw_pairs)

                if self.debug:
                    print("[DB] pairs_after_filter:", len(pairs))
                    if pairs:
                        m0 = pairs[0][0].metadata or {}
                        print("[DB] first meta keys:", list(m0.keys()))
                        print("[DB] first link:", (m0.get("link") or "")[:120])

                # 기사 단위 묶기 + best_score 기반 topN
                articles = self._group_pairs_to_articles(pairs, top_articles=self.top_articles)

                if self.debug:
                    print("[DB] articles:", len(articles))
                    for i, a in enumerate(articles[:3], start=1):
                        print(f"[DB] #{i} best_score={a.get('best_score')}, title={a.get('title')[:60]}")

                used_db = len(articles) > 0

            except Exception as e:
                print(f"[WARN] Vector DB search skipped: {e}")
                used_db = False
                articles = []

        prompt = self._build_news_prompt(question, articles) if used_db else self._build_general_prompt(question)
        answer = self.answer_llm.invoke(prompt).content
        return answer, used_db
