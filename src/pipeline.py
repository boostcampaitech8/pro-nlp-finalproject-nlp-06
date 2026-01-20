from __future__ import annotations

import os
from typing import Dict, List, Any

from crawler_naver_finance import crawl_last_hours_raw
from summarize_ollama import summarize_with_ollama, ollama_healthcheck
from keywords import extract_keywords_tfidf, refine_keywords_with_ollama
from chunking import chunk_by_chars
from chroma_store import add_chunked_documents
from chroma_cleanup import cleanup_old_documents
from csv_export import export_articles_csv


def build_chunk_rows_for_article(
    article: Dict,
    summary: str,
    keywords: List[str],
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    link = (article.get("link") or "").strip()
    content = (article.get("content") or "").strip()
    if not link or not content:
        return []

    chunks = chunk_by_chars(content, chunk_size=chunk_size, overlap=overlap)
    chunk_total = len(chunks)
    if chunk_total == 0:
        return []

    # Chroma metadata는 list 불가 -> 문자열로 저장
    keywords_str = ", ".join([k.strip() for k in (keywords or []) if k and k.strip()])

    base_meta = {
        "title": article.get("title", ""),
        "press": article.get("press", ""),
        "date": article.get("date", ""),
        "link": link,
        "summary": summary,
        "keywords": keywords_str,  # str
    }

    rows: List[Dict] = []
    for i, (chunk_text, start_idx, end_idx) in enumerate(chunks):
        rows.append({
            "id": f"{link}#chunk_{i}",
            "document": chunk_text,
            "metadata": {
                **base_meta,
                "chunk_index": i,
                "chunk_total": chunk_total,
                "chunk_start": start_idx,
                "chunk_end": end_idx,
            }
        })

    return rows


def run_pipeline(
    hours: int = 1,
    max_page: int = 10,
    # Chroma
    chroma_dir: str = "./chroma_news",
    chroma_collection: str = "naver_finance_news_chunks",
    # Ollama (요약/키워드 정제용)
    ollama_summary_base_url: str | None = None,
    ollama_summary_model: str | None = None,
    summarize_timeout: int = 90,
    # Ollama (임베딩용) model_ollama.py와 맞추기
    ollama_embed_base_url: str | None = None,
    ollama_embed_model: str | None = None,
    # Keywords
    keyword_top_k: int = 40,   # TF-IDF 후보 개수
    keyword_min_k: int = 1,    # 최종 키워드 최소 개수
    keyword_max_k: int = 20,   # 최종 키워드 최대 개수
    keyword_refine_timeout: int = 60,
    # Chunking
    chunk_size: int = 800,
    overlap: int = 120,
    # Cleanup
    cleanup_days: int | None = 14,
    # CSV
    save_csv: bool = True,
    csv_output_dir: str = "./csv_out",
) -> Dict[str, Any]:
    """
    반환:
      {
        "added_chunks": int,
        "csv_path": str|None
      }
    """

    # 요약/키워드 정제용 Ollama
    summary_base = ollama_summary_base_url or os.getenv("OLLAMA_SUMMARY_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    summary_model = ollama_summary_model or os.getenv("OLLAMA_SUMMARY_MODEL") or os.getenv("OLLAMA_MODEL", "llama3")

    # 임베딩용 Ollama (model_ollama.py에서 embedding_model/base_url과 동일하게 맞춤)
    embed_base = ollama_embed_base_url or os.getenv("OLLAMA_EMBED_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = ollama_embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # healthcheck는 요약 서버 기준으로만 체크 (원하면 embed도 체크 가능)
    if not ollama_healthcheck(base_url=summary_base, timeout=5):
        raise RuntimeError(f"Ollama healthcheck failed: {summary_base}")

    # 1) 크롤링
    articles = crawl_last_hours_raw(hours=hours, max_page=max_page)
    print(f"크롤링 완료: {len(articles)}개(필터 전)")

    enriched_articles: List[Dict] = []
    chunk_rows: List[Dict] = []

    # 2) 기사별 요약/키워드 + chunk 생성
    for a in articles:
        content = (a.get("content") or "").strip()
        if not content:
            continue

        # (1) 요약
        summary = summarize_with_ollama(
            text=content,
            title=a.get("title", ""),
            base_url=summary_base,
            model=summary_model,
            timeout=summarize_timeout,
        )

        # (2) TF-IDF로 후보 키워드 추출
        candidates = extract_keywords_tfidf(content, top_k=keyword_top_k)

        # (3) Ollama로 키워드 정제 (후보 내에서만 선택)
        keywords = refine_keywords_with_ollama(
            title=a.get("title", ""),
            summary=summary,
            candidates=candidates,
            base_url=summary_base,
            model=summary_model,
            min_k=keyword_min_k,
            max_k=keyword_max_k,
            timeout=keyword_refine_timeout,
        )

        # enriched_articles에도 keywords_str을 같이 붙여두면 CSV/디버깅에 편함(선택이지만 추천)
        keywords_str = ", ".join([k.strip() for k in (keywords or []) if k and k.strip()])

        a2 = dict(a)
        a2["summary"] = summary
        a2["keywords"] = keywords           # list (파이썬에서 쓰기 편함)
        a2["keywords_str"] = keywords_str   # str  (CSV/표시용)
        enriched_articles.append(a2)

        chunk_rows.extend(
            build_chunk_rows_for_article(
                article=a2,
                summary=summary,
                keywords=keywords,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    print(f"chunk 생성 완료: {len(chunk_rows)}개 chunk")

    # 3) CSV 저장(기사 단위)
    csv_path = None
    if save_csv:
        # csv_export.py가 keywords(list)도 처리하도록 되어있다면 그대로 OK
        # 아니면 a2["keywords_str"]를 쓰게 csv_export.py를 약간 수정하면 됨
        csv_path = export_articles_csv(enriched_articles, output_dir=csv_output_dir, hours=hours)
        print("CSV 저장:", csv_path)

    # 4) Chroma 저장(chunk 단위)
    added = add_chunked_documents(
        chunk_rows,
        persist_dir=chroma_dir,
        collection_name=chroma_collection,
        ollama_base_url=embed_base,     # 임베딩 서버
        ollama_embed_model=embed_model, # embedding_model 통일
    )

    # 5) 오래된 데이터 정리
    if cleanup_days is not None and cleanup_days > 0:
        cleanup_old_documents(
            persist_dir=chroma_dir,
            collection_name=chroma_collection,
            days=cleanup_days,
            ollama_base_url=embed_base,     # 임베딩 함수 동일하게 맞추는 게 안전
            ollama_embed_model=embed_model,
        )

    return {"added_chunks": added, "csv_path": csv_path}


if __name__ == "__main__":
    out = run_pipeline(
        hours=1,
        max_page=10,
        chroma_dir="./chroma_news",
        chroma_collection="naver_finance_news_chunks",
        chunk_size=800,
        overlap=120,
        cleanup_days=14,
        save_csv=True,
        csv_output_dir="./csv_out",
        # 키워드 옵션
        keyword_top_k=40,
        keyword_min_k=1,
        keyword_max_k=20,
        keyword_refine_timeout=60,
        # 필요하면 요약/임베딩 서버 분리해서 넣기
        # ollama_summary_base_url="http://localhost:11434",
        # ollama_embed_base_url="http://localhost:11434",
        # ollama_embed_model="nomic-embed-text",
    )
    print(out)
