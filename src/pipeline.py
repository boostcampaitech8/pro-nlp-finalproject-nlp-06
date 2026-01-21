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
        rows.append(
            {
                "id": f"{link}#chunk_{i}",
                "document": chunk_text,
                "metadata": {
                    **base_meta,
                    "chunk_index": i,
                    "chunk_total": chunk_total,
                    "chunk_start": start_idx,
                    "chunk_end": end_idx,
                },
            }
        )

    return rows


def _abs_path(p: str) -> str:
    """상대경로가 들어오면 현재 실행 위치 기준으로 달라져서 위험 -> 절대경로로 고정"""
    return os.path.abspath(p)


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
    # Ollama (임베딩용)
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

    # (중요) env가 있으면 env를 최우선으로 사용해 경로를 "한 곳"으로 고정
    chroma_dir = os.getenv("CHROMA_DIR", chroma_dir)
    csv_output_dir = os.getenv("CSV_DIR", csv_output_dir)

    chroma_dir = _abs_path(chroma_dir)
    csv_output_dir = _abs_path(csv_output_dir)

    # 요약/키워드 정제용 Ollama
    summary_base = (
        ollama_summary_base_url
        or os.getenv("OLLAMA_SUMMARY_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    summary_model = (
        ollama_summary_model
        or os.getenv("OLLAMA_SUMMARY_MODEL")
        or os.getenv("OLLAMA_MODEL", "llama3")
    )

    # 임베딩용 Ollama
    embed_base = (
        ollama_embed_base_url
        or os.getenv("OLLAMA_EMBED_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    embed_model = (
        ollama_embed_model
        or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )

    print("[pipeline] chroma_dir:", chroma_dir)
    print("[pipeline] csv_output_dir:", csv_output_dir)
    print("[pipeline] chroma_collection:", chroma_collection)
    print("[pipeline] summary_base:", summary_base, "summary_model:", summary_model)
    print("[pipeline] embed_base:", embed_base, "embed_model:", embed_model)

    # healthcheck는 요약 서버 기준으로만 체크
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

        keywords_str = ", ".join([k.strip() for k in (keywords or []) if k and k.strip()])

        a2 = dict(a)
        a2["summary"] = summary
        a2["keywords"] = keywords           # list
        a2["keywords_str"] = keywords_str   # str
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
        csv_path = export_articles_csv(enriched_articles, output_dir=csv_output_dir, hours=hours)
        print("CSV 저장:", csv_path)

    # 4) Chroma 저장(chunk 단위)
    added = add_chunked_documents(
        chunk_rows,
        persist_dir=chroma_dir,
        collection_name=chroma_collection,
        ollama_base_url=embed_base,
        ollama_embed_model=embed_model,
    )

    # 5) 오래된 데이터 정리
    if cleanup_days is not None and cleanup_days > 0:
        cleanup_old_documents(
            persist_dir=chroma_dir,
            collection_name=chroma_collection,
            days=cleanup_days,
            ollama_base_url=embed_base,
            ollama_embed_model=embed_model,
        )

    return {"added_chunks": added, "csv_path": csv_path}


if __name__ == "__main__":
    hours = int(os.getenv("HOURS", "1"))
    max_page = int(os.getenv("MAX_PAGE", "10"))

    out = run_pipeline(
        hours=hours,
        max_page=max_page,
        chroma_dir=os.getenv("CHROMA_DIR", "/data/ephemeral/home/project/chroma_news"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "naver_finance_news_chunks"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        cleanup_days=int(os.getenv("CLEANUP_DAYS", "14")),
        save_csv=os.getenv("SAVE_CSV", "true").lower() == "true",
        csv_output_dir=os.getenv("CSV_DIR", "/data/ephemeral/home/project/csv_out"),
        keyword_top_k=int(os.getenv("KEYWORD_TOP_K", "40")),
        keyword_min_k=int(os.getenv("KEYWORD_MIN_K", "1")),
        keyword_max_k=int(os.getenv("KEYWORD_MAX_K", "20")),
        keyword_refine_timeout=int(os.getenv("KEYWORD_REFINE_TIMEOUT", "60")),
        ollama_summary_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_summary_model=os.getenv("OLLAMA_MODEL", "llama3"),
        ollama_embed_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )
    print(out)
