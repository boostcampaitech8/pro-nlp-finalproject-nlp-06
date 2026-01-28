from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any

from .crawler_naver_finance import crawl_last_hours_raw
from .summarize_ollama import summarize_with_ollama, ollama_healthcheck
from .keywords import extract_keywords_tfidf, refine_keywords_with_ollama
from .chunking import chunk_by_chars
from .chroma_store import add_chunked_documents
from .chroma_cleanup import cleanup_old_documents
from .csv_export import export_articles_csv


# ----------------------------
# PROJECT_ROOT 기준 통일
# ----------------------------
THIS_FILE = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = THIS_FILE.parents[1]  # project/

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
)


def resolve_under_project(p: str) -> str:
    """상대경로는 PROJECT_ROOT 기준으로만 해석"""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


# ----------------------------
# Chunk builder
# ----------------------------
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
    if not chunks:
        return []

    keywords_str = ", ".join(k.strip() for k in keywords if k and k.strip())

    base_meta = {
        "title": article.get("title", ""),
        "press": article.get("press", ""),
        "date": article.get("date", ""),
        "date_iso": article.get("date_iso", ""),
        "date_ts": int(article.get("date_ts", 0) or 0),
        "link": link,
        "summary": summary,
        "keywords": keywords_str,
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
                    "chunk_total": len(chunks),
                    "chunk_start": start_idx,
                    "chunk_end": end_idx,
                },
            }
        )
    return rows


# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(
    hours: int = 1,
    max_page: int = 10,
    chunk_size: int = 800,
    overlap: int = 120,
    cleanup_days: int | None = 14,
    save_csv: bool = True,
) -> Dict[str, Any]:

    # env 우선
    chroma_dir = resolve_under_project(
        os.getenv("CHROMA_DIR", "chroma_news")
    )
    csv_output_dir = resolve_under_project(
        os.getenv("CSV_DIR", "csv_out")
    )
    chroma_collection = os.getenv(
        "CHROMA_COLLECTION", "naver_finance_news_chunks"
    )

    # Ollama
    summary_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    summary_model = os.getenv("OLLAMA_MODEL", "llama3")
    embed_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    print("[pipeline] PROJECT_ROOT:", PROJECT_ROOT)
    print("[pipeline] chroma_dir:", chroma_dir)
    print("[pipeline] csv_output_dir:", csv_output_dir)
    print("[pipeline] chroma_collection:", chroma_collection)

    if not ollama_healthcheck(base_url=summary_base, timeout=5):
        raise RuntimeError(f"Ollama healthcheck failed: {summary_base}")

    # 1) 크롤링
    articles = crawl_last_hours_raw(hours=hours, max_page=max_page)
    print(f"크롤링 완료: {len(articles)}개(필터 전)")

    enriched_articles: List[Dict] = []
    chunk_rows: List[Dict] = []

    # 2) 뉴스 요약 + 청크 
    for a in articles:
        content = (a.get("content") or "").strip()
        if not content:
            continue

        summary = summarize_with_ollama(
            text=content,
            title=a.get("title", ""),
            base_url=summary_base,
            model=summary_model,
            timeout=90,
        )

        candidates = extract_keywords_tfidf(content, top_k=40)
        keywords = refine_keywords_with_ollama(
            title=a.get("title", ""),
            summary=summary,
            candidates=candidates,
            base_url=summary_base,
            model=summary_model,
            min_k=1,
            max_k=20,
            timeout=60,
        )

        a2 = dict(a)
        a2["summary"] = summary
        a2["keywords"] = keywords
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

    print(f"chunk 생성 완료: {len(chunk_rows)}개")

    # 3) CSV 
    csv_path = None
    if save_csv:
        csv_path = export_articles_csv(
            enriched_articles,
            output_dir=csv_output_dir,
            hours=hours,
        )
        print("CSV 저장:", csv_path)

    # 4) Chroma
    added = add_chunked_documents(
        chunk_rows,
        persist_dir=chroma_dir,
        collection_name=chroma_collection,
        ollama_base_url=embed_base,
        ollama_embed_model=embed_model,
    )

    # 5) Cleanup
    if cleanup_days and cleanup_days > 0:
        cleanup_old_documents(
            persist_dir=chroma_dir,
            collection_name=chroma_collection,
            days=cleanup_days,
            ollama_base_url=embed_base,
            ollama_embed_model=embed_model,
        )

    return {"added_chunks": added, "csv_path": csv_path}


if __name__ == "__main__":
    out = run_pipeline(
        hours=int(os.getenv("HOURS", "1")),
        max_page=int(os.getenv("MAX_PAGE", "10")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        cleanup_days=int(os.getenv("CLEANUP_DAYS", "14")),
        save_csv=os.getenv("SAVE_CSV", "true").lower() == "true",
    )
    print(out)
