from __future__ import annotations

import os
import requests # vLLM 헬스체크용
from pathlib import Path
from typing import Dict, List, Any

from .crawler_naver_finance import crawl_last_hours_raw
from .summarize_vllm import summarize_with_vllm, vllm_healthcheck
from .keywords import extract_keywords_tfidf, refine_keywords_with_vllm
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
        "date": article.get("date", ""), # 원본 날짜 문자열
        "date_iso": article.get("date_iso", ""), # 표준화된 문자열 날짜
        "date_ts": int(article.get("date_ts", 0) or 0), # 정렬, 비교용 숫자 타임스탬프
        "link": link,
        "summary": summary,
        "keywords": keywords_str,
    }
    date_str = article.get("date", "").strip()
    title = article.get("title", "").strip()
        
    rows: List[Dict] = []
    for i, (chunk_text, start_idx, end_idx) in enumerate(chunks):

        for chunk_text, start_idx, end_idx in chunks:
            # ✅ 청크 앞에 날짜와 제목 붙이기
            header_parts = []
            if date_str:
                header_parts.append(f"[{date_str}]")
            if title:
                header_parts.append(f"[제목: {title}]")
            
            if header_parts:
                chunk_with_header = " ".join(header_parts) + "\n" + chunk_text
            else:
                chunk_with_header = chunk_text
                
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

    
    # [핵심 수정] 부모 디렉토리 아래에 'News_chroma_db' 폴더를 명시적으로 추가
    # chroma_dir = str(Path(base_chroma_dir) / "News_chroma_db")
    # 최종 news DB 경로
    chroma_dir = resolve_under_project(
    os.getenv("CHROMA_DIR", "Chroma_db/News_chroma_db")
    )

    csv_output_dir = resolve_under_project(
        os.getenv("CSV_DIR", "csv_out")
    )
    chroma_collection = os.getenv(
        "CHROMA_COLLECTION", "naver_finance_news_chunks"
    )

    # Ollama
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    vllm_model = os.getenv("VLLM_MODEL", "skt/A.X-4.0-Light") #수정 
    vllm_api_key = os.getenv("VLLM_API_KEY", "vllm-key")

    # Embedding 설정 (HuggingFace 로컬 모델)
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask") #수정

    print("[pipeline] PROJECT_ROOT:", PROJECT_ROOT)
    print("[pipeline] chroma_dir:", chroma_dir)
    print("[pipeline] csv_output_dir:", csv_output_dir)
    print("[pipeline] chroma_collection:", chroma_collection)
    print(f"[pipeline] Using vLLM Model: {vllm_model}")
    print(f"[pipeline] Using Embedding: {embedding_model_name}")

    if not vllm_healthcheck(base_url=vllm_base_url, api_key=vllm_api_key):
        raise RuntimeError(f"vLLM healthcheck failed: {vllm_base_url}")

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

        summary = summarize_with_vllm(
            text=content,
            title=a.get("title", ""),
            base_url=vllm_base_url,
            model=vllm_model,
            api_key=vllm_api_key,
            timeout=90,
        )

        # 키워드 정제 (vLLM 사용)
        candidates = extract_keywords_tfidf(content, top_k=40)
        keywords = refine_keywords_with_vllm(
            title=a.get("title", ""),
            summary=summary,
            candidates=candidates,
            base_url=vllm_base_url,
            model=vllm_model,
            api_key=vllm_api_key,
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
        embedding_model_name=embedding_model_name, # 인자명 변경
    )

    # 5) Cleanup
    if cleanup_days and cleanup_days > 0:
        cleanup_old_documents(
            persist_dir=chroma_dir,
            collection_name=chroma_collection,
            days=cleanup_days,
            embedding_model_name=embedding_model_name, # 인자명 변경
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
