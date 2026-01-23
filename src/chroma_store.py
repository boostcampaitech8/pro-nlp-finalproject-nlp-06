# chroma_store.py
from __future__ import annotations

from typing import Dict, List, Tuple
import math

import chromadb
from chromadb.errors import DuplicateIDError

from .ollama_embeddings import OllamaEmbeddingFunction

DEFAULT_PERSIST_DIR = "./chroma_news"
DEFAULT_COLLECTION = "naver_finance_news_chunks"


def get_collection(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    # Ollama embeddings
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "nomic-embed-text",
):
    embedding_fn = OllamaEmbeddingFunction(base_url=ollama_base_url, model=ollama_embed_model)

    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"source": "naver_finance_mainnews", "embeddings": "ollama", "embed_model": ollama_embed_model},
    )
    return client, col


def _batched(lst: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        return [lst]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def _fetch_existing_ids(col, ids: List[str], batch_size: int = 200) -> set:
    """
    컬렉션에 이미 존재하는 id들을 set으로 반환
    """
    existing = set()
    if not ids:
        return existing

    for chunk in _batched(ids, batch_size):
        try:
            got = col.get(ids=chunk, include=[])  # include=[] => ids만(버전에 따라 ids만 올 수 있음)
            for _id in (got.get("ids") or []):
                existing.add(_id)
        except Exception:
            # get(ids=...)가 환경에 따라 실패할 수 있으니, 실패하면 일단 넘어감
            pass

    return existing


def add_chunked_documents(
    chunked_rows: List[Dict],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    # Ollama embeddings
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "nomic-embed-text",
) -> int:
    """
    chunked_rows item 예:
    {
      "id": "...#chunk_0",
      "document": "chunk text",
      "metadata": {...}
    }

    DuplicateIDError 방지 포인트
    1) 이번 배치 내에서 id 중복 제거
    2) col.get(ids=[...])를 "벌크"로 조회해서 이미 존재하는 id는 통째로 제외
    3) 그래도 add에서 DuplicateIDError가 뜨면 한번 더 existing filter 후 재시도
    """
    if not chunked_rows:
        print("Chroma: 저장할 chunk 없음", flush=True)
        return 0

    client, col = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
    )

    # 0) 입력 정리 + "배치 내" 중복 제거(가장 중요!)
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    seen_in_batch = set()
    invalid = 0
    dup_in_batch = 0

    for row in chunked_rows:
        _id = (row.get("id") or "").strip()
        doc = (row.get("document") or "").strip()
        meta = row.get("metadata") or {}

        if not _id or not doc:
            invalid += 1
            continue

        if _id in seen_in_batch:
            dup_in_batch += 1
            continue

        seen_in_batch.add(_id)
        ids.append(_id)
        docs.append(doc)
        metas.append(meta)

    if not ids:
        print(f"Chroma: 유효 chunk 없음 (invalid={invalid}, dup_in_batch={dup_in_batch})", flush=True)
        return 0

    # 1) 컬렉션에 이미 있는 id는 벌크로 조회해서 제외
    existing = _fetch_existing_ids(col, ids, batch_size=300)

    if existing:
        new_ids, new_docs, new_metas = [], [], []
        skipped_exist = 0
        for _id, doc, meta in zip(ids, docs, metas):
            if _id in existing:
                skipped_exist += 1
                continue
            new_ids.append(_id)
            new_docs.append(doc)
            new_metas.append(meta)

        ids, docs, metas = new_ids, new_docs, new_metas

        if not ids:
            print(
                f"Chroma: 신규 chunk 없음 "
                f"(exist_skip={skipped_exist}, invalid={invalid}, dup_in_batch={dup_in_batch})",
                flush=True,
            )
            return 0
    else:
        skipped_exist = 0

    # 2) 가능하면 upsert (있으면 가장 튼튼함: 중복으로 죽지 않음)
    #    - 너처럼 주기적으로 돌리는 파이프라인이면 upsert가 안정성이 최고
    if hasattr(col, "upsert"):
        try:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            try:
                client.persist()
            except Exception:
                pass
            print(
                f"Chroma 저장(upsert): {len(ids)}개, exist_skip={skipped_exist}, invalid={invalid}, dup_in_batch={dup_in_batch}",
                flush=True,
            )
            return len(ids)
        except Exception as e:
            print(f"Chroma: upsert 실패 → add로 fallback ({type(e).__name__}: {e})", flush=True)

    # 3) add (DuplicateIDError 나면 한번 더 필터링하고 재시도)
    try:
        col.add(ids=ids, documents=docs, metadatas=metas)
    except DuplicateIDError as e:
        print(f"Chroma: DuplicateIDError 발생 → 재필터 후 재시도 ({e})", flush=True)

        # 다시 existing 조회해서 남아있는 중복 제거
        existing2 = _fetch_existing_ids(col, ids, batch_size=300)
        ids2, docs2, metas2 = [], [], []
        skipped_exist2 = 0
        for _id, doc, meta in zip(ids, docs, metas):
            if _id in existing2:
                skipped_exist2 += 1
                continue
            ids2.append(_id)
            docs2.append(doc)
            metas2.append(meta)

        if not ids2:
            print("Chroma: 재시도에서도 신규 chunk 없음 (모두 중복)", flush=True)
            return 0

        col.add(ids=ids2, documents=docs2, metadatas=metas2)
        ids, docs, metas = ids2, docs2, metas2

    try:
        client.persist()
    except Exception:
        pass

    print(
        f"Chroma 저장(add): {len(ids)}개, exist_skip={skipped_exist}, invalid={invalid}, dup_in_batch={dup_in_batch}",
        flush=True,
    )
    return len(ids)
