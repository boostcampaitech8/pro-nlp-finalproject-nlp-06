from __future__ import annotations

from typing import Dict, List

import chromadb

from ollama_embeddings import OllamaEmbeddingFunction

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
    """
    if not chunked_rows:
        print("Chroma: 저장할 chunk 없음")
        return 0

    client, col = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
    )

    ids, docs, metas = [], [], []
    skipped = 0

    for row in chunked_rows:
        _id = (row.get("id") or "").strip()
        doc = (row.get("document") or "").strip()
        meta = row.get("metadata") or {}

        if not _id or not doc:
            continue

        # 중복 스킵 (id 기준)
        try:
            got = col.get(ids=[_id])
            if got and got.get("ids"):
                skipped += 1
                continue
        except Exception:
            pass

        ids.append(_id)
        docs.append(doc)
        metas.append(meta)

    if not ids:
        print(f"Chroma: 신규 chunk 없음 (중복 {skipped}개 스킵)")
        return 0

    col.add(ids=ids, documents=docs, metadatas=metas)
    try:
        client.persist()
    except Exception:
        pass

    print(f"Chroma 저장: {len(ids)}개 chunk 추가, {skipped}개 중복 스킵")
    return len(ids)
