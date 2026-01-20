import csv
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ===== 설정 =====
PERSIST_DIR = "./chroma_news"
COLLECTION_NAME = "naver_finance_news_chunks"
EMBEDDING_MODEL = "nomic-embed-text"
OUTPUT_CSV = "chroma_articles_latest.csv"

# 기사 기준 최신 N개 (1~100)
MAX_ARTICLES = 50

# 기사당 합칠 최대 chunk 수
MAX_CHUNKS_PER_ARTICLE = 50
# =================


def kw_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join([str(x).strip() for x in v if str(x).strip()])
    return str(v).strip()


def parse_date(date_str: str) -> datetime:
    if not date_str:
        return datetime(1970, 1, 1)

    s = str(date_str).strip()
    s = s.replace(".", "-").replace("/", "-")
    s = re.sub(r"\s+", " ", s)

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass

    m = re.search(r"(\d{4})(\d{2})(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    return datetime(1970, 1, 1)


embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embedding,
)

collection = db._collection
total = collection.count()
print("총 chunk 수:", total)

# 전체 가져오기 (기사 단위 합치려면 limit 없이)
results = collection.get(
    include=["documents", "metadatas"]
)

docs: List[str] = results.get("documents", []) or []
metas: List[Dict[str, Any]] = results.get("metadatas", []) or []
ids: List[str] = results.get("ids", []) or []

# ---------- link(기사) 단위로 그룹핑 ----------
by_link: Dict[str, Dict[str, Any]] = {}

for i in range(len(ids)):
    text = docs[i] if i < len(docs) else ""
    m = metas[i] or {}

    link = (m.get("link") or "").strip()
    if not link:
        continue

    if link not in by_link:
        by_link[link] = {
            "link": link,
            "title": m.get("title", ""),
            "press": m.get("press", ""),
            "date": m.get("date", ""),
            "keywords": kw_to_str(m.get("keywords", "")),
            "summary": m.get("summary", ""),  
            "chunks": [],
        }

    by_link[link]["chunks"].append(
        {
            "id": ids[i],
            "chunk_index": m.get("chunk_index"),
            "chunk_total": m.get("chunk_total"),
            "text": text,
        }
    )

# chunk_index 기준 정렬 + 텍스트 합치기
articles: List[Dict[str, Any]] = []

for link, a in by_link.items():
    chunks = a["chunks"]

    def _key(c):
        v = c.get("chunk_index")
        try:
            return (0, int(v))
        except Exception:
            return (1, 10**9)

    chunks.sort(key=_key)

    merged_texts = []
    chunk_total = None

    for c in chunks[:MAX_CHUNKS_PER_ARTICLE]:
        t = (c.get("text") or "").strip()
        if t:
            merged_texts.append(t)
        if chunk_total is None and c.get("chunk_total") is not None:
            chunk_total = c.get("chunk_total")

    merged_text = "\n\n".join(merged_texts).strip()

    articles.append(
        {
            "link": link,
            "title": a.get("title", ""),
            "press": a.get("press", ""),
            "date": a.get("date", ""),
            "keywords": a.get("keywords", ""),
            "summary": a.get("summary", ""),   
            "text": merged_text,
            "chunk_count": len(chunks),
            "chunk_total": chunk_total if chunk_total is not None else "",
        }
    )

# ---------- 최신 기사 MAX_ARTICLES개로 제한 ----------
articles.sort(key=lambda x: parse_date(x.get("date", "")), reverse=True)
max_articles = min(max(1, int(MAX_ARTICLES)), 100, len(articles))
articles = articles[:max_articles]

# ---------- CSV 저장 (1행=1기사 + 요약 포함) ----------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "title",
            "press",
            "date",
            "link",
            "keywords",
            "summary",      
            "chunk_count",
            "chunk_total",
            "text",
        ]
    )

    for a in articles:
        writer.writerow(
            [
                a.get("title", ""),
                a.get("press", ""),
                a.get("date", ""),
                a.get("link", ""),
                a.get("keywords", ""),
                a.get("summary", ""),  
                a.get("chunk_count", ""),
                a.get("chunk_total", ""),
                a.get("text", ""),
            ]
        )

print(f"CSV 저장 완료: {OUTPUT_CSV} (articles={len(articles)})")
