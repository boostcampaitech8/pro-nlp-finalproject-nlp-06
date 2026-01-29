from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, urljoin

import requests
from bs4 import BeautifulSoup
import pendulum
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import re
from typing import List, Tuple


BASE_URL = "https://finance.naver.com/news/mainnews.naver"
KST = pendulum.timezone("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def _attach_kst(dt_naive: datetime) -> datetime:
    return dt_naive.replace(tzinfo=KST)


def to_nnews_link(link: str) -> str:
    link_full = urljoin("https://finance.naver.com", link)
    parsed = urlparse(link_full)

    if parsed.netloc == "finance.naver.com" and parsed.path == "/news/news_read.naver":
        qs = parse_qs(parsed.query)
        article_id = qs.get("article_id", [""])[0]
        office_id = qs.get("office_id", [""])[0]
        if article_id and office_id:
            return f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"

    return link_full


def parse_datetime_full_kst(text: str):
    # '2026-01-19 11:40:14'
    if not text:
        return None
    text = text.strip()
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return _attach_kst(dt)
    except Exception:
        return None


def parse_naver_datetime_fallback(text: str, base_now: datetime):
    # fallback: '2026.01.19 13:45' / '01.19 13:45' / '13:45'
    if not text:
        return None

    text = text.strip()
    year = base_now.year
    month = base_now.month
    day = base_now.day

    try:
        if re.match(r"^\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}$", text):
            dt = datetime.strptime(text, "%Y.%m.%d %H:%M")
            return _attach_kst(dt)

        if re.match(r"^\d{2}\.\d{2}\s+\d{2}:\d{2}$", text):
            dt = datetime.strptime(f"{year}.{text}", "%Y.%m.%d %H:%M")
            return _attach_kst(dt)

        if re.match(r"^\d{2}:\d{2}$", text):
            dt = datetime.strptime(
                f"{year}.{month:02d}.{day:02d} {text}",
                "%Y.%m.%d %H:%M",
            )
            return _attach_kst(dt)
    except Exception:
        return None

    return None


def parse_article_time_kst(wdate: str, base_now: datetime):
    dt = parse_datetime_full_kst(wdate)
    if dt:
        return dt
    return parse_naver_datetime_fallback(wdate, base_now=base_now)


def crawl_article_content(url: str, session: requests.Session) -> str:
    url = to_nnews_link(url)

    try:
        res = session.get(url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print("본문 요청 에러:", e, "→", url)
        return ""

    soup = BeautifulSoup(res.text, "html.parser")

    tag = soup.select_one("article#dic_area")
    if tag:
        return tag.get_text("\n", strip=True)

    tag = soup.select_one("#newsct_article")
    if tag:
        return tag.get_text("\n", strip=True)

    tag = soup.select_one("div#dic_area") or soup.select_one("div.article_viewer")
    if tag:
        return tag.get_text("\n", strip=True)

    return ""


def crawl_page(page: int, session: requests.Session, time_limit: datetime, base_now: datetime):
    res = session.get(BASE_URL, params={"page": page}, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    articles = []
    blocks = soup.select("li.block1")

    for block in blocks:
        subject_tag = block.select_one("dd.articleSubject > a")
        if not subject_tag:
            continue

        title = subject_tag.get_text(strip=True)
        link = to_nnews_link(subject_tag.get("href", ""))

        summary = block.select_one("dd.articleSummary")
        press = ""
        wdate = ""

        if summary:
            press_tag = summary.select_one("span.press")
            date_tag = summary.select_one("span.wdate")
            press = press_tag.get_text(strip=True) if press_tag else ""
            wdate = date_tag.get_text(strip=True) if date_tag else ""

        article_time = parse_article_time_kst(wdate, base_now=base_now)

        if article_time and article_time < time_limit:
            return articles, True

        content = crawl_article_content(link, session)
        time.sleep(0.4)

        articles.append({
            "title": title,
            "link": link,
            "press": press,
            "date": wdate,
            "content": content,
        })

    return articles, False


def crawl_last_hours_raw(hours: int = 1, max_page: int = 10):
    """
    현재 시각(KST) 기준 최근 `hours`시간 뉴스 수집.
    """
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours는 1 이상의 정수여야 합니다.")

    now = now_kst()
    time_limit = now - timedelta(hours=hours)

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    all_data = []
    with requests.Session() as session:
        session.headers.update(headers)

        for page in range(1, max_page + 1):
            articles, stop = crawl_page(page, session, time_limit=time_limit, base_now=now)
            all_data.extend(articles)
            if stop:
                break
            time.sleep(0.8)

    return all_data




def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_by_chars(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Tuple[str, int, int]]:
    """
    문자 기반 chunking (Korean 포함 안전).
    반환: [(chunk_text, start_idx, end_idx), ...]

    - chunk_size: 600~1000 권장 (기본 800)
    - overlap: 문맥 유지용 겹침 (기본 120)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: List[Tuple[str, int, int]] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # 끝을 가능한 문장/공백 경계로 살짝 당기기
        if end < n:
            window = text[start:end]
            candidates = [window.rfind(p) for p in [". ", "。", "다.", "다 ", "\n", " "]]
            cut = max(candidates)
            if cut >= int(chunk_size * 0.6):
                end = start + cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks



def save_news_to_vectorstore(news_list: list, db_path: str = "./Chroma_db/News_chroma_db"):
    """
    수집된 뉴스 데이터를 청킹하여 Chroma DB에 저장합니다.
    """
    # 1. 임베딩 모델 설정 (ko-sroberta-multitask)
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}, # GPU가 있다면 'cuda'로 변경 가능
        encode_kwargs={'normalize_embeddings': True}
    )

    documents = []

    for item in news_list:
        content = item.get("content", "")
        if not content:
            continue

        # 2. 기존에 정의한 chunk_by_chars 함수로 청킹 수행
        chunks = chunk_by_chars(content, chunk_size=800, overlap=120)

        for chunk_text, start_idx, end_idx in chunks:
            # 메타데이터에 원본 링크, 제목, 언론사 등을 포함하여 나중에 참조하기 좋게 함
            meta = {
                "title": item.get("title"),
                "link": item.get("link"),
                "press": item.get("press"),
                "date": item.get("date"),
                "start_idx": start_idx,
                "end_idx": end_idx
            }
            documents.append(Document(page_content=chunk_text, metadata=meta))

    if not documents:
        print("저장할 문서가 없습니다.")
        return

    # 3. Chroma DB 생성 및 로컬 저장
    # persist_directory에 경로를 지정하면 자동으로 ./Chroma_db/... 에 저장됩니다.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print(f"성공적으로 {len(documents)}개의 청크를 '{db_path}'에 저장했습니다.")
    return vectorstore

# --- 실행 예시 ---
if __name__ == "__main__":
    # 1. 최근 2시간 뉴스 수집
    raw_news = crawl_last_hours_raw(hours=72)
    
    # 2. 벡터 스토어 저장
    db = save_news_to_vectorstore(raw_news)