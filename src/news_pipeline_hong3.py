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
    if not text:
        return None
    text = text.strip()
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return _attach_kst(dt)
    except Exception:
        return None


def parse_naver_datetime_fallback(text: str, base_now: datetime):
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
        print(f"⚠️  본문 요청 에러: {e} → {url}")
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

    print(f"⚠️  본문 파싱 실패: {url}")
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


def crawl_last_hours_raw(hours: int = 1, max_page: int = 50):  # ✅ max_page 증가
    """
    현재 시각(KST) 기준 최근 `hours`시간 뉴스 수집.
    """
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours는 1 이상의 정수여야 합니다.")

    now = now_kst()
    time_limit = now - timedelta(hours=hours)

    print(f"\n📅 수집 기간: {time_limit.strftime('%Y-%m-%d %H:%M')} ~ {now.strftime('%Y-%m-%d %H:%M')} (최근 {hours}시간)")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    all_data = []
    with requests.Session() as session:
        session.headers.update(headers)

        for page in range(1, max_page + 1):
            print(f"🔍 페이지 {page} 크롤링 중...", end=" ")
            articles, stop = crawl_page(page, session, time_limit=time_limit, base_now=now)
            all_data.extend(articles)
            print(f"✅ {len(articles)}개 수집 (누적: {len(all_data)}개)")
            
            if stop:
                print(f"⏹️  시간 제한 도달. 크롤링 종료 (총 {page}페이지)")
                break
            time.sleep(0.8)

    # ✅ 통계 출력
    valid_content = [a for a in all_data if a.get("content", "").strip()]
    empty_content = len(all_data) - len(valid_content)
    
    print(f"\n📊 크롤링 완료:")
    print(f"   - 총 뉴스: {len(all_data)}개")
    print(f"   - 본문 있음: {len(valid_content)}개")
    print(f"   - 본문 없음: {empty_content}개")
    
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
    각 청크 앞에 날짜와 제목을 붙여 컨텍스트를 강화합니다.
    """
    print(f"\n💾 벡터 스토어 저장 시작...")
    
    # 1. 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    documents = []
    news_with_content = 0
    news_without_content = 0
    total_chunks_per_news = []

    for idx, item in enumerate(news_list, 1):
        content = item.get("content", "")
        if not content or not content.strip():
            news_without_content += 1
            continue

        news_with_content += 1
        
        # ✅ 날짜와 제목 정보 추출
        date_str = item.get("date", "").strip()
        title = item.get("title", "").strip()
        
        # 2. 청킹 수행
        chunks = chunk_by_chars(content, chunk_size=800, overlap=120)
        total_chunks_per_news.append(len(chunks))

        if idx <= 3:  # 처음 3개만 상세 로그
            print(f"   [{idx}] [{date_str}] '{title[:40]}...' → {len(chunks)}개 청크 생성")

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
            
            meta = {
                "title": title,
                "link": item.get("link"),
                "press": item.get("press"),
                "date": date_str,
                "start_idx": start_idx,
                "end_idx": end_idx
            }
            documents.append(Document(page_content=chunk_with_header, metadata=meta))

    if not documents:
        print("❌ 저장할 문서가 없습니다.")
        return None

    # 3. Chroma DB 저장
    print(f"\n🔄 ChromaDB에 임베딩 중... (시간이 걸릴 수 있습니다)")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # 4. 통계 출력
    print(f"\n✅ 벡터 스토어 저장 완료!")
    print(f"{'='*60}")
    print(f"📰 뉴스 통계:")
    print(f"   - 본문 있는 뉴스: {news_with_content}개")
    print(f"   - 본문 없는 뉴스: {news_without_content}개")
    print(f"\n📦 청크 통계:")
    print(f"   - 총 청크 수: {len(documents)}개")
    if total_chunks_per_news:
        avg_chunks = sum(total_chunks_per_news) / len(total_chunks_per_news)
        print(f"   - 평균 청크/뉴스: {avg_chunks:.1f}개")
        print(f"   - 최소 청크: {min(total_chunks_per_news)}개")
        print(f"   - 최대 청크: {max(total_chunks_per_news)}개")
    print(f"\n💾 저장 위치: {os.path.abspath(db_path)}")
    print(f"{'='*60}\n")
    
    return vectorstore



# --- 실행 예시 ---
if __name__ == "__main__":
    print("="*60)
    print("🚀 네이버 금융 뉴스 크롤러 & 벡터DB 저장")
    print("="*60)
    
    # 1. 최근 72시간 뉴스 수집
    raw_news = crawl_last_hours_raw(hours=72, max_page=50)
    
    # 2. 벡터 스토어 저장
    db = save_news_to_vectorstore(raw_news)
    
    print("\n✨ 모든 작업 완료!")