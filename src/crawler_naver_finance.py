import re
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, urljoin

import requests
from bs4 import BeautifulSoup
import pendulum

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

        date_iso = article_time.isoformat() if article_time else ""
        date_ts = int(article_time.timestamp()) if article_time else 0

        content = crawl_article_content(link, session)
        time.sleep(0.4)

        articles.append({
            "title": title,
            "link": link,
            "press": press,
            "date": wdate,
            "date_iso": date_iso,  # 표준화(선택)
            "date_ts": date_ts,    # 정렬용
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
