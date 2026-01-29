import re
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, urljoin

import requests
from bs4 import BeautifulSoup
import pendulum

KST = pendulum.timezone("Asia/Seoul")

# 수집할 리스트 URL들 (mainnews + 원하는 섹션들)
LIST_SOURCES = [
    "https://finance.naver.com/news/mainnews.naver",
    "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401",
    "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402",
    "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403",
    "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404",
    "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406",
    "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=429",
]


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def _attach_kst(dt_naive: datetime) -> datetime:
    return dt_naive.replace(tzinfo=KST)


def to_nnews_link(link: str) -> str:
    """
    finance.naver.com/news/news_read.naver?... 를 n.news.naver.com/mnews/article/{office}/{article} 로 변환
    - article_id / office_id (snake) + articleId / officeId (camel) 모두 대응
    """
    link_full = urljoin("https://finance.naver.com", link)
    parsed = urlparse(link_full)

    if parsed.netloc == "finance.naver.com" and parsed.path == "/news/news_read.naver":
        qs = parse_qs(parsed.query)
        article_id = (qs.get("article_id") or qs.get("articleId") or [""])[0]
        office_id = (qs.get("office_id") or qs.get("officeId") or [""])[0]
        if article_id and office_id:
            return f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"

    return link_full


def parse_datetime_full_kst(text: str):
    """
    지원 포맷
    - 2026-01-29 14:14:02 (초 있음)
    - 2026-01-29 12:00 (초 없음)
    """
    if not text:
        return None
    text = text.strip()

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(text, fmt)
            return _attach_kst(dt)
        except ValueError:
            continue

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


def _merge_query(url: str, extra_params: dict) -> str:
    """
    url에 이미 querystring이 있어도(extra: page 등) 안전하게 합쳐서 새 url을 만든다.
    """
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    flat = {k: (v[-1] if isinstance(v, list) else v) for k, v in q.items()}
    for k, v in extra_params.items():
        flat[k] = str(v)
    new_query = urlencode(flat, doseq=False)
    return urlunparse(parsed._replace(query=new_query))


def _normalize_url_for_dedupe(url: str) -> str:
    """
    중복 제거용 canonical URL:
    - finance 링크가 들어오면 n.news 링크로 변환해서 통일
    - 그 외는 절대 URL로 통일
    """
    return to_nnews_link(url)


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


def crawl_page(
    list_url: str,
    page: int,
    session: requests.Session,
    time_limit: datetime,
    base_now: datetime,
):
    url = _merge_query(list_url, {"page": page})

    res = session.get(url, timeout=10)
    res.raise_for_status()

    # 인코딩 이슈 방지
    if not res.encoding or res.encoding.lower() == "iso-8859-1":
        res.encoding = res.apparent_encoding

    # 실제 요청된 URL 확인 (리다이렉트/쿼리 포함)
    print(f"[DEBUG] fetched = {res.url}", flush=True)

    soup = BeautifulSoup(res.text, "html.parser")
    articles = []

    # main content 영역으로 좁혀서(사이드바 링크 오염 방지)
    root = soup.select_one("#contentarea_left") or soup

    # ------------------------------------------------------------
    # 1) mainnews 스타일: li.block1
    # ------------------------------------------------------------
    blocks = root.select("li.block1")
    if blocks:
        for block in blocks:
            subject_tag = block.select_one("dd.articleSubject > a")
            if not subject_tag:
                continue

            title = subject_tag.get_text(strip=True)
            href = (subject_tag.get("href") or "").strip()
            if not href:
                continue

            link = to_nnews_link(href)

            summary = block.select_one("dd.articleSummary")
            press = ""
            wdate = ""
            if summary:
                press_tag = summary.select_one("span.press")
                date_tag = summary.select_one("span.wdate")
                press = press_tag.get_text(strip=True) if press_tag else ""
                wdate = date_tag.get_text(strip=True) if date_tag else ""

            article_time = parse_article_time_kst(wdate, base_now=base_now)

            # 시간 제한보다 오래된 기사가 나오면: 이 소스는 여기서 중단
            if article_time and article_time < time_limit:
                return articles, True

            date_iso = article_time.isoformat() if article_time else ""
            date_ts = int(article_time.timestamp()) if article_time else 0

            print(f"[DEBUG] article = {link} | {title} | wdate='{wdate}' | parsed={date_iso}", flush=True)

            content = crawl_article_content(link, session)
            time.sleep(0.4)

            articles.append(
                {
                    "title": title,
                    "link": link,
                    "press": press,
                    "date": wdate,
                    "date_iso": date_iso,
                    "date_ts": date_ts,
                    "content": content,
                    "source": list_url,
                }
            )

        return articles, False

    # ------------------------------------------------------------
    # 2) news_list 스타일: dd.articleSubject > a (li.block1이 없어도 잡히는 케이스)
    # ------------------------------------------------------------
    subject_anchors = root.select("dd.articleSubject > a")
    if subject_anchors:
        for a in subject_anchors:
            title = a.get_text(strip=True)
            href = (a.get("href") or "").strip()
            if not title or not href:
                continue

            link = to_nnews_link(href)

            press = ""
            wdate = ""
            dl = a.find_parent("dl")
            if dl:
                summary = dl.select_one("dd.articleSummary")
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

            print(f"[DEBUG] article = {link} | {title} | wdate='{wdate}' | parsed={date_iso}", flush=True)

            content = crawl_article_content(link, session)
            time.sleep(0.4)

            articles.append(
                {
                    "title": title,
                    "link": link,
                    "press": press,
                    "date": wdate,
                    "date_iso": date_iso,
                    "date_ts": date_ts,
                    "content": content,
                    "source": list_url,
                }
            )

        return articles, False

    # ------------------------------------------------------------
    # 3) table.type5 (옛 스타일/다른 모드)
    # ------------------------------------------------------------
    rows = root.select("table.type5 tr")
    if rows:
        for row in rows:
            a = row.select_one("td.title a")
            if not a:
                continue

            title = a.get_text(strip=True)
            href = (a.get("href") or "").strip()
            if not href:
                continue

            link = to_nnews_link(href)

            press_tag = row.select_one("td.info")
            date_tag = row.select_one("td.date")
            press = press_tag.get_text(strip=True) if press_tag else ""
            wdate = date_tag.get_text(strip=True) if date_tag else ""

            article_time = parse_article_time_kst(wdate, base_now=base_now)
            if article_time and article_time < time_limit:
                return articles, True

            date_iso = article_time.isoformat() if article_time else ""
            date_ts = int(article_time.timestamp()) if article_time else 0

            print(f"[DEBUG] article = {link} | {title} | wdate='{wdate}' | parsed={date_iso}", flush=True)

            content = crawl_article_content(link, session)
            time.sleep(0.4)

            articles.append(
                {
                    "title": title,
                    "link": link,
                    "press": press,
                    "date": wdate,
                    "date_iso": date_iso,
                    "date_ts": date_ts,
                    "content": content,
                    "source": list_url,
                }
            )

        return articles, False

    # ------------------------------------------------------------
    # 4) 최후 fallback: content 영역에서 news_read 링크 전부 긁기
    # ------------------------------------------------------------
    link_tags = root.select('a[href*="news_read.naver"]')
    if not link_tags:
        print(f"[DEBUG] NO parsable nodes: url={url}", flush=True)
        return articles, False

    seen_href = set()
    for a in link_tags:
        href = (a.get("href") or "").strip()
        title = a.get_text(" ", strip=True)
        if not href or not title or href in seen_href:
            continue
        seen_href.add(href)

        link = to_nnews_link(href)

        # press/date는 못 찾을 수 있으니 빈값 허용
        press = ""
        wdate = ""
        article_time = parse_article_time_kst(wdate, base_now=base_now)

        date_iso = article_time.isoformat() if article_time else ""
        date_ts = int(article_time.timestamp()) if article_time else 0

        # 시간이 끝내 안 잡히면, 안전하게 스킵(시간필터 안 걸려서 오래된 기사 쌓이는 문제 방지)
        if not article_time:
            continue

        print(f"[DEBUG] article = {link} | {title} | wdate='{wdate}' | parsed={date_iso}", flush=True)

        content = crawl_article_content(link, session)
        time.sleep(0.4)

        articles.append(
            {
                "title": title,
                "link": link,
                "press": press,
                "date": wdate,
                "date_iso": date_iso,
                "date_ts": date_ts,
                "content": content,
                "source": list_url,
            }
        )

    return articles, False


def crawl_last_hours_raw(hours: int = 1, max_page: int = 10, sources: list[str] | None = None):
    """
    현재 시각(KST) 기준 최근 `hours`시간 뉴스 수집.
    - sources: 리스트 페이지 URL들(기본: LIST_SOURCES)
    """
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours는 1 이상의 정수여야 합니다.")

    sources = sources or LIST_SOURCES

    now = now_kst()
    time_limit = now - timedelta(hours=hours)

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    all_data: list[dict] = []
    seen: set[str] = set()  # canonical URL로 중복 제거

    with requests.Session() as session:
        session.headers.update(headers)

        for list_url in sources:
            prev_fp = None  # 페이지 반복 감지용 fingerprint

            for page in range(1, max_page + 1):
                print(f"[DEBUG] start = {list_url}, page={page}", flush=True)

                articles, stop = crawl_page(
                    list_url=list_url,
                    page=page,
                    session=session,
                    time_limit=time_limit,
                    base_now=now,
                )

                # 페이지 fingerprint: 상위 10개 링크로 "같은 페이지 반복" 감지
                fp = "|".join(a.get("link", "") for a in articles[:10])
                if fp and fp == prev_fp:
                    print(
                        f"[DEBUG] same page repeated -> break: {list_url} page={page}",
                        flush=True,
                    )
                    break
                prev_fp = fp

                for a in articles:
                    key = _normalize_url_for_dedupe(a.get("link", ""))
                    if key and key not in seen:
                        seen.add(key)
                        all_data.append(a)

                if stop:
                    break

                time.sleep(0.8)

    # 최신순 정렬(선택)
    all_data.sort(key=lambda x: x.get("date_ts", 0), reverse=True)
    return all_data