from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List
import pendulum

KST = pendulum.timezone("Asia/Seoul")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_articles_csv(
    articles_enriched: List[Dict],
    output_dir: str,
    hours: int,
) -> str:
    """
    1행 = 기사 1개
    columns: title, link, press, date, summary, keywords, content
    """
    ensure_dir(output_dir)

    now = datetime.now(tz=KST)
    filename = f"naver_finance_last_{hours}h_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["title", "link", "press", "date", "summary", "keywords", "content"])

        for a in articles_enriched:
            w.writerow([
                a.get("title", ""),
                a.get("link", ""),
                a.get("press", ""),
                a.get("date", ""),
                a.get("summary", ""),
                ",".join(a.get("keywords", []) or []),
                (a.get("content", "") or "").replace("\r", ""),
            ])

    return output_path
