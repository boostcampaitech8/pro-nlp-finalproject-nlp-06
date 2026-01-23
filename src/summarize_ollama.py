from __future__ import annotations

import json
import re
from typing import Optional

import requests


def ollama_healthcheck(base_url: str = "http://localhost:11434", timeout: int = 5) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _clean_summary(text: str) -> str:
    """
    모델이 자꾸 붙이는 머리말/불필요한 장식 제거
    """
    if not text:
        return ""

    s = text.strip()

    # 코드블럭 제거
    s = re.sub(r"^```[\s\S]*?\n", "", s).strip()
    s = re.sub(r"\n```$", "", s).strip()

    # 흔한 영어 머리말 제거
    patterns = [
        r"^Here is a summary of the article in 3-5 sentences:\s*",
        r"^Here is a summary in 3-5 sentences:\s*",
        r"^Summary:\s*",
        r"^요약\s*[:：]\s*",
        r"^다음은 .*?요약.*?[:：]\s*",
    ]
    for p in patterns:
        s = re.sub(p, "", s, flags=re.IGNORECASE).strip()

    # 맨 앞에 불필요한 따옴표/불릿 정리
    s = s.strip(" \n\t\"'")

    # 여러 줄이면 1줄로 과하게 합치지 말고, 공백만 정리
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


def summarize_with_ollama(
    text: str,
    title: str = "",
    base_url: str = "http://localhost:11434",
    model: str = "llama3",
    timeout: int = 90,
) -> str:
    """
    한국어로 3~10문장 요약 생성.
    - 머리말/접두문(Here is..., Summary:) 금지
    - 결과 후처리로 접두문 제거
    """
    base_url = base_url.rstrip("/")
    text = (text or "").strip()
    title = (title or "").strip()

    if not text:
        return ""

    system = (
        "너는 한국어 경제 뉴스 요약가다. "
        "항상 한국어로만 답하고, 불필요한 머리말/접두문(예: 'Here is a summary...' 또는 '요약:')을 절대 붙이지 마라. "
        "요약문만 3~10문장으로 출력해라."
    )

    user = (
        f"제목: {title}\n\n"
        "다음 뉴스 본문을 한국어로 3~10문장으로 요약해줘.\n"
        "- 반드시 한국어\n"
        "- 머리말/접두문 없이 요약문만\n"
        "- 과한 추측 금지, 기사에 있는 내용만\n\n"
        f"[본문]\n{text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        out = (data.get("message", {}) or {}).get("content", "")
        return _clean_summary(out)
    except Exception as e:
        print("요약 요청 에러:", e)
        return ""
