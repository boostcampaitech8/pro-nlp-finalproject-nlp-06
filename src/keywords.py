import json
import re
import os
from typing import List, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_keywords_tfidf(text: str, top_k: int = 25) -> List[str]:
    """
    단일 문서에서도 동작하는 간단 TF-IDF 후보 키워드 추출.
    - 결과는 '후보'이며, LLM으로 정제하기 좋게 20~40개 정도가 적당.
    """
    text = _normalize(text)
    if not text:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9][가-힣A-Za-z0-9\-\_]+\b",
        max_features=8000,
    )

    try:
        X = vectorizer.fit_transform([text])
    except ValueError:
        return []

    scores = X.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    keywords = [t for t, s in pairs if s > 0][:top_k]
    return [k[:40] for k in keywords]


# -----------------------------
# LLM 키워드 정제 (1~20개)
# -----------------------------
def _safe_parse_json_array(text: str) -> List[str]:
    """
    LLM이 JSON 배열만 준다고 해도, 가끔 앞뒤에 잡문이 붙을 수 있어
    가장 가까운 [...]만 잘라서 파싱 시도.
    """
    if not text:
        return []
    text = text.strip()

    # JSON 배열 구간만 추출
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = []
            for x in data:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        out.append(s)
            return out
    except Exception:
        return []

    return []


def refine_keywords_with_vllm(
    *,
    title: str,
    summary: str,
    candidates: List[str],
    base_url: str = "http://127.0.0.1:8001/v1",
    model: str = "skt/A.X-4.0-Light", #수정 
    api_key: str = "vllm-key",
    min_k: int = 1,
    max_k: int = 20,
    timeout: int = 60,
) -> List[str]:
    """
    TF-IDF 후보(candidates)를 Ollama로 정제해서 [min_k..max_k]개 키워드를 반환.
    - 후보 안에서만 선택하도록 강제
    - JSON 배열만 출력하도록 강제
    - 반환 결과는 dedup + 길이 제한 처리
    """
    # 후보가 없으면 LLM 호출 의미 없음
    candidates = [c.strip() for c in (candidates or []) if c and c.strip()]
    if not candidates:
        return []

    # 안전장치: 후보가 너무 많으면 LLM 입력만 커져서 느려짐
    candidates = candidates[:40]

    # min/max 보정
    min_k = max(1, int(min_k))
    max_k = min(20, int(max_k))
    if min_k > max_k:
        min_k, max_k = 1, 20

    # 후보를 번호 리스트로
    cand_lines = "\n".join([f"{i+1}) {c}" for i, c in enumerate(candidates)])

    system = (
        "너는 경제/금융 뉴스의 키워드 편집기다.\n"
        "규칙:\n"
        f"- 후보 목록에서만 선택해라 (후보에 없는 새 단어를 만들지 마라)\n"
        f"- 불필요하게 일반적인 단어(예: 기자, 이날, 오전, 사진, 보도 등)는 제거해라\n"
        f"- 중복/동의어는 하나로 합쳐라\n"
        f"- 가능한 경우 2단어 표현을 우선하되, 후보에서만 선택해라\n"
        f"- 최종 키워드는 최소 {min_k}개, 최대 {max_k}개\n"
        "- 출력은 JSON 배열만. 예: [\"키워드1\", \"키워드2\"]\n"
        "- 다른 설명 문장을 절대 붙이지 마라"
    )

    user = (
        f"[제목]\n{(title or '').strip()}\n\n"
        f"[요약]\n{(summary or '').strip()}\n\n"
        f"[후보]\n{cand_lines}\n\n"
        f"위 후보에서만 골라 최종 키워드를 JSON 배열로 출력해."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1, # 키워드 추출은 정교해야 하므로 온도를 더 낮춤
        "max_tokens": 512,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    try:
        r = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        raw = data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("vLLM 키워드 정제 요청 에러:", e)
        return candidates[:min_k] # 에러 시 후보 앞부분이라도 반환

    refined = _safe_parse_json_array(raw)

    # 후보 검증 및 필터링
    cand_set = set(candidates)
    seen = set()
    out = []
    
    for k in refined:
        if k in seen:
            continue
        seen.add(k)
        out.append(k[:40])

    # 개수 보정: 부족하면 TF-IDF 후보로 채움(일반어가 섞일 수 있으니 뒤에서 채움)
    if len(out) < min_k:
        for c in candidates:
            if c not in seen:
                out.append(c[:40])
                seen.add(c)
            if len(out) >= min_k:
                break

    # 최대 개수 제한
    out = out[:max_k]
    return out
