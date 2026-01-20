from __future__ import annotations

import re
from typing import List, Tuple


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
