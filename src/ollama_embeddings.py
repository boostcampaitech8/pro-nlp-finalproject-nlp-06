from __future__ import annotations

from typing import List, Union
import requests


class OllamaEmbeddingFunction:
    """
    Chroma EmbeddingFunction 인터페이스(중요):
      - name(self) -> str
      - __call__(self, input) -> List[List[float]]
        (파라미터 이름이 반드시 input 이어야 검증 통과)
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def name(self) -> str:
        return f"ollama:{self.model}@{self.base_url}"

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        # Chroma는 보통 List[str]를 주지만, 혹시 str이 오면 리스트로 감싼다
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        embeddings: List[List[float]] = []

        for t in texts:
            t = (t or "").strip()
            if not t:
                embeddings.append([])
                continue

            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=60,
            )
            r.raise_for_status()
            embeddings.append(r.json()["embedding"])

        return embeddings
