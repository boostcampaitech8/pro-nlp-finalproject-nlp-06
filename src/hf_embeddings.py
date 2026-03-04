from __future__ import annotations

from typing import List, Union
import requests
from langchain_huggingface import HuggingFaceEmbeddings


class HuggingFaceEmbeddingFunction:
    """
    Chroma EmbeddingFunction 인터페이스:
    로컬 HuggingFace 모델을 사용하여 임베딩을 생성합니다.
    """

    #임베딩 모델 수정
    def __init__(self, model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko", device: str = "cuda"):
        """
        model_name: 사용할 HuggingFace 모델명
        device: 'cuda' (GPU) 또는 'cpu'
        """
        # GPU 사용 가능 여부 자동 체크 (cuda가 기본이지만 없으면 cpu)
        import torch
        actual_device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        print(f"[INFO] 로컬 임베딩 모델 로드 중: {model_name} (장치: {actual_device})")
        
        self.model_name = model_name
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': actual_device},
            encode_kwargs={'normalize_embeddings': True} # 코사인 유사도를 위해 정규화
        )

    def name(self) -> str:
        return f"hf:{self.model_name}"

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        ChromaDB 호출 규격에 맞춰 임베딩 생성 (파라미터명 'input' 유지 필수)
        """
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        # 텍스트가 비어있는 경우 예외 처리
        valid_texts = [t.strip() if t.strip() else "empty" for t in texts]
        
        # 임베딩 생성 (HuggingFaceEmbeddings의 embed_documents 활용)
        embeddings = self.embedder.embed_documents(valid_texts)
        
        return embeddings
