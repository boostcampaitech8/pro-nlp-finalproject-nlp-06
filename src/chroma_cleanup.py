from datetime import datetime, timedelta
import pendulum

import chromadb
from .hf_embeddings import HuggingFaceEmbeddingFunction

KST = pendulum.timezone("Asia/Seoul")


def parse_kst_datetime(text: str):
    """
    '2026-01-19 11:40:14' -> KST datetime
    """
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=KST)
    except Exception:
        return None


def cleanup_old_documents(
    persist_dir: str,
    collection_name: str,
    days: int = 14,
    # [변경] HuggingFace 모델명 전달 (Ollama 인자 제거)
    embedding_model_name: str = "jhgan/ko-sroberta-multitask",
):
    """
    설정된 기간(days)보다 오래된 뉴스를 ChromaDB에서 삭제합니다.
    """
    # 1. 클라이언트 및 컬렉션 로드 (동일한 임베딩 함수 사용 필수)
    client = chromadb.PersistentClient(path=persist_dir)
    ef = HuggingFaceEmbeddingFunction(model_name=embedding_model_name)
    
    try:
        col = client.get_collection(collection_name, embedding_function=ef)
    except Exception as e:
        print(f"🧹 [Cleanup] 컬렉션을 찾을 수 없습니다: {e}")
        return 0
    now = datetime.now(tz=KST)
    cutoff = now - timedelta(days=days)

    print(f"🧹 정리 기준: {cutoff.strftime('%Y-%m-%d %H:%M:%S %Z')} 이전")

    # ids는 include에 넣지 않는다. (기본 반환)
    data = col.get(include=["metadatas"])

    if not data.get("ids"):
        print("삭제할 데이터 없음")
        return 0

    delete_ids = []
    for _id, meta in zip(data["ids"], data.get("metadatas", [])):
        date_str = (meta or {}).get("date", "")
        article_time = parse_kst_datetime(date_str)
        if article_time and article_time < cutoff:
            delete_ids.append(_id)

    if not delete_ids:
        print("삭제할 오래된 문서 없음")
        return 0

    col.delete(ids=delete_ids)
    try:
        client.persist()
    except Exception:
        pass

    print(f"삭제 완료: {len(delete_ids)}개 chunk 제거")
    return len(delete_ids)

