from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import pendulum

import chromadb

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
    embedding_model_name: str = "jhgan/ko-sroberta-multitask",
):

    persist_dir = str(Path(persist_dir).expanduser().resolve())

    # 1) 클라이언트 로드
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        # 컬렉션이 없으면 create로 만들지 말고 그냥 종료(의도치 않은 새 컬렉션 방지)
        print(f"🧹 [Cleanup] 컬렉션을 찾을 수 없습니다: {e}")
        return 0

    now = datetime.now(tz=KST)
    cutoff = now - timedelta(days=days)
    print(f"🧹 정리 기준: {cutoff.strftime('%Y-%m-%d %H:%M:%S %Z')} 이전")

    # ids는 include에 넣지 않는다. (기본 반환)
    data = col.get(include=["metadatas"])

    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []

    if not ids:
        print("삭제할 데이터 없음")
        return 0

    delete_ids = []

    for _id, meta in zip(ids, metadatas):
        meta = meta or {}

        # 가장 안정적인 우선순위: date_ts (int) > date_iso > date(원문)
        ts = meta.get("date_ts")
        if ts is not None:
            try:
                article_time = datetime.fromtimestamp(int(ts), tz=KST)
            except Exception:
                article_time = None
        else:
            iso = (meta.get("date_iso") or "").strip()
            if iso:
                try:
                    # 예: 2026-02-02T13:42:48+09:00
                    article_time = pendulum.parse(iso).in_timezone(KST)
                except Exception:
                    article_time = None
            else:
                date_str = (meta.get("date") or "").strip()
                article_time = parse_kst_datetime(date_str)

        if article_time and article_time < cutoff:
            delete_ids.append(_id)

    if not delete_ids:
        print("삭제할 오래된 문서 없음")
        return 0

    col.delete(ids=delete_ids)

    # PersistentClient는 보통 자동으로 저장되지만, 버전별로 persist가 있을 수 있어 방어
    try:
        client.persist()
    except Exception:
        pass

    print(f"삭제 완료: {len(delete_ids)}개 chunk 제거")
    return len(delete_ids)
