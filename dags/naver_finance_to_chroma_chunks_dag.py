from __future__ import annotations

import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

# Airflow 컨테이너/서버에서 project 경로가 다를 수 있으니, src import를 위해 경로 추가
# /opt/airflow/project 형태로 마운트했다면 아래를 그에 맞게 조정
# PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
PROJECT_ROOT = "/data/ephemeral/home/project"

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from pipeline import run_pipeline  # noqa: E402

KST = pendulum.timezone("Asia/Seoul")

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="naver_finance_to_chroma_chunks_kst",
    description="Crawl -> Ollama(llama3) summarize -> keywords -> chunk -> Chroma(Ollama embeddings) + cleanup (KST)",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, 0, 0, tz=KST),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    timezone=KST,
    tags=["naver", "chroma", "ollama", "chunking", "kst"],
) as dag:

    run = PythonOperator(
        task_id="run_pipeline_hourly",
        python_callable=run_pipeline,
        op_kwargs={
            # 최근 N시간 수집
            "hours": 1,
            "max_page": 10,

            # Chroma 저장 위치 (영속화하려면 volume mount 권장)
            "chroma_dir": os.getenv("CHROMA_DIR", "/opt/airflow/chroma_news"),
            "chroma_collection": os.getenv("CHROMA_COLLECTION", "naver_finance_news_chunks"),

            # Ollama 요약(llama3)
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
            "summarize_timeout": 120,

            # 키워드(간단 TF-IDF)
            "keyword_top_k": 10,

            # chunking
            "chunk_size": 800,
            "overlap": 120,

            # 2주 지난 문서 삭제
            "cleanup_days": 14,

            # CSV 저장(기사 단위) - 원하면 False로 끄면 됨
            "save_csv": True,
            "csv_output_dir": os.getenv("CSV_DIR", "/opt/airflow/csv_out"),
        },
    )

    run
