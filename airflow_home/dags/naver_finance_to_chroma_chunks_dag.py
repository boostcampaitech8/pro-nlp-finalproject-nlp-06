from __future__ import annotations

import os
import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

KST = pendulum.timezone("Asia/Seoul")

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# === 환경변수에서만 읽는다 (airflow_env.sh가 source됨) ===
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
PY310_PYTHON = os.environ["PIPELINE_PYTHON"]

CHROMA_DIR = os.environ["CHROMA_DIR"]
CSV_DIR = os.environ["CSV_DIR"]
CHROMA_COLLECTION = os.environ["CHROMA_COLLECTION"]

OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]
OLLAMA_EMBED_MODEL = os.environ["OLLAMA_EMBED_MODEL"]

SCHEDULE = os.environ.get("PIPELINE_SCHEDULE", "0 * * * *")

with DAG(
    dag_id="naver_finance_to_chroma_chunks_kst",
    description="Crawl → summarize → keywords → chunk → chroma (KST)",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, 0, 0, tz=KST),
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["naver", "chroma", "ollama", "chunking", "kst"],
) as dag:

    run_pipeline = BashOperator(
        task_id="run_news_rag_hourly",
        bash_command=(
            f'"{PY310_PYTHON}" "{PROJECT_ROOT}/src/pipeline.py"'
        ),
    )

    run_pipeline
