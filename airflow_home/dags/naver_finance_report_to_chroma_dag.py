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

CHROMA_DIR = os.environ["PDF_CHROMA_DIR"]
CSV_DIR = os.environ["CSV_DIR"]
CHROMA_COLLECTION = os.environ["PDF_CHROMA_COLLECTION"]

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
VLLM_MODEL = os.environ["VLLM_MODEL"]
VLLM_API_KEY = os.environ["VLLM_API_KEY"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

SCHEDULE = os.environ.get("PDF_PIPELINE_SCHEDULE", "0 12 * * *")

with DAG(
    dag_id="naver_finance_report_to_chroma_kst",
    description="Crawl → pdf parsing -> summarize → chroma (KST)",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, 0, 0, tz=KST),
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["naver", "chroma", "vllm", "chunking", "kst"],
) as dag:

    run_pipeline = BashOperator(
        task_id="run_news_hourly",
        bash_command=f'cd "{PROJECT_ROOT}" && "{PY310_PYTHON}" -m src.report_pipeline',
    )

    run_pipeline
