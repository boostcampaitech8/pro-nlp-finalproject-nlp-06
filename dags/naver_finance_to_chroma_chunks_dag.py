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

# 프로젝트/venv 경로
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/data/ephemeral/home/project")
PY310_PYTHON = os.getenv("PIPELINE_PYTHON", "/data/ephemeral/home/py310/bin/python")

# 실행 주기: 10분마다
SCHEDULE = os.getenv("PIPELINE_SCHEDULE", "0 * * * *")

# 저장 경로 통일 (FastAPI도 같은 CHROMA_DIR을 바라봐야 함)
CHROMA_DIR = os.getenv("CHROMA_DIR", f"{PROJECT_ROOT}/chroma_news")
CSV_DIR = os.getenv("CSV_DIR", f"{PROJECT_ROOT}/csv_out")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "naver_finance_news_chunks")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

with DAG(
    dag_id="naver_finance_to_chroma_chunks_kst",
    description="Crawl -> summarize -> keywords -> chunk -> chroma (run pipeline in py310 venv) (KST)",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, 0, 0, tz=KST),
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["naver", "chroma", "ollama", "chunking", "kst"],
) as dag:

    run_pipeline = BashOperator(
        task_id="run_pipeline_hourly",
        bash_command=(
            # 환경변수로 파이프라인 설정을 전달
            f'export PROJECT_ROOT="{PROJECT_ROOT}"; '
            f'export CHROMA_DIR="{CHROMA_DIR}"; '
            f'export CSV_DIR="{CSV_DIR}"; '
            f'export CHROMA_COLLECTION="{CHROMA_COLLECTION}"; '
            f'export OLLAMA_BASE_URL="{OLLAMA_BASE_URL}"; '
            f'export OLLAMA_MODEL="{OLLAMA_MODEL}"; '
            f'export OLLAMA_EMBED_MODEL="{OLLAMA_EMBED_MODEL}"; '
            # 파이프라인 실행 (py310 venv로)
            f'"{PY310_PYTHON}" "{PROJECT_ROOT}/src/pipeline.py"'
        ),
    )

    run_pipeline
