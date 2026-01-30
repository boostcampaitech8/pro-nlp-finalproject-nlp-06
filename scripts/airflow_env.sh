#!/usr/bin/env bash
set -euo pipefail

# source로 실행되는 파일에서는 $0가 쉘이 될 수 있으니 BASH_SOURCE 사용
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TZ=Asia/Seoul

export PROJECT_ROOT
export AIRFLOW_HOME=/data/ephemeral/home/pro-nlp-finalproject-nlp-06/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__DEFAULT_TIMEZONE="Asia/Seoul"

# DAG에서 읽는 값들 기본 세팅
export PIPELINE_PYTHON="/data/ephemeral/home/.venv/bin/python"
export CHROMA_DIR="${CHROMA_DIR:-$PROJECT_ROOT/Chroma_DB}"
export CSV_DIR="${CSV_DIR:-$PROJECT_ROOT/csv_out}"
export CHROMA_COLLECTION="${CHROMA_COLLECTION:-naver_finance_news_chunks}"
# [변경] Ollama 설정을 vLLM 및 HF 설정으로 교체
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
export VLLM_MODEL="${VLLM_MODEL:-skt/A.X-4.0-Light}"
export VLLM_API_KEY="${VLLM_API_KEY:-vllm-key}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-jhgan/ko-sroberta-multitask}"
export PIPELINE_SCHEDULE="${PIPELINE_SCHEDULE:-0 * * * *}" 