#!/usr/bin/env bash
# set -euo pipefail

# source로 실행되는 파일에서는 $0가 쉘이 될 수 있으니 BASH_SOURCE 사용
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TZ=Asia/Seoul

export PROJECT_ROOT
export AIRFLOW_HOME="$PROJECT_ROOT/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__DEFAULT_TIMEZONE="Asia/Seoul"


# DAG에서 읽는 값들 기본 세팅
export CHROMA_DIR="${CHROMA_DIR:-$PROJECT_ROOT/Chroma_db/News_chroma_db}"
export PIPELINE_SCHEDULE="${PIPELINE_SCHEDULE:-0 * * * *}"
export CHROMA_COLLECTION="${CHROMA_COLLECTION:-naver_finance_news_chunks}"


export PDF_CHROMA_DIR="${PDF_CHROMA_DIR:-$PROJECT_ROOT/Chroma_db/News_chroma_db}"
export PDF_PIPELINE_SCHEDULE="${PDF_PIPELINE_SCHEDULE:-0 12 * * *}"
# export PDF_PIPELINE_SCHEDULE="${PDF_PIPELINE_SCHEDULE:-0 * * * *}"
export PDF_CHROMA_COLLECTION="${PDF_CHROMA_COLLECTION:-langchain}"


export CSV_DIR="${CSV_DIR:-$PROJECT_ROOT/csv_out}"


export PIPELINE_PYTHON="${PIPELINE_PYTHON:-$PROJECT_ROOT/../py310/bin/python}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

# [변경] Ollama 설정을 vLLM 및 HF 설정으로 교체
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
export VLLM_MODEL="${VLLM_MODEL:-skt/A.X-4.0-Light}"
export VLLM_API_KEY="${VLLM_API_KEY:-vllm-key}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-jhgan/ko-sroberta-multitask}"

# [추가] TFT Airflow Environment Variables
# TFT_PYTHON 경로는 가상환경 경로에 맞게 설정하기
# export TFT_PYTHON="${TFT_PYTHON:-$PROJECT_ROOT/../py310/bin/python}"
export TFT_PYTHON="${TFT_PYTHON:-$PROJECT_ROOT/../.venv/bin/python}"
export STOCK_CSV="$PROJECT_ROOT/tft/data/kospi200_merged_2021_2025_v2.csv"
export HOLIDAY_CSV="$PROJECT_ROOT/tft/data/krx_close.csv"
export ARTIFACT_DIR="$PROJECT_ROOT/tft/result"
export MODEL_CKPT="$PROJECT_ROOT/model/epoch=5-step=10716.ckpt"
export TFT_SCHEDULE="${TFT_SCHEDULE:-0 7 * * *}"
