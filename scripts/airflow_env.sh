#!/usr/bin/env bash
# set -euo pipefail

# source로 실행되는 파일에서는 $0가 쉘이 될 수 있으니 BASH_SOURCE 사용
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PROJECT_ROOT
export AIRFLOW_HOME="$PROJECT_ROOT/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__DEFAULT_TIMEZONE="Asia/Seoul"

# DAG에서 읽는 값들 기본 세팅
export PIPELINE_PYTHON="${PIPELINE_PYTHON:-$PROJECT_ROOT/../.venv-ollama/bin/python}"
export CHROMA_DIR="${CHROMA_DIR:-$PROJECT_ROOT/chroma_news}"
export CSV_DIR="${CSV_DIR:-$PROJECT_ROOT/csv_out}"
export CHROMA_COLLECTION="${CHROMA_COLLECTION:-naver_finance_news_chunks}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
export PIPELINE_SCHEDULE="${PIPELINE_SCHEDULE:-0 * * * *}"

# TFT Airflow Environment Variables
export TZ="${TZ:-Asia/Seoul}"
export TFT_PYTHON="${TFT_PYTHON:-$PROJECT_ROOT/../../.venv/bin/python}"
export STOCK_CSV="$PROJECT_ROOT/tft/data/kospi200_merged_2021_2025_v2.csv"
export HOLIDAY_CSV="$PROJECT_ROOT/tft/data/krx_close.csv"
export ARTIFACT_DIR="$PROJECT_ROOT/tft/result"
export MODEL_CKPT="$PROJECT_ROOT/model/epoch=5-step=10716.ckpt"
export TFT_SCHEDULE="${TFT_SCHEDULE:-0 7 * * *}"