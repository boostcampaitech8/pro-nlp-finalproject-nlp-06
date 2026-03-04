#!/usr/bin/env bash
set -euo pipefail

# === 환경변수 주입 ===
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/airflow_env.sh"

# === 디렉토리 보장 ===
mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/logs" "$AIRFLOW_HOME/plugins" "$CSV_DIR"

echo "[airflow] PROJECT_ROOT=$PROJECT_ROOT"
echo "[airflow] AIRFLOW_HOME=$AIRFLOW_HOME"
echo "[airflow] DAGS_FOLDER=$AIRFLOW__CORE__DAGS_FOLDER"

# === DB 초기화 (이미 돼 있으면 skip) ===
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
  echo "[airflow] Initializing DB..."
  airflow db init
else
  echo "[airflow] DB already initialized"
fi

# === admin 유저 생성 (있으면 skip) ===
if ! airflow users list | grep -q "^admin"; then
  echo "[airflow] Creating admin user..."
  airflow users create \
      --username admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email admin@example.com \
      --password admin
else
  echo "[airflow] Admin user already exists"
fi

# === Airflow 실행 ===
echo "[airflow] Starting airflow standalone..."
airflow standalone
