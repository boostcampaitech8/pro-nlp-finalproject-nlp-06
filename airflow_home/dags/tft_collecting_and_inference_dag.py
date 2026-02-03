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
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")

TFT_PYTHON = os.getenv("TFT_PYTHON", "")
STOCK_CSV = os.getenv("STOCK_CSV", "")
HOLIDAY_CSV = os.getenv("HOLIDAY_CSV", "")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "")
MODEL_CKPT = os.getenv("MODEL_CKPT", "")

SCHEDULE = os.getenv("TFT_SCHEDULE", "0 7 * * *")

with DAG(
    dag_id="kospi200_tft_daily_kst",
    description="Update KRX data → TFT inference → save json (KST)",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, 0, 0, tz=KST),
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["stocks", "tft", "kst"],
) as dag:
    # Airflow Template
    run_date = "{{ ds }}"
    run_dir = f"{ARTIFACT_DIR}"

    # Output 파일 경로
    # updated_csv = f"{run_dir}/kospi200_updated.csv"
    updated_csv = f"{PROJECT_ROOT}/tft/data/kospi200_merged_2021_2025_updated.csv"
    inference_json = f"{run_dir}/inference_results.json"

    update_data = BashOperator(
        task_id="update_krx_data_csv",
        bash_command=(
            # 'set -euo pipefail\n'
            f'mkdir -p "{run_dir}"\n'
            f'cd "{PROJECT_ROOT}/tft"\n'
            f'"{TFT_PYTHON}" -m krx_data_update '
            f'--master_csv "{STOCK_CSV}"\n'
            # f'--output_csv "{updated_csv}"\n'
        )
    )
    
    run_inference = BashOperator(
        task_id="run_tft_inference",
        bash_command=(
            # 'set -euo pipefail\n'
            f'mkdir -p "{run_dir}"\n'
            f'cd "{PROJECT_ROOT}/tft"\n'
            f'"{TFT_PYTHON}" -m krx_inference '
            f'--data_csv "{updated_csv}" '
            f'--holiday_csv "{HOLIDAY_CSV}"\n'
            f'mv -f "inference_results.json" "{inference_json}"\n'
            f'echo "[INFO] inference_json saved: {inference_json}"\n'
        )
    )

    update_data >> run_inference