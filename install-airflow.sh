#!/usr/bin/env bash
set -euxo pipefail

echo "▶ Activating airflow venv..."
source /data/ephemeral/home/py310-airflow/bin/activate

echo "▶ Upgrading pip..."
pip install --upgrade pip

AIRFLOW_VERSION=2.6.3
PYTHON_VERSION="$(python --version | cut -d ' ' -f 2 | cut -d '.' -f 1-2)"

CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

echo "▶ Installing Airflow ${AIRFLOW_VERSION} with constraints..."
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

echo "▶ Installing Providers..."
pip install \
  apache-airflow-providers-common-sql \
  apache-airflow-providers-http \
  apache-airflow-providers-sqlite

echo "Airflow installation complete."
