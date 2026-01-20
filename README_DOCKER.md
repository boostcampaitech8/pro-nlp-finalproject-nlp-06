
## Prerequisites
- Docker + Docker Compose
- NVIDIA driver installed
- nvidia-container-toolkit installed (for `gpus: all`)
- Confirm:
  - `nvidia-smi` works on host
  - `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi` works

## Setup
```bash
cp .env.example .env

## Init(first time only)
docker compose up airflow-init

## Start services
docker compose up -d --build
docker exec -it $(docker ps --filter "name=ollama" -q) ollama pull llama3
