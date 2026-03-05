# EdgeQuake Docker Deployment

This directory contains Docker configuration for deploying EdgeQuake.

## PDF Parsing Microservice (Two-Track VLM)

> **Important Setup Note**: EdgeQuake uses a dedicated `pdf-parser` microservice to extract highly complex structures (merging multi-page tables, recognizing rowspans) and analyze charts via Vision Language Models (Qwen2-VL). 
> 
> * **Hardware requirements:** The `pdf-parser` container utilizes GPU acceleration. To run the Qwen2-VL model locally, ensure your host environment has at least 16GB-24GB of VRAM and the NVIDIA Container Toolkit is installed.

## Quick Start

```bash
# Build and start all services via Makefile
make docker-up

# Alternatively, using docker-compose directly
docker-compose up -d

# View logs
make docker-logs
```

## Services

| Service | Port | Description |
|---|---|---|
| `edgequake` | 8080 | EdgeQuake API server |
| `frontend` | 3000 | Next.js Frontend |
| `pdf-parser` | 8000 | FastAPI + Docling + vLLM PDF parser pipeline |
| `postgres` | 5432 | PostgreSQL with pgvector |

## Environment Variables

Create a `.env` file in the root project directory:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key

# Optional API config
EDGEQUAKE_PORT=8080
POSTGRES_PASSWORD=edgequake_secret

# PDF Parser Optional Config
VLM_MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
```

## Managing PDF-Parser Specifically

If you want to only interact with the PDF parsing module:

```bash
make docker-pdf-parser-build  # Build the base CUDA dependencies
make docker-pdf-parser-up     # Spin up the single microservice
make docker-pdf-parser-logs   # Monitor VLM and layout extraction logs
```

## Production Deployment

For production, you may adapt `docker-compose.prod.yml` to fit your orchestration tool, ensuring the `deploy.resources.reservations.devices.capabilities: [gpu]` parameter is preserved for the `pdf-parser` service.

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
