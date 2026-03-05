# RAG Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot.
Loads PDFs and audio/video files, transcribes, chunks, embeds, stores in ChromaDB, and answers questions via any OpenAI-compatible LLM (including local LM Studio models).
Served as an OpenAI-compatible HTTP API and accessible through **Open WebUI** in the browser.

---

## Project structure

```
RAG-Homework/
├── src/
│   ├── __init__.py        package entry point
│   ├── config.py          environment-based configuration
│   ├── loaders.py         PDFLoader, AudioLoader (faster-whisper)
│   ├── chunking.py        TextChunker, Chunk
│   ├── embeddings.py      EmbeddingModel (local or OpenAI-compatible)
│   ├── vector_store.py    ChromaStore (local + HTTP)
│   ├── rag.py             RAGPipeline, OpenAIClient, RAGResponse
│   ├── api.py             FastAPI OpenAI-compatible server
│   └── chat.py            interactive CLI (local use)
├── scripts/
│   ├── ingest.py          one-shot ingestion pipeline
│   └── test_rag.py        smoke-test script (3 fixed questions → log file)
├── data/
│   ├── pdf/               ← drop PDF files here
│   ├── audio/             ← drop .mp3 / .wav / .mp4 files here
│   └── transcripts/       auto-generated transcripts
├── Dockerfile
├── entrypoint.sh
├── docker-compose.yml
├── .env                   ← copy from .env and edit
└── requirements.txt
```

## Pipeline overview

```
data/pdf/   ──→ PDFLoader   ──┐
                               ├──→ TextChunker ──→ EmbeddingModel ──→ ChromaDB
data/audio/ ──→ AudioLoader ──┘       (background on startup)

User question ──→ EmbeddingModel ──→ ChromaDB.search() ──→ LLM ──→ Answer
     ↑                                                              ↓
Open WebUI  ←───────────────── RAG API (:8080) ─────────────────────
```

---

## Quick start with Docker (recommended)

### 1 — Start LM Studio

1. Download [LM Studio](https://lmstudio.ai) and load a chat model and an embedding model.
2. Go to **Local Server** and click **Start Server** (default port `1234`).
3. Note the model identifiers shown in the UI.

### 2 — Configure environment

Edit `.env` to match your loaded models:

```dotenv
LLM_API_BASE=http://host.docker.internal:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=qwen2.5-7b-instruct-1m

EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
```

### 3 — Add documents

```bash
cp /path/to/your/docs/*.pdf   data/pdf/
cp /path/to/your/audio/*.mp4  data/audio/   # mp3, wav, mp4 supported
```

### 4 — Build and run

```bash
docker compose up --build
```

On startup the container will:
1. Wait for ChromaDB to be ready.
2. Run `scripts/ingest.py` in the background (skips already-indexed files).
3. Start the RAG API server on port **8080** immediately.

### 5 — Open the browser UI

```
http://localhost:3000
```

Select the **rag** model in Open WebUI and start chatting.

### Verify the API is up

```
http://localhost:8080/v1/models
```

Interactive API docs: `http://localhost:8080/docs`

---

## Smoke test

Run a fixed set of questions against the live API and save the answers to a log file:

```bash
python scripts/test_rag.py
```

Output is written to `rag_test_results.log` in the current directory.

Options:

```
python scripts/test_rag.py --url http://localhost:8080   # custom endpoint
python scripts/test_rag.py --out results/my_test.log    # custom output file
```

The script uses only stdlib (`urllib`, `json`) — no extra dependencies required.

---

## Running locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set env vars or create a .env file:
export LLM_API_BASE=http://localhost:1234/v1
export LLM_API_KEY=lm-studio
export LLM_MODEL_NAME=local-model
export EMBEDDING_BACKEND=openai
export EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
export CHROMA_HOST=               # leave empty to use local PersistentClient
export COLLECTION_NAME=rag
export DATA_DIR=./data

python scripts/ingest.py          # index documents
uvicorn src.api:app --port 8080   # start API server
```

---

## Environment variables

| Variable             | Description                                              | Default                                    |
|----------------------|----------------------------------------------------------|--------------------------------------------|
| `LLM_API_BASE`       | OpenAI-compatible endpoint base URL                      | `http://host.docker.internal:1234/v1`      |
| `LLM_API_KEY`        | API key (any string for LM Studio)                       | `lm-studio`                                |
| `LLM_MODEL_NAME`     | Model identifier sent in each request                    | `local-model`                              |
| `EMBEDDING_BACKEND`  | `openai` (LM Studio) or `local` (sentence-transformers)  | `local`                                    |
| `EMBEDDING_MODEL`    | Embedding model name                                     | `sentence-transformers/all-MiniLM-L6-v2`  |
| `CHROMA_HOST`        | ChromaDB hostname (blank = local PersistentClient)       | *(blank)*                                  |
| `CHROMA_PORT`        | ChromaDB port                                            | `8000`                                     |
| `CHROMA_PERSIST_DIR` | Local path for PersistentClient                          | `.chroma`                                  |
| `COLLECTION_NAME`    | Chroma collection name                                   | `rag`                                      |
| `DATA_DIR`           | Root directory for pdf/, audio/, transcripts/            | `./data`                                   |

---

## Re-ingesting documents

The ingest script is **idempotent** — files whose source path is already in the
collection are skipped.  To force a re-index of a specific document, delete its
chunks first:

```python
from src.config import get_config
from src.vector_store import ChromaStore

cfg = get_config()
store = ChromaStore(cfg.collection_name, host=cfg.chroma_host, port=cfg.chroma_port)
store.delete_by_source("data/pdf/report.pdf")
```

Then re-run `python scripts/ingest.py`.

---

## Requirements

- Python 3.11+
- Docker & Docker Compose (for containerised deployment)
- LM Studio (or any OpenAI-compatible API endpoint)
