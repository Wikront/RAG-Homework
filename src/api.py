"""
api.py — OpenAI-compatible FastAPI server backed by the RAG pipeline.

Exposes:
    GET  /v1/models                 — model list (required by Open WebUI)
    POST /v1/chat/completions       — chat endpoint; supports streaming

Connect Open WebUI (or any OpenAI-compatible client) to this server by
setting OPENAI_API_BASE_URLS=http://<host>:8080/v1.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.config import get_config
from src.embeddings import create_embedding_model
from src.rag import OpenAIClient, RAGPipeline
from src.vector_store import ChromaStore

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

# ---------------------------------------------------------------------------
# Shared pipeline (initialised at startup)
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline

    cfg = get_config()
    logger.info("Initialising RAG pipeline …")

    embedder = create_embedding_model(cfg)
    store = ChromaStore(
        collection_name=cfg.collection_name,
        persist_dir=cfg.chroma_persist_dir,
        host=cfg.chroma_host,
        port=cfg.chroma_port,
    )
    llm = OpenAIClient(
        model=cfg.llm_model_name,
        base_url=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
    )
    _pipeline = RAGPipeline(store=store, embedder=embedder, llm=llm)
    logger.info("RAG pipeline ready (collection=%s, docs=%d).", cfg.collection_name, store.count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_ID = "rag"


def _model_card() -> dict[str, Any]:
    return {
        "id": _MODEL_ID,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }


def _sse_chunk(content: str, completion_id: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done(completion_id: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [_model_card()]}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]) -> Any:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    messages: list[dict[str, str]] = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty.")

    # Use the last user message as the RAG query.
    question = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if not question:
        raise HTTPException(status_code=400, detail="No user message found.")

    stream: bool = body.get("stream", False)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if stream:
        def _generate():
            for token in _pipeline.stream_query(question):
                yield _sse_chunk(token, completion_id)
            yield _sse_done(completion_id)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = _pipeline.query(question)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
