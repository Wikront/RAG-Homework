"""
config.py — Central configuration loaded from environment variables.

All settings have sensible defaults so local dev works without a .env file.
In Docker the values are supplied via docker-compose env_file.

Usage:
    from src.config import get_config
    cfg = get_config()
    print(cfg.llm_api_base, cfg.chroma_host)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_api_base: str
    """Base URL of the OpenAI-compatible chat endpoint.
    LM Studio default: http://host.docker.internal:1234/v1"""

    llm_api_key: str
    """API key sent in the Authorization header.  LM Studio accepts any string."""

    llm_model_name: str
    """Model identifier forwarded in the 'model' field of each completion request."""

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_backend: str
    """``"openai"`` → use OpenAI-compatible /v1/embeddings endpoint (LM Studio).
    ``"local"`` → use sentence-transformers (runs inside the container)."""

    embedding_model: str
    """Model identifier. For ``"local"``: a sentence-transformers HuggingFace path.
    For ``"openai"``: the model name sent to the /v1/embeddings API."""

    embedding_api_base: str
    """Base URL for the embeddings API when backend is ``"openai"``.
    Defaults to ``LLM_API_BASE`` so both chat and embeddings share one server."""

    embedding_api_key: str
    """API key for the embeddings endpoint. Defaults to ``LLM_API_KEY``."""

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_host: str | None
    """Hostname of the remote ChromaDB service.  None → use local PersistentClient."""

    chroma_port: int
    """Port of the remote ChromaDB service (default 8000)."""

    chroma_persist_dir: str
    """Local path for PersistentClient when CHROMA_HOST is not set."""

    collection_name: str
    """Name of the Chroma collection that holds all indexed chunks."""

    # ── Data paths ────────────────────────────────────────────────────────────
    data_dir: Path
    """Root directory that contains pdf/, audio/, and transcripts/ sub-folders."""


def get_config() -> Config:
    """Load configuration from environment variables (and .env if present).

    Raises:
        KeyError: If a required variable is missing and has no default.
    """
    load_dotenv()

    chroma_host_raw = os.getenv("CHROMA_HOST", "").strip()

    return Config(
        # LLM
        llm_api_base=os.environ.get("LLM_API_BASE", "http://host.docker.internal:1234/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", "lm-studio"),
        llm_model_name=os.environ.get("LLM_MODEL_NAME", "local-model"),
        # Embeddings
        embedding_backend=os.environ.get("EMBEDDING_BACKEND", "local"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_api_base=os.environ.get("EMBEDDING_API_BASE")
            or os.environ.get("LLM_API_BASE", "http://host.docker.internal:1234/v1"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("LLM_API_KEY", "lm-studio"),
        # ChromaDB
        chroma_host=chroma_host_raw or None,
        chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
        chroma_persist_dir=os.environ.get("CHROMA_PERSIST_DIR", ".chroma"),
        collection_name=os.environ.get("COLLECTION_NAME", "rag"),
        # Paths
        data_dir=Path(os.environ.get("DATA_DIR", "./data")),
    )
