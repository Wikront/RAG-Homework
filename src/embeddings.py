"""
embeddings.py — Vector encoding layer.

Provides:
    EmbeddingModel       — local sentence-transformers backend (HuggingFace).
    OpenAIEmbeddingModel — remote OpenAI-compatible /v1/embeddings backend
                           (LM Studio, OpenAI API, or any compatible server).
    create_embedding_model(cfg) — factory that returns the right backend based
                           on cfg.embedding_backend ("local" or "openai").
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

Vector = np.ndarray  # shape (dim,), dtype float32
Matrix = np.ndarray  # shape (n, dim), dtype float32


# ---------------------------------------------------------------------------
# Local backend — sentence-transformers
# ---------------------------------------------------------------------------

# Module-level cache: model_name → SentenceTransformer instance.
_MODEL_CACHE: dict = {}


def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    if model_name not in _MODEL_CACHE:
        logger.info("Loading local embedding model '%s' …", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        logger.info("Model '%s' loaded (dim=%d).",
                    model_name, _MODEL_CACHE[model_name].get_sentence_embedding_dimension())
    return _MODEL_CACHE[model_name]


class EmbeddingModel:
    """Encode text into dense vectors using a local sentence-transformers model.

    The underlying ``SentenceTransformer`` is cached per model name so only
    the first construction pays the load cost.

    Args:
        model_name: HuggingFace Hub path or local directory.
                    Default: ``"sentence-transformers/all-MiniLM-L6-v2"``.
        batch_size: Texts per forward pass (default 32).
        normalize:  L2-normalise vectors (default True).

    Example:
        # model = EmbeddingModel()
        # matrix = model.encode(["RAG combines retrieval with generation."])
        # vec    = model.encode_query("What is RAG?")
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = _load_st_model(model_name)

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def encode(self, texts: list[str]) -> Matrix:
        """Embed a list of strings.  Returns float32 array of shape (N, dim)."""
        if not texts:
            raise ValueError("texts must be a non-empty list.")
        logger.debug("Encoding %d text(s) with batch_size=%d.", len(texts), self.batch_size)
        vectors: Matrix = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> Vector:
        """Embed a single query string.  Returns float32 array of shape (dim,)."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return self.encode([query])[0]


# ---------------------------------------------------------------------------
# Remote backend — OpenAI-compatible /v1/embeddings (LM Studio, OpenAI, …)
# ---------------------------------------------------------------------------

class OpenAIEmbeddingModel:
    """Encode text via an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Suitable for LM Studio (load an embedding model alongside the chat model),
    the OpenAI API, or any server that speaks the OpenAI embeddings protocol.

    Args:
        model_name: Model identifier sent in the ``model`` field of each request,
                    e.g. ``"text-embedding-qwen3-embedding-4b"`` for LM Studio.
        api_base:   Base URL of the endpoint, e.g. ``"http://localhost:1234/v1"``.
        api_key:    API key (LM Studio accepts any non-empty string).
        batch_size: Texts sent per API call (default 32).
        normalize:  L2-normalise returned vectors (default True).  Keeps cosine
                    similarity consistent with ChromaDB's cosine distance metric.

    Example:
        # model = OpenAIEmbeddingModel(
        #     model_name="text-embedding-qwen3-embedding-4b",
        #     api_base="http://localhost:1234/v1",
        #     api_key="lm-studio",
        # )
        # vec = model.encode_query("What is RAG?")
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAIEmbeddingModel. "
                "Install it with: pip install openai"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._client = openai.OpenAI(base_url=api_base, api_key=api_key)

        # Probe once to discover the vector dimension and validate connectivity.
        logger.info(
            "Connecting to embedding endpoint '%s' with model '%s' …", api_base, model_name
        )
        probe = self._client.embeddings.create(model=model_name, input=["probe"])
        self._dim = len(probe.data[0].embedding)
        logger.info("Embedding model ready (dim=%d).", self._dim)

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim

    def encode(self, texts: list[str]) -> Matrix:
        """Embed a list of strings via the API.  Returns float32 array (N, dim)."""
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        all_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug("Embedding batch %d–%d via API.", i, i + len(batch) - 1)
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            # OpenAI guarantees order, but sort by index defensively.
            ordered = sorted(response.data, key=lambda d: d.index)
            batch_matrix = np.array([d.embedding for d in ordered], dtype=np.float32)
            all_vectors.append(batch_matrix)

        result: Matrix = np.vstack(all_vectors) if len(all_vectors) > 1 else all_vectors[0]

        if self.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.where(norms == 0, 1.0, norms)

        return result

    def encode_query(self, query: str) -> Vector:
        """Embed a single query string.  Returns float32 array of shape (dim,)."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return self.encode([query])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedding_model(cfg) -> EmbeddingModel | OpenAIEmbeddingModel:
    """Return the embedding backend specified by ``cfg.embedding_backend``.

    Args:
        cfg: A :class:`~src.config.Config` instance.

    Returns:
        :class:`OpenAIEmbeddingModel` when ``cfg.embedding_backend == "openai"``,
        :class:`EmbeddingModel` (local sentence-transformers) otherwise.
    """
    if cfg.embedding_backend == "openai":
        return OpenAIEmbeddingModel(
            model_name=cfg.embedding_model,
            api_base=cfg.embedding_api_base,
            api_key=cfg.embedding_api_key,
        )
    return EmbeddingModel(model_name=cfg.embedding_model)
