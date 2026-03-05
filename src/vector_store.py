"""
vector_store.py — Persistent vector storage and similarity search via ChromaDB.

Provides:
    SearchResult  — dataclass for a single search hit (text + metadata + score).
    ChromaStore   — create/load a named Chroma collection, add chunks with their
                    embeddings, and run top-k similarity queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import chromadb

# Settings moved to the top-level package in chromadb >= 0.5; keep a fallback.
try:
    from chromadb import Settings
except ImportError:
    from chromadb.config import Settings  # type: ignore[no-redef]

from src.chunking import Chunk

logger = logging.getLogger(__name__)

# Chroma stores distances, not similarities.  We convert with: score = 1 - distance.
# This is exact for cosine distance when vectors are L2-normalised.
_DISTANCE_TO_SCORE = lambda d: round(1.0 - float(d), 6)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single hit returned by :meth:`ChromaStore.search`.

    Attributes:
        text:       The chunk text.
        score:      Similarity score in ``[0, 1]`` (higher is more similar).
                    Computed as ``1 - cosine_distance``.
        source:     Origin document identifier (file path, URL, …).
        chunk_id:   The Chroma document ID for this chunk.
        metadata:   Full metadata dict stored alongside the chunk.
    """

    text: str
    score: float
    source: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ChromaStore:
    """Persistent vector store backed by ChromaDB.

    Collections are stored on disk under *persist_dir* and reloaded
    automatically on subsequent runs — no re-indexing required unless the
    source documents change.

    Args:
        collection_name: Name of the Chroma collection.  A new collection is
                         created if it does not already exist.
        persist_dir:     Directory where Chroma writes its SQLite + vector
                         files.  Defaults to ``"./.chroma"``.
        distance_fn:     Distance metric used by the collection.
                         ``"cosine"`` *(default)* works best with L2-normalised
                         vectors from :class:`~src.embeddings.EmbeddingModel`.
                         Also accepts ``"l2"`` or ``"ip"`` (inner product).

    Example:
        # store = ChromaStore("rag-demo", persist_dir=".chroma")
        # store.add(chunks, vectors)
        # results = store.search(query_vector, top_k=5)
        # for r in results:
        #     print(f"[{r.score:.3f}] {r.source}  {r.text[:80]}")
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | Path = ".chroma",
        distance_fn: str = "cosine",
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        if distance_fn not in ("cosine", "l2", "ip"):
            raise ValueError(f"distance_fn must be 'cosine', 'l2', or 'ip', got '{distance_fn}'")

        self.collection_name = collection_name
        self.distance_fn = distance_fn

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
            logger.info("ChromaStore connecting to remote Chroma at %s:%d", host, port)
        else:
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_fn},
        )
        logger.info(
            "ChromaStore ready — collection='%s', docs=%d",
            collection_name, self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of documents currently stored in the collection."""
        return self._collection.count()

    def has_source(self, source: str) -> bool:
        """Return True if at least one chunk with this *source* is indexed.

        Used by the ingest script to skip already-processed documents.
        """
        result = self._collection.get(where={"source": source}, limit=1, include=[])
        return len(result["ids"]) > 0

    def add(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        filename: str = "",
    ) -> None:
        """Store *chunks* and their pre-computed *vectors* in the collection.

        Chunks that share an ID with an existing document are silently skipped
        (Chroma's ``add`` raises on duplicates; we use ``upsert`` to make
        re-indexing idempotent).

        Args:
            chunks:   Ordered list of :class:`~src.chunking.Chunk` objects.
            vectors:  Float32 numpy array of shape ``(len(chunks), dim)``
                      produced by :class:`~src.embeddings.EmbeddingModel`.
            filename: Optional human-readable filename to include in metadata
                      (e.g. ``"report.pdf"``).  Falls back to the chunk's
                      ``source`` field when empty.

        Raises:
            ValueError: If ``len(chunks) != len(vectors)``.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length "
                f"(got {len(chunks)} vs {len(vectors)})"
            )
        if not chunks:
            logger.warning("add() called with empty chunks list — nothing to store.")
            return

        ids, documents, embeddings, metadatas = [], [], [], []

        for chunk, vector in zip(chunks, vectors):
            chunk_id = _make_id(chunk)
            ids.append(chunk_id)
            documents.append(chunk.text)
            embeddings.append(vector.tolist())
            metadatas.append({
                "source": chunk.source,
                "filename": filename or chunk.source,
                "chunk_index": chunk.index,
                "char_start": chunk.char_start,
                **{k: str(v) for k, v in chunk.extra.items()},  # Chroma requires str values
            })

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunk(s) into collection '%s'.", len(chunks), self.collection_name)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the *top_k* most similar chunks to *query_vector*.

        Args:
            query_vector: 1-D float32 numpy array of shape ``(dim,)``.
            top_k:        Number of results to return (default 5).
            where:        Optional Chroma metadata filter, e.g.
                          ``{"source": "report.pdf"}``.

        Returns:
            List of :class:`SearchResult` objects ordered by descending
            similarity score (best match first).

        Raises:
            ValueError: If the collection is empty.
        """
        if self._collection.count() == 0:
            raise ValueError("The collection is empty — call add() before search().")

        top_k = min(top_k, self._collection.count())

        query_kwargs: dict[str, Any] = dict(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        results: list[SearchResult] = []
        for doc, meta, dist, cid in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
            raw["ids"][0],
        ):
            results.append(SearchResult(
                text=doc,
                score=_DISTANCE_TO_SCORE(dist),
                source=meta.get("source", ""),
                chunk_id=cid,
                metadata=meta,
            ))

        return results

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by their IDs.

        Args:
            chunk_ids: List of Chroma document IDs to delete.
        """
        if not chunk_ids:
            return
        self._collection.delete(ids=chunk_ids)
        logger.info("Deleted %d chunk(s) from collection '%s'.", len(chunk_ids), self.collection_name)

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks whose ``source`` metadata field matches *source*.

        Useful for re-indexing a single document without touching others.

        Args:
            source: The source identifier used when the chunks were added.
        """
        self._collection.delete(where={"source": source})
        logger.info("Deleted all chunks with source='%s' from '%s'.", source, self.collection_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(chunk: Chunk) -> str:
    """Build a stable, unique Chroma document ID for a chunk.

    Format: ``<source>__chunk<index>``  e.g. ``report.pdf__chunk7``
    Falls back to ``__chunk<index>`` when source is empty.
    """
    prefix = chunk.source.replace("/", "_").replace("\\", "_") if chunk.source else ""
    return f"{prefix}__chunk{chunk.index}" if prefix else f"__chunk{chunk.index}"
