"""
rag.py — Core Retrieval-Augmented Generation pipeline.

Provides:
    RAGResponse   — dataclass holding the answer, retrieved context, and
                    source citations from a single query.
    LLMClient     — thin protocol / abstract base that any LLM backend must
                    satisfy, keeping generation swappable without touching the
                    pipeline.
    OpenAIClient  — concrete LLMClient backed by the OpenAI Chat Completions
                    API (also compatible with any OpenAI-compatible endpoint
                    such as Azure OpenAI, Together AI, Groq, local Ollama, …).
    RAGPipeline   — orchestrates embed → retrieve → prompt → generate and
                    exposes a single .query() entry point.
"""

from __future__ import annotations

import logging
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.embeddings import EmbeddingModel
from src.vector_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """The result of a single RAG query.

    Attributes:
        question:  The original user question.
        answer:    The LLM-generated answer string.
        sources:   Deduplicated list of source identifiers (file paths, URLs,
                   …) whose chunks were used to build the context.
        context:   The retrieved :class:`~src.vector_store.SearchResult` objects,
                   ordered by descending similarity score.
    """

    question: str
    answer: str
    sources: list[str]
    context: list[SearchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM abstraction
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Minimal interface that every LLM backend must implement.

    Keeping the interface this small means any provider — OpenAI, Anthropic,
    a local Ollama server, a mock in tests — can be plugged in by subclassing
    and overriding a single method.
    """

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a system + user message pair and return the assistant reply.

        Args:
            system: The system prompt (instructions, persona, constraints).
            user:   The user turn (question with injected context).

        Returns:
            The model's text reply.
        """

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream the reply token by token.  Default: yields the full reply as one chunk.

        Override in subclasses that support native streaming.
        """
        yield self.complete(system, user)


class OpenAIClient(LLMClient):
    """LLM backend backed by the OpenAI Chat Completions API.

    Compatible with any OpenAI-compatible endpoint — pass a custom *base_url*
    to point at Azure OpenAI, Together AI, Groq, a local Ollama server, etc.

    Args:
        model:    Chat model identifier (default ``"gpt-4o-mini"``).
        base_url: Optional alternative base URL for OpenAI-compatible APIs.
        api_key:  API key.  Reads ``OPENAI_API_KEY`` from the environment
                  when omitted.
        temperature: Sampling temperature (default ``0.2`` for factual Q&A).
        max_tokens:  Upper bound on response length (default ``1024``).

    Example:
        # client = OpenAIClient(model="gpt-4o-mini")
        # answer = client.complete(system="You are a helpful assistant.", user="What is RAG?")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)

    def complete(self, system: str, user: str) -> str:
        logger.debug("Calling %s (temp=%.2f, max_tokens=%d).", self.model, self.temperature, self.max_tokens)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        logger.debug("Streaming %s (temp=%.2f, max_tokens=%d).", self.model, self.temperature, self.max_tokens)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise and concise question-answering assistant.
    Answer the user's question using ONLY the context passages provided below.
    If the context does not contain enough information to answer, say so clearly
    rather than speculating.
    Always cite the source of your answer by referencing the passage numbers
    (e.g. "[1]", "[2]") where relevant.
""")

_USER_TEMPLATE = textwrap.dedent("""\
    ### Context passages

    {context_block}

    ### Question

    {question}
""")


def _build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks as a numbered list for inclusion in the prompt."""
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        source_tag = f"  (source: {r.source})" if r.source else ""
        lines.append(f"[{i}]{source_tag}\n{r.text.strip()}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Wires together an :class:`~src.embeddings.EmbeddingModel`, a
    :class:`~src.vector_store.ChromaStore`, and an :class:`LLMClient` into a
    single ``.query()`` call.

    The three stages are intentionally kept in separate private methods so
    any one of them can be overridden in a subclass without touching the
    others:

    * :meth:`_retrieve`  — embed the question and fetch top-k chunks.
    * :meth:`_build_prompt` — format the context into a (system, user) pair.
    * :meth:`_generate`  — call the LLM and return the answer string.

    Args:
        embedder: An :class:`~src.embeddings.EmbeddingModel` instance.
        store:    A :class:`~src.vector_store.ChromaStore` instance.
        llm:      Any :class:`LLMClient` implementation.
        top_k:    Number of chunks to retrieve per query (default 5).
        score_threshold: Minimum similarity score ``[0, 1]`` a chunk must
                         reach to be included in the prompt.  Chunks below
                         this threshold are silently dropped.  Set to ``0.0``
                         to disable filtering (default ``0.0``).

    Example:
        # pipeline = RAGPipeline(
        #     embedder=EmbeddingModel(),
        #     store=ChromaStore("my-collection"),
        #     llm=OpenAIClient(),
        # )
        # response = pipeline.query("What are the main conclusions?")
        # print(response.answer)
        # print("Sources:", response.sources)
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        store: ChromaStore,
        llm: LLMClient,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}")

        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        where: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Answer *question* using retrieved context from the vector store.

        Args:
            question: The user's natural-language question.
            where:    Optional metadata filter forwarded to
                      :meth:`~src.vector_store.ChromaStore.search`
                      (e.g. ``{"source": "report.pdf"}`` to restrict retrieval
                      to a single document).

        Returns:
            A :class:`RAGResponse` with the answer, sources, and full context.

        Raises:
            ValueError: If *question* is blank.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        logger.info("RAG query: %r  (top_k=%d, threshold=%.2f)", question, self.top_k, self.score_threshold)

        context = self._retrieve(question, where=where)
        system_prompt, user_prompt = self._build_prompt(question, context)
        answer = self._generate(system_prompt, user_prompt)

        sources = list(dict.fromkeys(r.source for r in context if r.source))
        logger.info("Answer generated (%d chars) from %d chunk(s).", len(answer), len(context))

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context=context,
        )

    # ------------------------------------------------------------------
    # Pipeline stages (override individually in subclasses)
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        question: str,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Stage 1 — embed the question and fetch top-k chunks."""
        query_vector = self.embedder.encode_query(question)
        results = self.store.search(query_vector, top_k=self.top_k, where=where)

        if self.score_threshold > 0.0:
            before = len(results)
            results = [r for r in results if r.score >= self.score_threshold]
            dropped = before - len(results)
            if dropped:
                logger.debug("Dropped %d chunk(s) below score_threshold=%.2f.", dropped, self.score_threshold)

        if not results:
            logger.warning("No chunks passed the score threshold — answer will lack context.")

        return results

    def _build_prompt(
        self,
        question: str,
        context: list[SearchResult],
    ) -> tuple[str, str]:
        """Stage 2 — format context chunks and question into a prompt pair."""
        if context:
            context_block = _build_context_block(context)
        else:
            context_block = "(No relevant context was found in the knowledge base.)"

        user_prompt = _USER_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )
        return _SYSTEM_PROMPT, user_prompt

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """Stage 3 — call the LLM and return its text reply."""
        return self.llm.complete(system=system_prompt, user=user_prompt)

    def stream_query(
        self,
        question: str,
        where: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Like :meth:`query` but yields LLM reply tokens as they arrive.

        Useful for streaming HTTP responses (e.g. SSE to a browser).
        Retrieval and prompt-building happen synchronously before the first
        token is yielded.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        logger.info("RAG stream_query: %r  (top_k=%d, threshold=%.2f)", question, self.top_k, self.score_threshold)
        context = self._retrieve(question, where=where)
        system_prompt, user_prompt = self._build_prompt(question, context)
        yield from self.llm.stream(system=system_prompt, user=user_prompt)
