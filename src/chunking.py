"""
chunking.py — Text chunking utilities for RAG pipelines.

Provides:
    Chunk       — dataclass representing a single text chunk with metadata.
    TextChunker — splits raw text into overlapping chunks, optionally
                  respecting sentence boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single piece of text together with its provenance metadata.

    Attributes:
        text:        The chunk's content.
        index:       Zero-based position of this chunk in the sequence
                     produced for a given source.
        source:      Arbitrary identifier for the origin document (file path,
                     URL, DB key, …).  Empty string when not provided.
        extra:       Any additional caller-supplied metadata (page number,
                     section title, …).
        char_start:  Character offset of the first character in the original
                     text (best-effort; -1 when unknown).
    """

    text: str
    index: int
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    char_start: int = -1

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

# Regex that matches sentence-ending punctuation followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


class TextChunker:
    """Split a text string into overlapping chunks for downstream embedding.

    Two splitting strategies are available via the *mode* parameter:

    ``"sentence"`` *(default)*
        Accumulates complete sentences until *chunk_size* would be exceeded,
        then starts a new chunk.  The overlap is filled from the tail of the
        preceding chunk's sentences, so chunk boundaries always fall at
        sentence ends.  Best for natural prose.

    ``"character"``
        Purely mechanical sliding-window split on character count.  Faster and
        predictable, but may cut mid-word or mid-sentence.  Useful for
        structured / non-prose text (code, tables, logs).

    Args:
        chunk_size: Target maximum length of each chunk in characters
                    (default 400).
        overlap:    Number of characters (or sentence-boundary characters in
                    sentence mode) carried over from the previous chunk to
                    provide context continuity (default 50).
        mode:       ``"sentence"`` or ``"character"`` (default ``"sentence"``).

    Example:
        # chunker = TextChunker(chunk_size=400, overlap=50)
        # chunks = chunker.split("Long article text …", source="article.txt")
        # for c in chunks:
        #     print(c.index, c.char_start, c.text[:60])
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        mode: str = "sentence",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if mode not in ("sentence", "character"):
            raise ValueError(f"mode must be 'sentence' or 'character', got '{mode}'")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        text: str,
        source: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split *text* into a list of :class:`Chunk` objects.

        Args:
            text:   The raw input text to split.
            source: Identifier for the origin document (propagated to every
                    chunk's ``source`` field).
            extra:  Additional metadata dict propagated to every chunk's
                    ``extra`` field (e.g. ``{"page": 3}``).

        Returns:
            Ordered list of :class:`Chunk` objects.  Empty list when *text*
            contains only whitespace.
        """
        text = text.strip()
        if not text:
            return []

        extra = extra or {}

        if self.mode == "sentence":
            spans = self._split_sentence(text)
        else:
            spans = self._split_character(text)

        return [
            Chunk(
                text=chunk_text,
                index=i,
                source=source,
                extra=extra,
                char_start=char_start,
            )
            for i, (chunk_text, char_start) in enumerate(spans)
        ]

    # ------------------------------------------------------------------
    # Private: sentence-aware splitting
    # ------------------------------------------------------------------

    def _split_sentence(self, text: str) -> list[tuple[str, int]]:
        """Return (chunk_text, char_start) pairs using sentence boundaries."""
        sentences = _SENTENCE_BOUNDARY.split(text)

        # Re-attach the whitespace that was consumed by the split so that
        # char offsets remain correct when we reassemble chunks.
        # We rebuild them as (sentence_str, offset_in_original) pairs.
        sentence_spans: list[tuple[str, int]] = []
        cursor = 0
        for sent in sentences:
            start = text.index(sent, cursor)
            sentence_spans.append((sent, start))
            cursor = start + len(sent)

        chunks: list[tuple[str, int]] = []
        current_sents: list[tuple[str, int]] = []
        current_len = 0

        for sent, offset in sentence_spans:
            sent_len = len(sent)

            # If a single sentence already exceeds chunk_size, emit it alone.
            if not current_sents and sent_len > self.chunk_size:
                chunks.append((sent, offset))
                continue

            # Would adding this sentence overflow the budget?
            # (+1 for the space separator between sentences)
            separator = 1 if current_sents else 0
            if current_len + separator + sent_len > self.chunk_size and current_sents:
                chunk_text = " ".join(s for s, _ in current_sents)
                chunk_start = current_sents[0][1]
                chunks.append((chunk_text, chunk_start))

                # Seed next chunk with overlap: take sentences from the tail
                # of the current group until we have >= overlap chars.
                overlap_sents: list[tuple[str, int]] = []
                overlap_len = 0
                for s, o in reversed(current_sents):
                    overlap_sents.insert(0, (s, o))
                    overlap_len += len(s)
                    if overlap_len >= self.overlap:
                        break

                current_sents = overlap_sents
                current_len = sum(len(s) for s, _ in current_sents)

            separator = 1 if current_sents else 0
            current_sents.append((sent, offset))
            current_len += separator + sent_len

        # Flush remainder.
        if current_sents:
            chunks.append((" ".join(s for s, _ in current_sents), current_sents[0][1]))

        return chunks

    # ------------------------------------------------------------------
    # Private: fixed-size character splitting
    # ------------------------------------------------------------------

    def _split_character(self, text: str) -> list[tuple[str, int]]:
        """Return (chunk_text, char_start) pairs using a sliding window."""
        step = self.chunk_size - self.overlap
        chunks: list[tuple[str, int]] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append((text[start:end], start))
            if end == len(text):
                break
            start += step
        return chunks
