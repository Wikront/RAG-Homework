"""
scripts/ingest.py — One-shot knowledge-base ingestion pipeline.

Run directly:
    python scripts/ingest.py

What it does:
    1. Scans data/pdf/   for *.pdf files  → extracts text via PDFLoader
    2. Scans data/audio/ for *.mp3/*.wav  → transcribes via AudioLoader
    3. Chunks each document with TextChunker
    4. Embeds chunks with EmbeddingModel
    5. Upserts into ChromaDB via ChromaStore

Already-indexed documents (identified by source path) are skipped so the
script is safe to re-run after adding new files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when run as `python scripts/ingest.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.chunking import TextChunker
from src.config import get_config
from src.embeddings import EmbeddingModel, OpenAIEmbeddingModel, create_embedding_model
from src.loaders import AudioLoader, PDFLoader
from src.vector_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# ── Tunables ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
AUDIO_EXTENSIONS = {".mp3", ".wav", ".mp4"}
PDF_EXTENSION = ".pdf"


def _build_store(cfg) -> ChromaStore:
    return ChromaStore(
        collection_name=cfg.collection_name,
        persist_dir=cfg.chroma_persist_dir,
        host=cfg.chroma_host,
        port=cfg.chroma_port,
    )


def ingest_pdf(path: Path, store: ChromaStore, chunker: TextChunker, embedder: EmbeddingModel | OpenAIEmbeddingModel) -> int:
    """Load, chunk, embed, and store a single PDF.  Returns number of new chunks added."""
    source = str(path)
    if store.has_source(source):
        logger.info("SKIP  %s  (already indexed)", path.name)
        return 0

    logger.info("PDF   %s", path.name)
    text = PDFLoader(path).load()
    if not text.strip():
        logger.warning("      No text extracted from %s — skipping.", path.name)
        return 0

    chunks = chunker.split(text, source=source, extra={"filename": path.name, "type": "pdf"})
    vectors = embedder.encode([c.text for c in chunks])
    store.add(chunks, vectors, filename=path.name)
    logger.info("      %d chunks indexed.", len(chunks))
    return len(chunks)


def ingest_audio(
    path: Path,
    store: ChromaStore,
    chunker: TextChunker,
    embedder: EmbeddingModel | OpenAIEmbeddingModel,
    transcripts_dir: Path,
) -> int:
    """Transcribe, chunk, embed, and store a single audio file.  Returns new chunk count."""
    source = str(path)
    if store.has_source(source):
        logger.info("SKIP  %s  (already indexed)", path.name)
        return 0

    logger.info("AUDIO %s", path.name)
    text = AudioLoader(path, backend="huggingface", transcripts_dir=transcripts_dir).load()
    if not text.strip():
        logger.warning("      No transcript produced for %s — skipping.", path.name)
        return 0

    chunks = chunker.split(text, source=source, extra={"filename": path.name, "type": "audio"})
    vectors = embedder.encode([c.text for c in chunks])
    store.add(chunks, vectors, filename=path.name)
    logger.info("      %d chunks indexed.", len(chunks))
    return len(chunks)


def main() -> None:
    cfg = get_config()

    pdf_dir = cfg.data_dir / "pdf"
    audio_dir = cfg.data_dir / "audio"
    transcripts_dir = cfg.data_dir / "transcripts"

    for d in (pdf_dir, audio_dir, transcripts_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to ChromaDB …")
    store = _build_store(cfg)

    logger.info(
        "Loading embedding model '%s' (backend=%s) …",
        cfg.embedding_model, cfg.embedding_backend,
    )
    embedder = create_embedding_model(cfg)
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    pdf_files = sorted(pdf_dir.glob(f"*{PDF_EXTENSION}"))
    audio_files = sorted(
        f for f in audio_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
    )

    total_docs = len(pdf_files) + len(audio_files)
    if total_docs == 0:
        logger.warning(
            "No documents found in %s or %s. "
            "Add PDF files to data/pdf/ and audio files to data/audio/.",
            pdf_dir, audio_dir,
        )
        return

    logger.info("Found %d PDF(s) and %d audio file(s).", len(pdf_files), len(audio_files))

    total_chunks = 0

    for path in tqdm(pdf_files, desc="PDFs", unit="file"):
        total_chunks += ingest_pdf(path, store, chunker, embedder)

    for path in tqdm(audio_files, desc="Audio", unit="file"):
        total_chunks += ingest_audio(path, store, chunker, embedder, transcripts_dir)

    logger.info(
        "Ingestion complete. %d new chunk(s) added. "
        "Collection '%s' now holds %d chunk(s) total.",
        total_chunks, cfg.collection_name, store.count,
    )


if __name__ == "__main__":
    main()
