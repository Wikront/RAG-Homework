"""
chat.py — Interactive CLI for the RAG chatbot.

Usage:
    python -m src.chat                         # default collection + model
    python -m src.chat --collection my-docs    # custom Chroma collection
    python -m src.chat --show-sources          # print source filenames
    python -m src.chat --show-context          # print retrieved passages
    python -m src.chat --top-k 8               # retrieve more chunks
    python -m src.chat --help

Exit cleanly with Ctrl-C or by typing 'exit' / 'quit'.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv

from src.config import get_config
from src.embeddings import create_embedding_model
from src.rag import OpenAIClient, RAGPipeline, RAGResponse
from src.vector_store import ChromaStore

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# ANSI codes — silently disabled on Windows or when not writing to a terminal.
_USE_COLOR = sys.stdout.isatty() and os.name != "nt"

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BOLD   = lambda t: _c("1", t)
DIM    = lambda t: _c("2", t)
CYAN   = lambda t: _c("36", t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)

_TERM_WIDTH = min(os.get_terminal_size().columns if _USE_COLOR else 88, 100)

DIVIDER       = DIM("─" * _TERM_WIDTH)
THICK_DIVIDER = DIM("━" * _TERM_WIDTH)

PROMPT_SYMBOL = CYAN("you") + " › "
ANSWER_HEADER = GREEN("assistant") + " › "


def _print_answer(response: RAGResponse, show_sources: bool, show_context: bool) -> None:
    """Render one RAG response to stdout."""
    print()
    print(ANSWER_HEADER)

    # Wrap the answer to terminal width, indented by two spaces.
    for line in response.answer.splitlines():
        wrapped = textwrap.fill(line or " ", width=_TERM_WIDTH - 2, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

    if show_sources and response.sources:
        print()
        print(DIM("  Sources:"))
        for src in response.sources:
            print(DIM(f"    • {src}"))

    if show_context and response.context:
        print()
        print(DIM("  Retrieved passages:"))
        for i, hit in enumerate(response.context, start=1):
            score_str = DIM(f"score={hit.score:.3f}")
            src_str   = DIM(f"  {hit.source}") if hit.source else ""
            header    = DIM(f"  [{i}] {score_str}{src_str}")
            snippet   = textwrap.shorten(hit.text, width=120, placeholder="…")
            print(f"{header}\n  {DIM(snippet)}")

    print()
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.chat",
        description="Interactive RAG chatbot CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--collection",    default=None,  help="Chroma collection name (default: from COLLECTION_NAME env).")
    p.add_argument("--chroma-dir",    default=None,  help="Chroma local persist dir (default: from CHROMA_PERSIST_DIR env).")
    p.add_argument("--llm-model",     default=None,  help="Override LLM model (default: from LLM_MODEL_NAME env).")
    p.add_argument("--top-k",         default=5,   type=int,                 help="Number of chunks to retrieve.")
    p.add_argument("--threshold",     default=0.0, type=float,               help="Minimum similarity score [0–1].")
    p.add_argument("--show-sources",  action="store_true",                   help="Print source filenames after each answer.")
    p.add_argument("--show-context",  action="store_true",                   help="Print retrieved passages after each answer.")
    p.add_argument("--log-level",     default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],            help="Logging verbosity.")
    return p


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Initialise the pipeline and start the REPL."""
    load_dotenv()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    cfg = get_config()
    collection = args.collection or cfg.collection_name
    llm_model  = args.llm_model  or cfg.llm_model_name

    print(THICK_DIVIDER)
    print(BOLD("  RAG Chatbot"))
    print(DIM(f"  collection={collection}  model={llm_model}  top_k={args.top_k}"))
    print(DIM(f"  embedding={cfg.embedding_backend}:{cfg.embedding_model}"))
    print(DIM("  Type 'exit' or press Ctrl-C to quit."))
    print(THICK_DIVIDER)
    print()

    print(DIM("  Loading embedding model…"), end="", flush=True)
    embedder = create_embedding_model(cfg)
    print(DIM(f" done (dim={embedder.dim})."))

    print(DIM(f"  Connecting to Chroma collection '{collection}'…"), end="", flush=True)
    store = ChromaStore(
        collection_name=collection,
        persist_dir=args.chroma_dir or cfg.chroma_persist_dir,
        host=cfg.chroma_host,
        port=cfg.chroma_port,
    )
    print(DIM(f" done ({store.count} chunks indexed)."))

    if store.count == 0:
        print()
        print(YELLOW("  Warning: the collection is empty. Index some documents before querying."))

    llm = OpenAIClient(model=llm_model, base_url=cfg.llm_api_base, api_key=cfg.llm_api_key)
    pipeline = RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k,
        score_threshold=args.threshold,
    )

    print()
    print(DIVIDER)

    while True:
        try:
            question = input(f"\n{PROMPT_SYMBOL}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{DIM('  Goodbye.')}\n")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print(f"\n{DIM('  Goodbye.')}\n")
            break

        try:
            response = pipeline.query(question)
        except ValueError as exc:
            print(f"\n{RED(f'  Error: {exc}')}\n")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\n{RED(f'  Unexpected error: {exc}')}\n")
            logging.getLogger(__name__).exception("Unhandled exception during query.")
            continue

        _print_answer(response, show_sources=args.show_sources, show_context=args.show_context)


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
