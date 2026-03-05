"""
scripts/test_rag.py — Smoke-test the RAG API with a fixed set of questions.

Usage:
    python scripts/test_rag.py                        # default: http://localhost:8080
    python scripts/test_rag.py --url http://host:8080 # custom endpoint
    python scripts/test_rag.py --out my_results.log   # custom output file
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

QUESTIONS = [
    "What are the production 'Do's' for RAG?",
    "What is the difference between standard retrieval and the ColPali approach?",
    "Why is hybrid search better than vector-only search?",
]


def ask(base_url: str, question: str) -> str:
    payload = json.dumps({
        "model": "rag",
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())

    return body["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG smoke-test")
    parser.add_argument("--url", default="http://localhost:8080", help="RAG API base URL")
    parser.add_argument("--out", default="rag_test_results.log", help="Output log file")
    args = parser.parse_args()

    out_path = Path(args.out)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "=" * 72,
        f"RAG smoke-test  —  {timestamp}",
        f"Endpoint: {args.url}",
        "=" * 72,
        "",
    ]

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        try:
            answer = ask(args.url, question)
        except urllib.error.URLError as exc:
            answer = f"ERROR: could not reach API — {exc}"
            print(f"  ✗ {answer}")
        else:
            print(f"  ✓ received {len(answer)} chars")

        lines += [
            f"── Q{i} ──────────────────────────────────────────────────────────",
            f"Q: {question}",
            "",
            f"A: {answer}",
            "",
        ]

    lines += ["=" * 72, ""]
    output = "\n".join(lines)

    out_path.write_text(output, encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
