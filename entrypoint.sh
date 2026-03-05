#!/usr/bin/env bash
set -euo pipefail

# ── Wait for ChromaDB ─────────────────────────────────────────────────────────
CHROMA_HOST="${CHROMA_HOST:-chroma}"
CHROMA_PORT="${CHROMA_PORT:-8000}"

if [ -n "$CHROMA_HOST" ]; then
    echo "[entrypoint] Waiting for ChromaDB at ${CHROMA_HOST}:${CHROMA_PORT} …"
    retries=0
    until curl -sf "http://${CHROMA_HOST}:${CHROMA_PORT}/api/v2/heartbeat" > /dev/null; do
        retries=$((retries + 1))
        if [ "$retries" -ge 30 ]; then
            echo "[entrypoint] ERROR: ChromaDB did not become ready after 60 s. Aborting."
            exit 1
        fi
        sleep 2
    done
    echo "[entrypoint] ChromaDB is up."
fi

# ── Ingest knowledge base in the background ───────────────────────────────────
# Run ingest concurrently so the API is reachable immediately.
# Documents indexed while the API is already serving will be available for
# retrieval as soon as their chunks land in ChromaDB.
echo "[entrypoint] Starting ingestion pipeline in background …"
python scripts/ingest.py &
INGEST_PID=$!

# ── Launch API server (foreground) ────────────────────────────────────────────
echo "[entrypoint] Starting RAG API server on :8080 …"
uvicorn src.api:app --host 0.0.0.0 --port 8080 --log-level info &
API_PID=$!

# Wait for both processes; exit with non-zero if either fails.
wait $INGEST_PID || { echo "[entrypoint] Ingestion failed."; kill $API_PID 2>/dev/null; exit 1; }
echo "[entrypoint] Ingestion complete."

# Keep the container alive on the API process.
wait $API_PID
