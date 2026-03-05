FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg  : audio decoding required by Whisper / transformers
# build-essential : C extensions needed by some Python packages (e.g. chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of code.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Project source ────────────────────────────────────────────────────────────
COPY src/       ./src/
COPY scripts/   ./scripts/

# ── Data directories (overridden at runtime by the volume mount) ──────────────
RUN mkdir -p data/pdf data/audio data/transcripts

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
