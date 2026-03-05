"""
loaders.py — Concrete document loaders.

Currently implemented:
    PDFLoader   — extracts plain text from a PDF file using pypdf.
    AudioLoader — transcribes audio files (.mp3 / .wav) via Whisper and
                  saves the transcript to a transcripts/ directory.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import pypdf

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load a PDF file and extract its text content.

    Args:
        path: Path to the PDF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path does not point to a .pdf file.

    Example:
        # loader = PDFLoader("report.pdf")
        # text = loader.load()
        # print(text[:200])
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")
        if self.path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {self.path.suffix}")

    def load(self) -> str:
        """Extract and return all text from the PDF as a single string.

        Pages are separated by a form-feed character (``\\f``).
        Pages that yield no text (e.g. scanned images) are skipped with a
        warning so the rest of the document is still returned.

        Returns:
            Concatenated text of all readable pages.

        Raises:
            RuntimeError: If the PDF cannot be opened or is encrypted.
        """
        pages: list[str] = []

        with open(self.path, "rb") as fh:
            reader = pypdf.PdfReader(fh)

            if reader.is_encrypted:
                raise RuntimeError(f"PDF is encrypted and cannot be read: {self.path}")

            total = len(reader.pages)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # pragma: no cover
                    logger.warning("Page %d/%d could not be parsed: %s", i + 1, total, exc)
                    continue

                if not text.strip():
                    logger.warning("Page %d/%d contains no extractable text (skipped).", i + 1, total)
                    continue

                pages.append(text)

        return "\f".join(pages)


# ---------------------------------------------------------------------------
# Audio loader
# ---------------------------------------------------------------------------

SUPPORTED_AUDIO = {".mp3", ".wav", ".mp4"}
LONG_FILE_THRESHOLD_S = 60  # warn + time any file longer than this

WhisperBackend = Literal["openai", "huggingface"]


class AudioLoader:
    """Transcribe an audio file using OpenAI Whisper or a Hugging Face Whisper model.

    The transcript is returned as a string **and** written to a .txt file
    inside a ``transcripts/`` directory that is created automatically next to
    the audio file.

    Args:
        path:       Path to the audio file (.mp3 or .wav).
        backend:    ``"openai"`` — calls the OpenAI Whisper API (requires
                    ``OPENAI_API_KEY`` in the environment).
                    ``"huggingface"`` — runs a local ``openai/whisper-*``
                    model via the ``transformers`` pipeline (no API key needed).
        model:      Model name / size.
                    OpenAI backend: ``"whisper-1"`` (only option via API).
                    HF backend: any Whisper checkpoint on the Hub, e.g.
                    ``"openai/whisper-base"`` or ``"openai/whisper-large-v3"``.
        transcripts_dir: Where to save .txt transcripts.  Defaults to a
                    ``transcripts/`` folder next to the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError:        If the file extension is not .mp3 or .wav.

    Example:
        # loader = AudioLoader("interview.mp3", backend="huggingface")
        # text = loader.load()
        # print(text[:300])
    """

    def __init__(
        self,
        path: str | Path,
        backend: WhisperBackend = "huggingface",
        model: str = "openai/whisper-base",
        transcripts_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_AUDIO:
            raise ValueError(
                f"Unsupported audio format '{self.path.suffix}'. "
                f"Expected one of: {SUPPORTED_AUDIO}"
            )

        self.backend: WhisperBackend = backend
        self.model = model
        self.transcripts_dir = (
            Path(transcripts_dir)
            if transcripts_dir
            else self.path.parent / "transcripts"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> str:
        """Transcribe the audio file and persist the transcript.

        Returns:
            The full transcription as a plain string.

        Raises:
            RuntimeError: If the chosen backend is unavailable or the
                          transcription call fails.
        """
        file_size_mb = self.path.stat().st_size / (1024 ** 2)
        logger.info("Transcribing '%s' (%.1f MB) with backend=%s model=%s",
                    self.path.name, file_size_mb, self.backend, self.model)

        is_long = self._is_long_file()
        if is_long:
            logger.warning(
                "'%s' exceeds the %d-second threshold — transcription may take a while.",
                self.path.name, LONG_FILE_THRESHOLD_S,
            )

        t0 = time.monotonic()

        if self.backend == "openai":
            transcript = self._transcribe_openai()
        else:
            transcript = self._transcribe_huggingface()

        elapsed = time.monotonic() - t0
        logger.info("Transcription finished in %.1f s (%d chars).", elapsed, len(transcript))

        self._save(transcript)
        return transcript

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_long_file(self) -> bool:
        """Return True if the audio duration exceeds LONG_FILE_THRESHOLD_S.

        Uses mutagen when available for an exact duration; falls back to a
        file-size heuristic (128 kbps) so mutagen is not a hard dependency.
        """
        try:
            from mutagen import File as MutagenFile  # optional dep

            audio = MutagenFile(self.path)
            if audio is not None and audio.info is not None:
                return audio.info.length > LONG_FILE_THRESHOLD_S
        except ImportError:
            pass

        # Heuristic: assume ~128 kbps average bitrate
        estimated_seconds = (self.path.stat().st_size * 8) / 128_000
        return estimated_seconds > LONG_FILE_THRESHOLD_S

    def _transcribe_openai(self) -> str:
        """Call the OpenAI Whisper API."""
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for the 'openai' backend. "
                "Install it with: pip install openai"
            ) from exc

        client = openai.OpenAI()  # reads OPENAI_API_KEY from env
        with open(self.path, "rb") as fh:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=fh,
                response_format="text",
            )
        return response if isinstance(response, str) else response.text

    def _transcribe_huggingface(self) -> str:
        """Run a local Whisper model via faster-whisper (CTranslate2, int8 quantized).

        faster-whisper is ~4x faster than the HuggingFace pipeline on CPU and
        handles mp4/mp3/wav natively through its internal ffmpeg bindings.
        VAD filtering skips silence, giving another large speedup on lecture recordings.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required for the 'huggingface' backend. "
                "Install it with: pip install faster-whisper"
            ) from exc

        # faster-whisper model names are short (e.g. "base"), not HF paths.
        model_name = self.model
        if "/" in model_name:
            model_name = model_name.rsplit("/", 1)[-1]          # openai/whisper-base → whisper-base
        if model_name.startswith("whisper-"):
            model_name = model_name[len("whisper-"):]           # whisper-base → base

        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(self.path),
            beam_size=1,        # greedy decoding — faster, minimal accuracy loss
            vad_filter=True,    # skip silent segments — big speedup on lecture videos
        )
        return " ".join(seg.text.strip() for seg in segments)

    def _save(self, text: str) -> Path:
        """Write *text* to a .txt file in ``self.transcripts_dir``.

        Returns:
            Path to the saved transcript file.
        """
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.transcripts_dir / (self.path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        logger.info("Transcript saved to '%s'.", out_path)
        return out_path
