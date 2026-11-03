"""Speech-to-text tool (port of smolagents v1.26.0 ``SpeechToTextTool``).

Transcribes local audio files through ``faster-whisper``. The heavy stack is
imported lazily on first use (the same pattern as ``src.tools.speech`` for
piper), and tests override the ``_STT_FACTORY`` seam. ``configure`` reads the
optional ``[stt]`` section (``model_size``, ``device``).
"""
from __future__ import annotations

import os
from logging import getLogger

__all__ = ["configure", "transcribe_audio"]

_config_cache = None       # [stt] section injected via configure()
_model_cache = None        # cached WhisperModel instance (None = not loaded)
_STT_FACTORY = None        # test seam: factory(section) -> transcriber

_MAX_FILE_BYTES = 50 * 1024 * 1024  # refuse absurd inputs before loading


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Only the ``[stt]`` section is kept; a missing section
    means the faster-whisper defaults apply. Resets the cached model so a
    config change is honored on the next call.
    """
    global _config_cache, _model_cache
    _config_cache = getattr(config, "stt", None)
    _model_cache = None


def _get_transcriber():
    """Load (and cache) the faster-whisper model for the configured section."""
    global _model_cache
    if _STT_FACTORY is not None:  # test seam
        return _STT_FACTORY()
    if _model_cache is not None:
        return _model_cache
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed; "
            "install it to use transcribe_audio") from exc
    model_size = getattr(_config_cache, "model_size", "base")
    device = getattr(_config_cache, "device", "cpu")
    getLogger(__name__).info(
        "Loading faster-whisper model %r on %r ...", model_size, device)
    _model_cache = WhisperModel(model_size, device=device)
    return _model_cache


def transcribe_audio(path: str, language: str | None = None) -> str:
    """
    Transcribe a local audio file (wav/mp3/flac/...) into text.

    Args:
        path: Path of the audio file to transcribe.
        language: Optional ISO language hint (e.g. "en"); auto-detected
            when omitted.

    Returns:
        The recognized text as a single string. "Error:" lines describe
        missing files/dependencies; transcription failures return an
        "Error:" line instead of raising to the model.
    """
    path = str(path or "").strip()
    if not path:
        return "Error: no audio file given"
    if not os.path.isfile(path):
        return f"Error: audio file not found: {path}"
    if os.path.getsize(path) > _MAX_FILE_BYTES:
        return (f"Error: audio file too large "
                f"({os.path.getsize(path)} bytes, limit {_MAX_FILE_BYTES})")
    try:
        model = _get_transcriber()
        segments, _info = model.transcribe(
            path, language=language or None)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text or "(no speech recognized)"
    except RuntimeError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: transcription failed: {exc}"
