import pytest

# audio_wrapper imports sounddevice/faster_whisper at module top; these may be absent.
pytest.importorskip("sounddevice", reason="audio_wrapper requires sounddevice to import")
pytest.importorskip("faster_whisper", reason="audio_wrapper requires faster_whisper to import")

from src.audio_wrapper import format_level  # noqa: E402


def test_format_level_hidden():
    assert format_level(0.5, 0.1, show=False) == ""


def test_format_level_shown():
    s = format_level(0.5, 0.1, show=True)
    assert "0.500" in s and "0.100" in s
