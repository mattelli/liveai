"""Graceful allow_speech(True) degradation when Piper is missing (T16 addendum).

`allow_speech(True)` must return an "Error: ..." string instead of raising
when the Piper voice model file or the piper package is unavailable.
"""
import sys
from types import SimpleNamespace

import pytest

import src.tools.speech as speech_mod


@pytest.fixture
def speech_seam(monkeypatch):
    monkeypatch.setattr(speech_mod, "_speech_generator", None)
    monkeypatch.setattr(speech_mod, "_SPEECH_FACTORY", None)
    monkeypatch.setattr(speech_mod, "_config_cache", None)
    return speech_mod


def test_allow_speech_missing_model_returns_error_string(speech_seam):
    """Missing Piper voice model: 'Error: ...' string, no exception raised."""

    class ExplodingFactory:
        def __init__(self, **kw):
            raise FileNotFoundError("Voice model file not found: m")

    speech_seam._SPEECH_FACTORY = ExplodingFactory
    speech_seam.configure(SimpleNamespace(speech_gen=SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.5, speech_speed=1.0)))

    result = speech_seam.allow_speech(True)
    assert result.startswith("Error:")
    assert "Voice model file not found" in result
    assert speech_seam._speech_generator is None  # nothing half-initialized


def test_allow_speech_missing_package_returns_error_string(speech_seam):
    """Missing piper package: 'Error: ...' string, no exception raised."""
    saved = sys.modules.get("src.audio_generator")
    sys.modules["src.audio_generator"] = None  # import -> ImportError

    try:
        speech_seam.configure(SimpleNamespace(speech_gen=SimpleNamespace(
            model_path="m", samplerate=22050, volume=0.5, speech_speed=1.0)))
        result = speech_seam.allow_speech(True)
        assert result.startswith("Error:")
        assert "speech generation unavailable" in result
        assert speech_seam._speech_generator is None
    finally:
        if saved is not None:
            sys.modules["src.audio_generator"] = saved
        else:
            sys.modules.pop("src.audio_generator", None)


def test_allow_speech_still_initializes_when_available(speech_seam):
    """Regression guard: normal path still returns the success message."""

    class Recorder:
        def __init__(self, **kw):
            pass

        def close(self):
            pass

    speech_seam._SPEECH_FACTORY = Recorder
    speech_seam.configure(SimpleNamespace(speech_gen=SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.5, speech_speed=1.0)))

    assert speech_seam.allow_speech(True) == "Speech Generator initialized"
    assert speech_seam.allow_speech(False) == "Speech Generator released"
