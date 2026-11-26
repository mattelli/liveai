"""Contract of the shared test helper ``require_import`` (see conftest).

``sounddevice`` initializes PortAudio at import time, so on a machine with no
audio server (a headless CI runner) importing it raises ``PortAudioError``
rather than ``ModuleNotFoundError``. ``pytest.importorskip`` only skips on the
latter, so the audio modules used to fail at *collection* instead of skipping.
``require_import`` treats any import failure as "capability unavailable".
"""
import pytest

from tests import conftest


def test_require_import_returns_when_importable():
    conftest.require_import("json")  # must not raise


def test_require_import_skips_on_missing_module(monkeypatch):
    with pytest.raises(pytest.skip.Exception) as excinfo:
        conftest.require_import("definitely_not_a_real_module_xyz")
    assert conftest.AUDIO_SKIP_REASON in str(excinfo.value)


def test_require_import_skips_on_non_import_error(monkeypatch):
    """The headless-runner case: a non-ImportError raised while importing."""

    class PortAudioError(Exception):
        pass

    def boom(name):
        raise PortAudioError(
            "Error initializing PortAudio: Unanticipated host error "
            "[PaErrorCode -9999]: 'PulseAudio_Initialize: Can't connect to "
            "server'"
        )

    monkeypatch.setattr(conftest.importlib, "import_module", boom)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        conftest.require_import("sounddevice")

    message = str(excinfo.value)
    assert conftest.AUDIO_SKIP_REASON in message
    assert "PulseAudio_Initialize" in message


def test_skip_reason_prefix_matches_the_ci_allowlist():
    """The workflow allowlists this exact prefix, so it must not drift.

    .github/workflows/tests.yml greps 'audio stack unavailable' out of the
    SKIPPED lines to tell a legitimate capability skip from a missing package.
    """
    workflow = (
        conftest.__file__.rsplit("tests", 1)[0]
        + ".github/workflows/tests.yml"
    )
    with open(workflow, encoding="utf-8") as handle:
        assert conftest.AUDIO_SKIP_REASON in handle.read()
