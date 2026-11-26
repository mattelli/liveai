import importlib

import pytest

import numpy as np
from types import SimpleNamespace


# Prefix of the skip reason used by require_import(). The CI workflow
# allowlists exactly this prefix, so a legitimate "no audio stack" skip does not
# fail the "no skipped tests" guard (see .github/workflows/tests.yml).
AUDIO_SKIP_REASON = "audio stack unavailable"


def require_import(*names):
    """Skip the current test/module when any of ``names`` cannot be imported.

    A tolerant ``pytest.importorskip``. That helper only skips on
    ``ModuleNotFoundError``, but ``sounddevice`` initializes PortAudio at import
    time, so on a machine with no audio server (a headless CI runner) it raises
    ``PortAudioError`` instead - which would surface as a collection error or a
    test failure rather than a skip. Any import failure means the same thing
    here: the capability is unavailable, so the test cannot run.
    """
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any import failure means skip
            pytest.skip(
                f"{AUDIO_SKIP_REASON}: cannot import {name!r}: {exc}",
                allow_module_level=True,
            )


@pytest.fixture(autouse=True)
def _drain_background_results():
    """Keep background results from leaking between tests.

    Background workers (scheduler tasks) queue answers
    into a module-level list consumed by the session on its next turn; a
    leftover entry from one test would crash or pollute an unrelated test's
    fake agent.
    """
    try:
        from src.live_ai import drain_agent_results
    except ImportError:  # pragma: no cover - live_ai always exists
        yield
        return
    drain_agent_results()  # clear anything left from a previous test
    yield
    drain_agent_results()


@pytest.fixture(autouse=True)
def _reset_model_meta_cache():
    """Keep the process-wide model-metadata cache from leaking between tests.

    Brain._fetch_model_metadata caches per (base_url, model) process-wide so
    the main agent and its pooled workers share one fetch; tests must start
    with a cold cache to exercise the real fetch paths.
    """
    import src.live_ai as live_ai
    live_ai._MODEL_META_CACHE.clear()
    yield
    live_ai._MODEL_META_CACHE.clear()


class FakeDevice:
    """Stand-in for InputAudioDevice; no hardware touched."""

    name = "Fake Mic"
    index = 0
    default_samplerate = 16000
    max_channel = 1


class FakeStream:
    """Context-managed InputStream double.

    chunks: iterable of fixed-size numpy arrays returned by read().
    Raises StopIteration when exhausted (the caller must bound the loop).
    """

    def __init__(self, chunks):
        self._it = iter(chunks)
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, chunk_size):
        self.read_count += 1
        return next(self._it), False


class FakeWhisperModel:
    """Returns one fixed segment so transcribe() can be tested without weights."""

    def __init__(self, text="fake transcript"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio, language=None, beam_size=None):
        self.calls += 1
        return [SimpleNamespace(text=self.text)], None


class FakeAudioChunk:
    """Stands in for piper.voice.AudioChunk."""

    def __init__(self, data: bytes = b"\x00\x00"):
        self.audio_int16_bytes = data


class FakePiperVoice:
    """Stand-in for PiperVoice; records calls, never loads a model."""

    def __init__(self, chunks=(b"\x01\x02", b"\x01\x02"),
                 sample_rate=22050, fail_after=None, wav_bytes=b"\x01\x02"):
        self._chunks = list(chunks)
        self.sample_rate = sample_rate
        self.config = SimpleNamespace(sample_rate=sample_rate)
        self._fail_after = fail_after
        self._wav_bytes = wav_bytes
        self.synth_calls = 0
        self.wav_calls = 0
        self.last_text = None

    def synthesize(self, text, syn_config=None):
        self.synth_calls += 1
        self.last_text = text
        for i, data in enumerate(self._chunks):
            if self._fail_after == i:
                raise RuntimeError("synth failed")
            yield FakeAudioChunk(data)

    def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True):
        self.wav_calls += 1
        self.last_text = text
        if set_wav_format:
            wav_file.setparams((1, 2, self.sample_rate, 0, "NONE", "not compressed"))
        wav_file.writeframes(self._wav_bytes * 2)


class FakeOutputStream:
    """Stand-in for sounddevice.RawOutputStream."""

    def __init__(self, **kw):
        self.kw = kw
        self.started = 0
        self.stopped = 0
        self.aborted = 0
        self.closed = 0
        self.written = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def abort(self):
        """Drop buffered audio: immediate stop without draining."""
        self.aborted += 1

    def write(self, data):
        self.written.append(data)

    def close(self):
        self.closed += 1


class Frag:
    def __init__(self, content):
        self.content = content


class FakeAgent:
    def __init__(self):
        self.acted, self.printed = [], []
        self.flush_count = 0

    def act(self, prompt):
        self.acted.append(prompt)

    def model_fragments(self, prompt):
        return [Frag("stream")]  # tools-off path

    def print_fragment(self, frag):
        self.printed.append(frag.content)

    def flush_stream(self):
        self.flush_count += 1

    def model_info(self):
        return "fake-model-info"


@pytest.fixture
def fake_agent():
    return FakeAgent()


@pytest.fixture
def pt_input():
    """A prompt_toolkit pipe input installed as the active app session."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            yield pipe


def make_session(pipe, **kw):
    """PromptSession bound to the test pipe, never touching the real TTY."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.output import DummyOutput

    return PromptSession(
        input=pipe,
        output=DummyOutput(),
        history=kw.pop("history", InMemoryHistory()),
        **kw,
    )


@pytest.fixture
def prompts():
    """Returns a helper: prompts(["a", None]) -> a read_prompt callable."""

    def _make(values):
        it = iter(values)
        return lambda: next(it)

    return _make
