import pytest

import numpy as np
from types import SimpleNamespace


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

class Frag:
    def __init__(self, content):
        self.content = content


class FakeAgent:
    def __init__(self):
        self.acted, self.printed, self.speech = [], [], []
        self.cleared = 0
        self.flush_count = 0

    def act(self, prompt):
        self.acted.append(prompt)

    def model_fragments(self, prompt):
        return [Frag("stream")]  # tools-off path

    def print_fragment(self, frag):
        self.printed.append(frag.content)

    def flush_stream(self):
        self.flush_count += 1

    def clear_chat(self):
        self.cleared += 1

    def allow_speech(self, enable):
        self.speech.append(enable)

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
