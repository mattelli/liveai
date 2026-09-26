import pytest


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
