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
def prompts():
    """Returns a helper: prompts(["a", None]) -> a read_prompt callable."""

    def _make(values):
        it = iter(values)
        return lambda: next(it)

    return _make
