import pathlib
from logging import getLogger
from types import SimpleNamespace

import pytest

import src.live_ai as live_ai_mod  # noqa: E402


def _bare_brain(speech_gen):
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(speech_gen=speech_gen)
    return brain


def test_allow_speech_forwards_config_and_closes(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)
            self.closed = 0

        def close(self):
            self.closed += 1

    monkeypatch.setattr(live_ai_mod, "SpeechGenerator", Recorder)

    brain = _bare_brain(SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.3, speech_speed=1.5,
        speaker_id=2, noise_scale=0.4, noise_w_scale=0.7, normalize_audio=True,
    ))

    assert brain.allow_speech(True) == "Speech Generator initialized"
    speech = brain.speech
    assert seen["speaker_id"] == 2 and seen["noise_scale"] == 0.4
    assert seen["noise_w_scale"] == 0.7 and seen["normalize_audio"] is True
    assert seen["speech_speed"] == 1.5

    assert brain.allow_speech(False) == "Speech Generator released"
    assert speech.closed == 1
    assert not hasattr(brain, "speech")


def test_allow_speech_without_optional_keys(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)

        def close(self):
            pass

    monkeypatch.setattr(live_ai_mod, "SpeechGenerator", Recorder)

    brain = _bare_brain(SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.5, speech_speed=1.0,
    ))
    brain.allow_speech(True)
    assert seen["speaker_id"] is None
    assert seen["noise_scale"] == 1.0
    assert seen["noise_w_scale"] == 1.0
    assert seen["normalize_audio"] is False


def _bare_summarization_brain(**overrides):
    cfg = dict(model_path="m", device="cpu", ratio=0.1, threshold=0.5, max_sentences=0)
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(summarization=SimpleNamespace(**cfg))
    return brain


def test_summarize_text_reads_file_and_caches(tmp_path, monkeypatch):
    seen = {"calls": [], "count": 0}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def summarize_file(self, path, max_sentences=None):
            seen["calls"].append((str(path), max_sentences))
            return "SUMMARY"

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)

    first = tmp_path / "first.txt"
    first.write_text("long text", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("more text", encoding="utf-8")

    brain = _bare_summarization_brain(ratio=0.2, threshold=0.4, max_sentences=3)
    assert brain.summarize_text(str(first)) == "SUMMARY"
    assert brain.summarize_text(str(second)) == "SUMMARY"

    assert seen["count"] == 1  # summarizer created once and cached
    assert seen["init"]["model_path"] == "m"
    assert seen["init"]["ratio"] == 0.2
    assert seen["init"]["threshold"] == 0.4
    assert seen["init"]["max_sentences"] == 3
    assert seen["calls"] == [(str(first), None), (str(second), None)]


def test_summarize_text_missing_file_returns_message(tmp_path):
    brain = _bare_summarization_brain()
    out = brain.summarize_text(str(tmp_path / "nope.txt"))
    assert "Unable to read" in out


def test_summarize_text_forwards_max_sentences(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize_file(self, path, max_sentences=None):
            seen["max"] = max_sentences
            return "S"

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    brain = _bare_summarization_brain()
    assert brain.summarize_text(str(target), max_sentences=2) == "S"
    assert seen["max"] == 2


def test_summarize_text_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize_file(self, path, max_sentences=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    brain = _bare_summarization_brain()
    out = brain.summarize_text(str(target))
    assert out.startswith("Error:") and "boom" in out


def _bare_document_query_brain(**overrides):
    cfg = dict(model_path="ce-model", device="cpu", chunk_size=3, top_k=5)
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(document_query=SimpleNamespace(**cfg))
    return brain


def test_query_document_reads_config_and_caches(tmp_path, monkeypatch):
    seen = {"init": {}, "count": 0, "calls": []}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def query(self, path, query, top_k=None):
            seen["calls"].append((str(path), query, top_k))
            return "HITS"

    monkeypatch.setattr(live_ai_mod, "DocumentQuery", Recorder)
    target = tmp_path / "doc.txt"
    target.write_text("text", encoding="utf-8")

    brain = _bare_document_query_brain(chunk_size=4, top_k=7)
    assert brain.query_document(str(target), "find x") == "HITS"
    assert brain.query_document(str(target), "find y", top_k=2) == "HITS"

    assert seen["count"] == 1  # created once and cached
    assert seen["init"]["model_path"] == "ce-model"
    assert seen["init"]["chunk_size"] == 4
    assert seen["init"]["top_k"] == 7
    assert seen["calls"] == [(str(target), "find x", None), (str(target), "find y", 2)]


def test_query_document_requires_query(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            raise AssertionError("must not be constructed")

    monkeypatch.setattr(live_ai_mod, "DocumentQuery", Recorder)
    brain = _bare_document_query_brain()
    out = brain.query_document(str(tmp_path / "a.txt"), "   ")
    assert out == "Error: a query is required."


def test_query_document_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, path, query, top_k=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "DocumentQuery", Recorder)
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    brain = _bare_document_query_brain()
    out = brain.query_document(str(target), "q")
    assert out.startswith("Error:") and "boom" in out


def test_agent_build_tools_includes_summarize_text():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "summarize_text" in names


def test_agent_build_tools_includes_query_document():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "query_document" in names


def test_agent_build_tools_keeps_existing_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    for name in ("system_info", "read_text_file", "web_search", "image_response"):
        assert name in names


def _bare_embedding_brain(**overrides):
    cfg = dict(model_path="m", device="cpu", extensions="txt, md",
               chunk_size=3, top_k=5, corpora_dir="/res/corpora")
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(embedding=SimpleNamespace(**cfg))
    return brain


def test_store_corpus_requires_name(monkeypatch):
    import inspect

    params = inspect.signature(live_ai_mod.Brain.store_corpus).parameters
    assert params["corpus_name"].default is inspect.Parameter.empty

    seen = {"store": 0}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, *args, **kwargs):
            seen["store"] += 1
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().store_corpus("/docs", "")
    assert out.startswith("Error: a corpus name is required")
    assert seen["store"] == 0


def test_query_corpus_requires_name(monkeypatch):
    import inspect

    params = inspect.signature(live_ai_mod.Brain.query_corpus).parameters
    assert params["corpus_name"].default is inspect.Parameter.empty

    seen = {"query": 0}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, *args, **kwargs):
            seen["query"] += 1
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().query_corpus("hello", "")
    assert out.startswith("Error: a corpus name is required")
    assert seen["query"] == 0


def test_store_corpus_uses_named_corpus(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, corpus_dir, overwrite=False):
            seen["store"] = (directory, str(corpus_dir), overwrite)
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(corpora_dir=str(tmp_path))
    assert brain.store_corpus("/docs", "autosar", overwrite=True) == "STORED"
    assert seen["store"] == ("/docs", str(tmp_path / "autosar"), True)


def test_store_corpus_invalid_name_returns_error(monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, *args, **kwargs):
            raise AssertionError("store must not be called")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().store_corpus("/docs", "a/b")
    assert out.startswith("Error:") and "corpus name" in out.lower()


def test_query_corpus_resolves_name_and_forwards(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_dir, query, top_k=None, kind=None):
            seen["query"] = (str(corpus_dir), query, top_k, kind)
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(corpora_dir=str(tmp_path))
    assert brain.query_corpus("hello", "autosar", kind="image", top_k=2) == "RESULTS"
    assert seen["query"] == (str(tmp_path / "autosar"), "hello", 2, "image")


def test_list_corpora_formats_entries(monkeypatch):
    entries = [
        {"name": "autosar", "mode": "multimodal", "records": 10, "text_records": 8,
         "image_records": 2, "sources": 3, "built_at": "2026-01-01 00:00:00 UTC"},
        {"name": "quora", "mode": "text", "records": 5, "text_records": 5,
         "image_records": 0, "sources": 1, "built_at": "2026-01-02 00:00:00 UTC"},
    ]
    monkeypatch.setattr(live_ai_mod, "list_corpora", lambda corpora_dir: entries)
    out = _bare_embedding_brain().list_corpora()
    assert "Available corpora (2)" in out
    assert "autosar: mode=multimodal" in out and "quora: mode=text" in out


def test_list_corpora_empty_returns_message(monkeypatch):
    monkeypatch.setattr(live_ai_mod, "list_corpora", lambda corpora_dir: [])
    assert _bare_embedding_brain().list_corpora() == "No corpora available."


def test_store_corpus_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, corpus_dir, overwrite=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain(corpora_dir=str(tmp_path)).store_corpus("/docs", "autosar")
    assert out.startswith("Error:") and "boom" in out


def test_query_corpus_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_dir, query, top_k=None, kind=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain(corpora_dir=str(tmp_path)).query_corpus("hi", "autosar")
    assert out.startswith("Error:") and "boom" in out


def test_agent_build_tools_includes_embedding_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "store_corpus" in names and "query_corpus" in names


def test_agent_build_tools_includes_list_corpora():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "list_corpora" in names


def test_store_corpus_passes_multimodal_config(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def store(self, directory, corpus_dir, overwrite=False):
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(
        corpora_dir=str(tmp_path),
        image_model_path="clip", multimodal=True, image_extensions="png, jpg",
    )
    assert brain.store_corpus("/docs", "autosar") == "STORED"
    assert seen["init"]["image_model_path"] == "clip"
    assert seen["init"]["multimodal"] is True
    assert seen["init"]["image_extensions"] == "png, jpg"
    assert "processed_name" not in seen["init"]


def test_embedding_corpus_defaults_when_config_is_minimal(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def query(self, corpus_dir, query, top_k=None, kind=None):
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(corpora_dir=str(tmp_path))
    brain.query_corpus("hi", "autosar")
    assert seen["init"]["multimodal"] is False
    assert seen["init"]["image_model_path"] is None
    assert "processed_name" not in seen["init"]


# ------------------------------------------------------------- Ctrl+C support


class _BareAgent(live_ai_mod.Agent):
    """Agent built without __init__ (no SDK connection) for unit tests."""

    def __init__(self):  # noqa: D107 - deliberately bypasses Brain.__init__
        self.logger = getLogger("test")
        self.config = SimpleNamespace(brain=SimpleNamespace(model="m"))
        self._model_name = "m"
        self._cancel_requested = False
        self._active_stream = None
        self._fragments_buffer = []
        self._stream = None


def test_cancel_generation_marks_flag_and_cancels_stream():
    class FakeStream:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    agent = _BareAgent()
    stream = FakeStream()
    agent._active_stream = stream

    assert agent.cancel_requested is False
    assert agent.cancel_generation() == "Cancel requested"
    assert agent.cancel_requested is True
    assert stream.close_calls == 1
    assert agent._active_stream is None  # stale handle dropped

    # A cancel without an active stream (act() rounds) still flags the turn.
    agent._cancel_requested = False
    agent.cancel_generation()
    assert agent.cancel_requested is True

    # Cancelling a dead stream must never raise.
    class BrokenStream:
        def close(self):
            raise RuntimeError("already closed")

    agent._active_stream = BrokenStream()
    assert agent.cancel_generation() == "Cancel requested"
    assert agent._active_stream is None


def test_cancelled_round_does_not_speak():
    class FakeSpeech:
        def __init__(self):
            self.spoken = []

        def speak(self, text):
            self.spoken.append(text)

    class FakeStream:
        def flush(self):
            pass

        def mark_line_clean(self):
            pass

        def note(self, text):
            pass

    agent = _BareAgent()
    agent.speech = FakeSpeech()
    agent._messages = []
    agent._context_window = 4096
    agent._stream = FakeStream()
    agent._fragments_buffer = ["partial"]

    agent._cancel_requested = True
    agent._handle_round_end(0)
    assert agent.speech.spoken == []

    agent._cancel_requested = False
    agent._fragments_buffer = ["full"]
    agent._handle_round_end(1)
    assert agent.speech.spoken == ["full"]


# ------------------------------------------------------------ response cap


def test_response_token_cap_defaults_to_full_context_window():
    assert live_ai_mod._response_token_cap(0, 4096) == 4096
    assert live_ai_mod._response_token_cap(None, 8192) == 8192


def test_response_token_cap_is_a_context_window_ratio():
    assert live_ai_mod._response_token_cap(0.1, 4096) == 409
    assert live_ai_mod._response_token_cap(0.25, 4096) == 1024
    assert live_ai_mod._response_token_cap(0.99, 1000) == 990


def test_response_token_cap_is_never_zero():
    # A tiny ratio must still yield a usable cap, not maxTokens = 0.
    assert live_ai_mod._response_token_cap(1e-9, 4096) == 1


@pytest.mark.parametrize(
    "bad", [-0.1, 1, 1.5, 4096, "high", float("nan")]
)
def test_response_token_cap_rejects_values_outside_0_1(bad):
    with pytest.raises(ValueError):
        live_ai_mod._response_token_cap(bad, 4096)


# --------------------------------------------- history repair after cancels


def _msg(role, content):
    return {"role": role, "content": content}


def _bare_repair_agent(messages):
    agent = _BareAgent()
    agent._messages = list(messages)
    return agent


def test_history_repair_pads_empty_assistant_turn():
    agent = _bare_repair_agent(
        [_msg("user", "hello"), _msg("assistant", "")]  # cancelled, no tokens
    )
    agent._add_user_prompt("next question")
    assert agent._messages[-2]["content"] == live_ai_mod.CANCELLED_PLACEHOLDER
    assert agent._messages[-1]["content"] == "next question"


def test_history_repair_drops_unanswered_user_prompts():
    # Trailing unanswered prompts are dropped (all of them: consecutive user
    # messages would fail role alternation anyway).
    agent = _bare_repair_agent(
        [_msg("user", "first"), _msg("user", "never answered")]  # error turns
    )
    agent._add_user_prompt("next question")
    assert [m["content"] for m in agent._messages] == ["next question"]


def test_history_repair_cascades_cancelled_then_errored_turn():
    agent = _bare_repair_agent(
        [
            _msg("user", "ask"),
            _msg("assistant", ""),  # cancelled with no tokens
            _msg("user", "never answered"),  # then this turn errored too
        ]
    )
    agent._add_user_prompt("next question")
    assert [m["content"] for m in agent._messages] == [
        "ask",
        live_ai_mod.CANCELLED_PLACEHOLDER,
        "next question",
    ]


def test_history_repair_keeps_partial_and_normal_history():
    partial = [_msg("user", "ask"), _msg("assistant", "partial text")]
    agent = _bare_repair_agent(list(partial))
    agent._add_user_prompt("next")
    assert [m["content"] for m in agent._messages] == [
        "ask",
        "partial text",
        "next",
    ]
    # a healthy history must be left untouched
    healthy = [_msg("user", "q"), _msg("assistant", "a")]
    agent = _bare_repair_agent(list(healthy))
    agent._add_user_prompt("next")
    assert [m["content"] for m in agent._messages] == ["q", "a", "next"]


class FakeRoundStream:
    """StreamPrinter stand-in for act() round-loop tests (no console output)."""

    def start_spinner(self, note):
        pass

    def stop_spinner(self):
        pass

    def flush(self):
        pass

    def mark_line_clean(self):
        pass

    def note(self, text):
        pass


def _bare_act_agent(rounds):
    """Agent whose completion rounds are scripted in ``rounds``.

    Each item is a ``(text, tool_calls)`` tuple returned by ``_stream_round``;
    ``tool_calls`` are executed against ``tools`` (name -> callable).
    """
    agent = _BareAgent()
    agent._stream = FakeRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    agent._tool_impls = {}
    scripted = iter(rounds)

    def fake_round(tools):
        return next(scripted)

    agent._stream_round = fake_round
    return agent


def test_act_plain_answer_round_appends_history():
    agent = _bare_act_agent([("hello there", [])])
    agent.act("hi")
    assert agent._messages[-1] == {"role": "assistant", "content": "hello there"}


def test_act_tool_round_feeds_result_and_continues():
    calls = []

    def get_time():
        calls.append("get_time")
        return "12:00"

    agent = _bare_act_agent([
        ("", [{"id": "c1", "name": "get_time", "arguments": "{}"}]),
        ("It is 12:00", []),
    ])
    agent._tool_impls = {"get_time": get_time}
    agent.act("What time is it?")
    assert calls == ["get_time"]
    # history: system, user, assistant(tool_calls), tool result, final answer
    assert [m["role"] for m in agent._messages[2:]] == ["assistant", "tool", "assistant"]
    assert agent._messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "12:00"}
    assert agent._messages[-1]["content"] == "It is 12:00"


def test_act_cancelled_empty_round_gets_placeholder():
    """An empty cancelled round must not append an empty assistant turn."""
    agent = _bare_act_agent([("", [])])

    def cancelled_round(tools):
        agent._cancel_requested = True  # Ctrl+C landed during the round
        return "", []

    agent._stream_round = cancelled_round
    agent.act("anything")
    # Empty text would be dropped by the model's prompt template, leaving a
    # user message directly followed by another user message (role alternation
    # error); the placeholder keeps the history renderable.
    assert agent._messages[-1] == {
        "role": "assistant", "content": live_ai_mod.CANCELLED_PLACEHOLDER,
    }


## Remote provider (.env / api key resolution) ##

def _fake_openai(monkeypatch):
    """Stub the openai package so the Brain can be built offline."""
    import sys
    import types

    stub = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            # A well-formed (but unrelated) models list: no exception, so the
            # raw-fallback HTTP request is never attempted in tests.
            self.models = types.SimpleNamespace(
                retrieve=self._raise_unsupported,
                list=lambda: types.SimpleNamespace(
                    data=[types.SimpleNamespace(id="unrelated-model")]),
            )

        @staticmethod
        def _raise_unsupported(model):
            raise RuntimeError("retrieve not supported")

    stub.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", stub)


def _remote_config(**brain_overrides):
    brain = dict(
        model="test-model", system_prompt="s",
        history_path="h", temperature=0.6, max_response_tokens=0.2,
        base_url="https://api.test/v1", api_key_env="TEST_API_KEY",
    )
    brain.update(brain_overrides)
    return SimpleNamespace(brain=SimpleNamespace(**brain))


def test_load_env_file_defaults_to_project_root(monkeypatch):
    pytest.importorskip("dotenv")
    seen = {}

    def fake_load(dotenv_path=None, override=None, **kw):
        seen["path"], seen["override"] = dotenv_path, override
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load)
    live_ai_mod._load_env_file()
    assert seen["path"].name == ".env"
    assert seen["path"].parent == pathlib.Path(live_ai_mod.__file__).resolve().parents[1]
    assert seen["override"] is False  # real environment variables win


def test_load_env_file_tolerates_missing_package(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    live_ai_mod._load_env_file()  # must not raise


def test_brain_key_resolved_from_env_file(monkeypatch):
    """With no shell env var, the key must come from the .env file."""
    pytest.importorskip("dotenv")
    _fake_openai(monkeypatch)

    def fake_load(dotenv_path=None, override=False, **kw):
        monkeypatch.setenv("TEST_API_KEY", "key-from-env")
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load)
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    agent = live_ai_mod.Brain(_remote_config())
    assert agent._client.api_key == "key-from-env"
    assert agent._base_url == "https://api.test/v1"  # explicit [brain] base_url


def test_brain_env_var_wins_over_env_file(monkeypatch):
    pytest.importorskip("dotenv")
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "key-from-shell")
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kw: True)  # file says otherwise

    agent = live_ai_mod.Brain(_remote_config())
    assert agent._client.api_key == "key-from-shell"



def test_brain_missing_key_raises_with_env_name(monkeypatch):
    _fake_openai(monkeypatch)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.setattr(live_ai_mod, "_load_env_file", lambda: None)
    with pytest.raises(ValueError, match="TEST_API_KEY"):
        live_ai_mod.Brain(_remote_config())


def test_brain_missing_base_url_raises(monkeypatch):
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "abc")
    with pytest.raises(ValueError, match="base_url"):
        live_ai_mod.Brain(_remote_config(base_url=""))


def test_brain_missing_api_key_env_raises(monkeypatch):
    """The env var name must be configured ([brain] api_key_env)."""
    _fake_openai(monkeypatch)
    monkeypatch.setattr(live_ai_mod, "_load_env_file", lambda: None)
    with pytest.raises(ValueError, match="API key not found"):
        live_ai_mod.Brain(_remote_config(api_key_env=""))


# ------------------------------------------------------------ model_info


def _bare_info_brain(**overrides):
    """A Brain assembled by hand, carrying only the model_info state."""
    attrs = dict(
        _model_name="test-model",
        _base_url="http://localhost:11434/v1",
        _context_window=4096,
        _context_window_source="config",
        _vision=False,
        _vision_source="config",
        _temperature=0.6,
        _max_tokens=409,
        _messages=[{"role": "system", "content": "sys"},
                   {"role": "user", "content": "hi"}],
    )
    attrs.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain._client = None
    for key, value in attrs.items():
        setattr(brain, key, value)
    return brain


def test_fetch_model_metadata_uses_openai_retrieve():
    """GET /models/{id} with server-specific extras, minus the noise."""
    def retrieve(model):
        assert model == "test-model"
        return SimpleNamespace(model_dump=lambda: {
            "id": "test-model", "object": "model", "created": 1700000000,
            "owned_by": "ollama", "root": "test-model",
            "max_context_length": 131072, "loaded_context_length": 4096,
            "capabilities": ["completion", "vision", "tools"],
            "quantization": "Q4_K_M", "arch": "qwen3",
            "permission": [{"id": "p"}],
        })

    brain = _bare_info_brain(
        _client=SimpleNamespace(models=SimpleNamespace(retrieve=retrieve)))
    meta = brain._fetch_model_metadata()

    assert meta["owned_by"] == "ollama"
    assert meta["created"] == 1700000000
    assert meta["max_context_length"] == 131072
    assert meta["loaded_context_length"] == 4096
    assert meta["capabilities"] == ["completion", "vision", "tools"]
    assert meta["quantization"] == "Q4_K_M"
    assert meta["arch"] == "qwen3"
    assert "permission" not in meta
    assert "root" not in meta and "object" not in meta


def test_fetch_model_metadata_falls_back_to_models_list():
    """Servers without retrieve-by-id (e.g. Ollama) answer via GET /models."""
    def broken_retrieve(model):
        raise RuntimeError("not supported")

    brain = _bare_info_brain(_client=SimpleNamespace(models=SimpleNamespace(
        retrieve=broken_retrieve,
        list=lambda: SimpleNamespace(data=[
            SimpleNamespace(id="other", owned_by="lib"),
            SimpleNamespace(id="test-model", owned_by="ollama"),
        ]),
    )))
    assert brain._fetch_model_metadata() == {"owned_by": "ollama"}


def test_fetch_model_metadata_falls_back_to_raw_models_list(monkeypatch):
    """Servers answering with a bare JSON array (no list envelope) are
    tolerated - the OpenAI SDK itself fails to parse them (e.g. Together.ai)."""
    def broken(*args, **kwargs):
        raise AttributeError("'list' object has no attribute '_set_private_attributes'")

    import httpx

    def fake_get(url, headers=None, timeout=None):
        assert url == "http://localhost:11434/v1/models"
        assert headers["Authorization"] == "Bearer test-key"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [  # bare array, no {"object": "list"} envelope
                {"id": "other", "owned_by": "lib"},
                {"id": "test-model", "owned_by": "together",
                 "context_length": 262144},
            ],
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    brain = _bare_info_brain(_client=SimpleNamespace(
        models=SimpleNamespace(retrieve=broken, list=broken), api_key="test-key"))
    assert brain._fetch_model_metadata() == {
        "owned_by": "together", "context_length": 262144,
    }


def test_fetch_model_metadata_degrades_gracefully(monkeypatch):
    """Every endpoint variant failing yields {} and stores no cache."""
    def broken(*args, **kwargs):
        raise RuntimeError("server down")

    import httpx
    monkeypatch.setattr(httpx, "get", broken)
    brain = _bare_info_brain(
        _client=SimpleNamespace(models=SimpleNamespace(retrieve=broken, list=broken)))
    assert brain._fetch_model_metadata() == {}
    assert brain._fetch_model_metadata() == {}  # second call retries
    assert not hasattr(brain, "_model_meta")


def test_fetch_model_metadata_caches_successful_result():
    calls = {"count": 0}

    def retrieve(model):
        calls["count"] += 1
        return SimpleNamespace(id="test-model", owned_by="ollama")

    brain = _bare_info_brain(
        _client=SimpleNamespace(models=SimpleNamespace(retrieve=retrieve)))
    first = brain._fetch_model_metadata()
    assert brain._fetch_model_metadata() is first
    assert calls["count"] == 1


def test_model_info_renders_server_metadata(monkeypatch):
    brain = _bare_info_brain()
    monkeypatch.setattr(brain, "_fetch_model_metadata", lambda: dict(
        owned_by="ollama", created=1700000000,
        max_context_length=131072, loaded_context_length=4096,
        capabilities=["completion", "vision", "tools"],
        quantization="Q4_K_M",
    ))
    info = brain.model_info()

    assert "model=test-model" in info
    assert "provider=ollama" in info
    assert "created=2023-11-14" in info
    assert "context_window=4096 tokens (configured)" in info
    assert "server_context=loaded_context_length=4096, max_context_length=131072" in info
    assert "capabilities=completion, vision, tools" in info
    assert "quantization=Q4_K_M" in info  # extra server fields pass through
    assert "vision=no" in info
    assert "session_messages=2" in info
    assert "warning" not in info  # configured window fits the server max


def test_model_info_warns_when_configured_window_exceeds_server_max(monkeypatch):
    brain = _bare_info_brain(_context_window=262144)
    monkeypatch.setattr(brain, "_fetch_model_metadata", lambda: dict(
        max_context_length=131072, loaded_context_length=4096))
    assert "warning: configured context_window exceeds" in brain.model_info()


def test_model_info_loaded_context_alone_never_warns(monkeypatch):
    """loaded_context_length is the current allocation, not a maximum."""
    brain = _bare_info_brain(_context_window=262144)
    monkeypatch.setattr(brain, "_fetch_model_metadata", lambda: dict(
        loaded_context_length=4096))
    assert "warning" not in brain.model_info()


def test_model_info_falls_back_to_local_config(monkeypatch):
    """With an unreachable server the locally configured values still render."""
    brain = _bare_info_brain()
    monkeypatch.setattr(brain, "_fetch_model_metadata", lambda: {})
    info = brain.model_info()

    assert "model=test-model" in info
    assert "base_url=http://localhost:11434/v1" in info
    assert "context_window=4096 tokens (configured)" in info
    assert "temperature=0.6" in info
    assert "max_response_tokens=409" in info
    assert "vision=no" in info
    assert "session_messages=2" in info


def test_resolve_vision_from_server_capabilities(monkeypatch):
    """A server-reported capabilities list is authoritative."""
    brain = _bare_info_brain()
    monkeypatch.setattr(
        brain, "_fetch_model_metadata",
        lambda use_raw_fallback=True: {"capabilities": ["completion", "vision", "tools"]})
    config = SimpleNamespace(brain=SimpleNamespace(vision=False))
    assert brain._resolve_vision_flag(config) is True
    assert brain._vision_source == "server"

    no_vision = _bare_info_brain()
    monkeypatch.setattr(
        no_vision, "_fetch_model_metadata",
        lambda use_raw_fallback=True: {"capabilities": ["completion", "tools"]})
    assert no_vision._resolve_vision_flag(config) is False
    assert no_vision._vision_source == "server"


def test_resolve_vision_falls_back_to_config(monkeypatch):
    """Servers without capability info fall back to the optional config flag."""
    brain = _bare_info_brain()
    monkeypatch.setattr(
        brain, "_fetch_model_metadata", lambda use_raw_fallback=True: {})
    # default: no vision
    assert brain._resolve_vision_flag(SimpleNamespace(brain=SimpleNamespace())) is False
    assert brain._vision_source == "config"
    # ... and the config value is honored as an override
    assert brain._resolve_vision_flag(
        SimpleNamespace(brain=SimpleNamespace(vision=True))) is True
    assert brain._vision_source == "config"


def test_model_info_reports_vision_source(monkeypatch):
    brain = _bare_info_brain(_vision=True, _vision_source="server")
    monkeypatch.setattr(brain, "_fetch_model_metadata", lambda use_raw_fallback=True: {})
    info = brain.model_info()
    assert "vision=yes" in info
    assert "vision_source=server" in info


def test_resolve_context_window_prefers_explicit_config(monkeypatch):
    brain = _bare_info_brain()

    def fail(*args, **kwargs):
        raise AssertionError("must not query the server when config sets the window")

    monkeypatch.setattr(brain, "_fetch_model_metadata", fail)
    assert brain._resolve_context_window(
        SimpleNamespace(brain=SimpleNamespace(context_window=4096))) == 4096
    assert brain._context_window_source == "config"


def test_resolve_context_window_from_server_metadata(monkeypatch):
    brain = _bare_info_brain()
    monkeypatch.setattr(
        brain, "_fetch_model_metadata",
        lambda use_raw_fallback=True: {"max_context_length": 131072})
    assert brain._resolve_context_window(SimpleNamespace(brain=SimpleNamespace())) == 131072
    assert brain._context_window_source == "server"

    together = _bare_info_brain()
    monkeypatch.setattr(
        together, "_fetch_model_metadata",
        lambda use_raw_fallback=True: {"context_length": 262144})
    assert together._resolve_context_window(SimpleNamespace(brain=SimpleNamespace())) == 262144


def test_resolve_context_window_ignores_loaded_allocation(monkeypatch):
    """loaded_context_length is the current allocation, not the capacity."""
    brain = _bare_info_brain()
    monkeypatch.setattr(
        brain, "_fetch_model_metadata",
        lambda use_raw_fallback=True: {"loaded_context_length": 4096})
    assert brain._resolve_context_window(
        SimpleNamespace(brain=SimpleNamespace())) == live_ai_mod.Brain.DEFAULT_CONTEXT_WINDOW
    assert brain._context_window_source == "default"


def test_brain_max_tokens_uses_server_context_window(monkeypatch):
    """End-to-end: the max_response_tokens ratio applies to the resolved window."""
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    import sys

    class FakeModels:
        def retrieve(self, model):
            raise RuntimeError("not supported")

        def list(self):
            return SimpleNamespace(data=[
                SimpleNamespace(id="test-model", max_context_length=131072)])

    stub = sys.modules["openai"]

    class Client(stub.OpenAI):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.models = FakeModels()

    monkeypatch.setattr(stub, "OpenAI", Client)
    agent = live_ai_mod.Brain(_remote_config())  # no context_window in config
    assert agent._context_window == 131072
    assert agent._context_window_source == "server"
    assert agent._max_tokens == int(131072 * 0.2)
