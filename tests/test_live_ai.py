from logging import getLogger
from types import SimpleNamespace

import pytest

pytest.importorskip("lmstudio", reason="live_ai imports lmstudio at module level")

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

        def summarize(self, text, max_sentences=None):
            seen["calls"].append((text, max_sentences))
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
    assert seen["calls"] == [("long text", None), ("more text", None)]


def test_summarize_text_missing_file_returns_message(tmp_path):
    brain = _bare_summarization_brain()
    out = brain.summarize_text(str(tmp_path / "nope.txt"))
    assert "Unable to read" in out


def test_summarize_text_forwards_max_sentences(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize(self, text, max_sentences=None):
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

        def summarize(self, text, max_sentences=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    brain = _bare_summarization_brain()
    out = brain.summarize_text(str(target))
    assert out.startswith("Error:") and "boom" in out


def test_agent_build_tools_includes_summarize_text():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "summarize_text" in names


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


def test_camel_tool_definition_advertises_and_translates():
    def query_corpus(query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
        """Search a named corpus."""
        return f"{query}|{corpus_name}|{kind}|{top_k}"

    definition = live_ai_mod._camel_tool_definition(query_corpus)
    assert set(definition["parameters"]) == {"query", "corpusName", "kind", "topK"}
    assert definition["parameters"]["corpusName"] is str
    assert definition["parameters"]["topK"] == {"type": int, "default": 0}
    # The model emits camelCase; the implementation maps it to the real names.
    assert definition["implementation"](
        query="q", corpusName="autosar", kind="image", topK=2
    ) == "q|autosar|image|2"


def test_camel_tool_definition_schema_uses_camel_case():
    from lmstudio.json_api import ChatResponseEndpoint

    def query_corpus(query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
        """Search a named corpus."""
        return "ok"

    definition = live_ai_mod._camel_tool_definition(query_corpus)
    tools, client_map = ChatResponseEndpoint.parse_tools([definition])
    schema = tools.to_dict()["tools"][0]["function"]["parameters"]
    assert schema["required"] == ["query", "corpusName"]
    assert "corpusName" in schema["properties"] and "topK" in schema["properties"]
    assert "corpus_name" not in schema["properties"]


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
        self._cancel_requested = False
        self._active_stream = None
        self._fragments_buffer = []
        self._stream = None


def test_cancel_generation_marks_flag_and_cancels_stream():
    class FakeStream:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    agent = _BareAgent()
    stream = FakeStream()
    agent._active_stream = stream

    assert agent.cancel_requested is False
    assert agent.cancel_generation() == "Cancel requested"
    assert agent.cancel_requested is True
    assert stream.cancel_calls == 1
    assert agent._active_stream is None  # stale handle dropped

    # A cancel without an active stream (act() rounds) still flags the turn.
    agent._cancel_requested = False
    agent.cancel_generation()
    assert agent.cancel_requested is True

    # Cancelling a dead stream must never raise.
    class BrokenStream:
        def cancel(self):
            raise RuntimeError("already closed")

    agent._active_stream = BrokenStream()
    assert agent.cancel_generation() == "Cancel requested"
    assert agent._active_stream is None


def test_prediction_stream_tracked_and_pending_cancel_applied():
    """Each act() round's stream is tracked; a pending cancel stops the new round."""
    class FakeStream:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    agent = _BareAgent()
    stream = FakeStream()
    agent._handle_prediction_stream(stream, 0)
    assert agent._active_stream is stream
    assert stream.cancel_calls == 0  # no pending cancel -> stream keeps running

    # Cancel requested while a tool was running (between rounds):
    # the next round's stream is cancelled immediately on registration.
    agent._cancel_requested = True
    stream2 = FakeStream()
    agent._handle_prediction_stream(stream2, 1)
    assert stream2.cancel_calls == 1
    assert agent._active_stream is stream2


def test_cancelled_round_does_not_speak():
    class FakeSpeech:
        def __init__(self):
            self.spoken = []

        def speak(self, text):
            self.spoken.append(text)

    class FakeModel:
        def apply_prompt_template(self, chat):
            return ""

        def tokenize(self, text):
            return []

    class FakeStream:
        def flush(self):
            pass

    agent = _BareAgent()
    agent.speech = FakeSpeech()
    agent._model = FakeModel()
    agent._chat = None
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


# ------------------------------------------- patched SDK: act() stream exposure


def _bare_llm(monkeypatch, fake_stream_factory):
    """LLM handle wired to FakeStreamSpec, bypassing any real connection."""
    from logging import getLogger

    import lmstudio as lms
    import lmstudio.sync_api as sync_api

    class FakeSession:
        def _create_channel(self, endpoint):
            return None  # the fake stream never touches the channel

    llm = sync_api.LLM.__new__(sync_api.LLM)
    llm.identifier = "test-model"
    llm._session = FakeSession()
    llm._logger = getLogger("test")
    monkeypatch.setattr(sync_api, "PredictionStream", fake_stream_factory)
    return lms, llm


class FakeStreamSpec:
    """Stand-in for act()'s per-round PredictionStream (patched SDK contract)."""

    def __init__(self, channel_cm, endpoint):
        self.endpoint = endpoint
        self.started = False
        self.cancel_calls = 0
        self._cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancel_calls += 1
        self._cancelled = True

    def _iter_events(self):
        yield ("fragment",)  # never a tool call event

    def result(self):
        if self._cancelled:
            return _FakePrediction("partial answer", "userStopped")
        return _FakePrediction("full answer", "eosFound")


class _FakePrediction(str):
    """PredictionResult double: str content + stats.stop_reason + .content."""

    def __new__(cls, content, stop_reason):
        obj = super().__new__(cls, content)
        obj.content = content
        obj.stats = SimpleNamespace(stop_reason=stop_reason)
        return obj


def dummy_tool() -> str:
    """Does nothing; must never run during a cancelled round."""
    return "ok"


def test_act_cancelled_round_delivers_partial_and_stops(monkeypatch):
    """on_prediction_stream exposes a started stream; cancel ends the agent loop."""
    lms, llm = _bare_llm(monkeypatch, FakeStreamSpec)
    received, messages, round_ends = [], [], []
    act_result = llm.act(
        lms.Chat("system"),
        [dummy_tool],
        on_message=messages.append,
        on_round_end=round_ends.append,
        on_prediction_stream=lambda stream, index: (
            received.append((stream, index)), stream.cancel()
        ),
    )
    assert act_result.rounds == 1
    (stream, index), = received
    assert index == 0
    assert stream.started  # callback gets an already-started, cancellable stream
    assert stream.cancel_calls == 1
    assert messages[0].content[0].text == "partial answer"
    assert round_ends == [0]  # partial round is still reported via on_round_end


def test_act_normal_round_reports_full_response(monkeypatch):
    lms, llm = _bare_llm(monkeypatch, FakeStreamSpec)
    received, messages, round_ends = [], [], []
    act_result = llm.act(
        lms.Chat("system"),
        [dummy_tool],
        on_message=messages.append,
        on_round_end=round_ends.append,
        on_prediction_stream=lambda stream, index: received.append(stream),
    )
    assert act_result.rounds == 1
    stream, = received
    assert stream.started and stream.cancel_calls == 0
    assert messages[0].content[0].text == "full answer"
    assert round_ends == [0]


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


class FakeHistoryChat:
    """Minimal self._chat double: struct-like messages in a plain list."""

    def __init__(self, messages):
        self._messages = list(messages)

    def append(self, message):
        self._messages.append(
            SimpleNamespace(
                role=message["role"],
                content=[SimpleNamespace(text=message["content"]["text"])],
            )
        )

    def add_user_message(self, text):
        self._messages.append(SimpleNamespace(role="user", content=[SimpleNamespace(text=text)]))


def _msg(role, *texts):
    return SimpleNamespace(role=role, content=[SimpleNamespace(text=t) for t in texts])


def _bare_repair_agent(messages):
    agent = _BareAgent()
    agent._chat = FakeHistoryChat(messages)
    return agent


def test_history_repair_pads_empty_assistant_turn():
    agent = _bare_repair_agent(
        [_msg("user", "hello"), _msg("assistant", "")]  # cancelled, no tokens
    )
    agent._add_user_prompt("next question")
    assert agent._chat._messages[-2].content[0].text == live_ai_mod.CANCELLED_PLACEHOLDER
    assert agent._chat._messages[-1].content[0].text == "next question"


def test_history_repair_drops_unanswered_user_prompts():
    # Trailing unanswered prompts are dropped (all of them: consecutive user
    # messages would fail role alternation anyway).
    agent = _bare_repair_agent(
        [_msg("user", "first"), _msg("user", "never answered")]  # error turns
    )
    agent._add_user_prompt("next question")
    assert [m.content[0].text for m in agent._chat._messages] == ["next question"]


def test_history_repair_cascades_cancelled_then_errored_turn():
    agent = _bare_repair_agent(
        [
            _msg("user", "ask"),
            _msg("assistant", ""),  # cancelled with no tokens
            _msg("user", "never answered"),  # then this turn errored too
        ]
    )
    agent._add_user_prompt("next question")
    assert [m.content[0].text for m in agent._chat._messages] == [
        "ask",
        live_ai_mod.CANCELLED_PLACEHOLDER,
        "next question",
    ]


def test_history_repair_keeps_partial_and_normal_history():
    partial = [_msg("user", "ask"), _msg("assistant", "partial text")]
    agent = _bare_repair_agent(list(partial))
    agent._add_user_prompt("next")
    assert [m.content[0].text for m in agent._chat._messages] == [
        "ask",
        "partial text",
        "next",
    ]
    # a healthy history must be left untouched
    healthy = [_msg("user", "q"), _msg("assistant", "a")]
    agent = _bare_repair_agent(list(healthy))
    agent._add_user_prompt("next")
    assert [m.content[0].text for m in agent._chat._messages] == ["q", "a", "next"]


class EmptyCancelStreamSpec(FakeStreamSpec):
    """Cancelled round that generated nothing (LM Studio stopped pre-token)."""

    def result(self):
        if self._cancelled:
            return _FakePrediction("", "userStopped")
        return _FakePrediction("full answer", "eosFound")


def test_act_cancelled_empty_round_gets_placeholder(monkeypatch):
    """An empty cancelled round must not append an empty assistant turn."""
    lms, llm = _bare_llm(monkeypatch, EmptyCancelStreamSpec)
    messages = []
    llm.act(
        lms.Chat("system"),
        [dummy_tool],
        on_message=messages.append,
        on_prediction_stream=lambda stream, index: stream.cancel(),
    )
    # Empty text would be dropped by the model's prompt template, leaving a
    # tool result directly followed by a user message (role alternation error).
    assert messages[0].content[0].text == live_ai_mod.CANCELLED_PLACEHOLDER
