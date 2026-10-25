import io
import pathlib
from logging import getLogger
from types import SimpleNamespace

import pytest

import src.live_ai as live_ai_mod  # noqa: E402


def test_agent_build_tools_includes_summarize_doc():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "summarize_doc" in names


def test_agent_build_tools_includes_query_document():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "query_document" in names


def test_agent_build_tools_includes_image_search():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "image_search" in names


def test_agent_build_tools_includes_download_file():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "download_file" in names


def test_agent_build_tools_includes_extract_archive():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "extract_archive" in names


def test_agent_build_tools_includes_zip_files():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "zip_files" in names


def test_agent_build_tools_registers_web_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    for name in ("web_search", "web_images", "web_videos", "web_news",
                 "web_books", "extract_webpage", "download_file"):
        assert name in names
    for removed in ("webpage_content", "webpage_headlines"):
        assert removed not in names


def test_agent_build_tools_keeps_existing_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    for name in ("system_info", "read_text_file", "web_search", "image_response"):
        assert name in names


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


def test_cancelled_round_does_not_speak(monkeypatch):
    spoken = []

    class FakeSpeech:
        def speak(self, text):
            spoken.append(text)

    class FakeStream:
        def flush(self):
            pass

        def mark_line_clean(self):
            pass

        def note(self, text):
            pass

    import src.tools.speech as speech_mod
    monkeypatch.setattr(speech_mod, "_speech_generator", FakeSpeech())

    agent = _BareAgent()
    agent._messages = []
    agent._context_window = 4096
    agent._stream = FakeStream()
    agent._fragments_buffer = ["partial"]

    agent._cancel_requested = True
    agent._handle_round_end(0)
    assert spoken == []

    agent._cancel_requested = False
    agent._fragments_buffer = ["full"]
    agent._handle_round_end(1)
    assert spoken == ["full"]


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
        memory_path="h", temperature=0.6, max_response_tokens=0.2,
        base_url="https://api.test/v1", api_key_env="TEST_API_KEY",
    )
    brain.update(brain_overrides)
    # Thinking display settings live in [ui] (moved from [brain]).
    ui = dict(
        show_thinking=brain_overrides.get("show_thinking", False),
        max_thinking_tokens=brain_overrides.get("max_thinking_tokens", 0),
    )
    for key in ("show_thinking", "max_thinking_tokens"):
        brain.pop(key, None)
    return SimpleNamespace(brain=SimpleNamespace(**brain),
                           ui=SimpleNamespace(**ui))


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


def test_fetch_model_metadata_falls_back_to_models_list(monkeypatch):
    """Servers without retrieve-by-id (e.g. Ollama) answer via GET /models."""
    def broken_retrieve(model):
        raise RuntimeError("not supported")

    import httpx
    monkeypatch.setattr(httpx, "post", broken_retrieve)  # native probes stay offline
    monkeypatch.setattr(httpx, "get", broken_retrieve)

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
    monkeypatch.setattr(httpx, "post", broken)
    monkeypatch.setattr(httpx, "get", broken)
    brain = _bare_info_brain(
        _client=SimpleNamespace(models=SimpleNamespace(retrieve=broken, list=broken)))
    assert brain._fetch_model_metadata() == {}
    assert brain._fetch_model_metadata() == {}  # second call retries
    assert not hasattr(brain, "_model_meta")


def test_fetch_model_metadata_caches_successful_result(monkeypatch):
    calls = {"count": 0}

    def retrieve(model):
        calls["count"] += 1
        return SimpleNamespace(id="test-model", owned_by="ollama")

    def broken(*args, **kwargs):
        raise RuntimeError("no native api")

    import httpx
    monkeypatch.setattr(httpx, "post", broken)  # no capabilities/context -> native fires
    monkeypatch.setattr(httpx, "get", broken)
    brain = _bare_info_brain(
        _client=SimpleNamespace(models=SimpleNamespace(retrieve=retrieve)))
    first = brain._fetch_model_metadata()
    assert brain._fetch_model_metadata() is first
    assert calls["count"] == 1


def test_fetch_model_metadata_merges_ollama_native_metadata(monkeypatch):
    """Ollama's OpenAI /models is thin; capabilities/context come from /api/show
    (+ the loaded allocation from /api/ps)."""
    def broken(*args, **kwargs):
        raise RuntimeError("not supported")

    import httpx

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == "http://localhost:11434/api/show"
        assert json == {"model": "test-model"}
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "capabilities": ["completion", "vision", "tools", "thinking"],
                "model_info": {"general.architecture": "qwen35",
                               "qwen35.context_length": 262144},
            },
        )

    def fake_ps(url, timeout=None):
        assert url == "http://localhost:11434/api/ps"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": [
                {"model": "other:1b", "context_length": 8192},
                {"name": "test-model", "context_length": 4096},
            ]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_ps)
    brain = _bare_info_brain(_client=SimpleNamespace(models=SimpleNamespace(
        retrieve=broken, list=lambda: SimpleNamespace(data=[
            SimpleNamespace(id="test-model", owned_by="ollama")]))))
    assert brain._fetch_model_metadata() == {
        "owned_by": "ollama",
        "capabilities": ["completion", "vision", "tools", "thinking"],
        "max_context_length": 262144,
        "loaded_context_length": 4096,
    }


def test_fetch_model_metadata_merges_lmstudio_native_metadata(monkeypatch):
    """LM Studio's native /api/v0/models adds capabilities (type vlm -> vision)
    and the context lengths its OpenAI endpoint hides."""
    def broken(*args, **kwargs):
        raise RuntimeError("no openai metadata")

    import httpx

    def fake_post(url, json=None, headers=None, timeout=None):
        raise RuntimeError("/api/show: no such route")  # not an Ollama server

    def fake_get(url, timeout=None):
        assert url == "http://localhost:1234/api/v0/models"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [
                {"id": "other-model", "type": "llm"},
                {"id": "test-model", "object": "model", "type": "vlm",
                 "publisher": "qwen", "arch": "qwen3vl", "quantization": "Q4_K_M",
                 "state": "loaded", "max_context_length": 262144,
                 "loaded_context_length": 8192, "capabilities": ["tool_use"]},
            ]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    brain = _bare_info_brain(
        _base_url="http://localhost:1234/v1",
        _client=SimpleNamespace(models=SimpleNamespace(retrieve=broken, list=broken)))
    assert brain._fetch_model_metadata() == {
        "type": "vlm", "publisher": "qwen", "arch": "qwen3vl",
        "quantization": "Q4_K_M", "state": "loaded",
        "max_context_length": 262144, "loaded_context_length": 8192,
        "capabilities": ["tool_use", "vision"],
    }


def test_fetch_model_metadata_skips_native_probes_for_cloud_hosts(monkeypatch):
    """Only loopback servers are probed with their native APIs."""
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("native probe attempted")

    import httpx
    monkeypatch.setattr(httpx, "post", must_not_be_called)
    monkeypatch.setattr(httpx, "get", must_not_be_called)
    brain = _bare_info_brain(
        _base_url="https://api.together.xyz/v1",
        _client=SimpleNamespace(models=SimpleNamespace(
            retrieve=lambda m: SimpleNamespace(owned_by="together"))))
    assert brain._fetch_model_metadata() == {"owned_by": "together"}


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


# ------------------------------------------- runtime model switching (/model)

def _write_models_file(tmp_path, data, name="models.json"):
    import json
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _switch_brain(monkeypatch, tmp_path, models_data, **brain_overrides):
    """A Brain (offline stub client) with a models JSON file configured."""
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "key-a")
    models_file = _write_models_file(tmp_path, models_data)
    brain_overrides.setdefault("models_file", str(models_file))
    return live_ai_mod.Brain(_remote_config(**brain_overrides))


def test_load_model_entries_accepts_mapping_list_and_wrapper(tmp_path):
    """All three JSON shapes: name->entry mapping, list, and a "models" wrapper."""
    mapping = _write_models_file(tmp_path, {"a": {"model": "m-a"}}, name="mapping.json")
    listing = _write_models_file(tmp_path, [
        {"name": "b", "model": "m-b"}, "m-c"], name="listing.json")
    wrapped = _write_models_file(tmp_path, {"models": {"d": {"model": "m-d"}}},
                                 name="wrapped.json")

    brain = _bare_info_brain()
    brain.config = SimpleNamespace(brain=SimpleNamespace(models_file=str(mapping)))
    assert brain.list_models() == {"a": {"model": "m-a"}}

    brain.config = SimpleNamespace(brain=SimpleNamespace(models_file=str(listing)))
    entries = brain.list_models()
    assert entries["b"] == {"name": "b", "model": "m-b"}
    assert entries["m-c"] == {"model": "m-c"}  # bare id selectable under its name

    brain.config = SimpleNamespace(brain=SimpleNamespace(models_file=str(wrapped)))
    assert brain.list_models() == {"d": {"model": "m-d"}}


def test_load_model_entries_skips_unusable_rows(tmp_path):
    """Entries without a name or a model id are dropped, not fatal."""
    path = _write_models_file(tmp_path, {
        "ok": {"model": "m"},
        "no-model": {"api_key_env": "X"},
        "": {"model": "m"},
        "shorthand": "just-a-model-id",
    })
    brain = _bare_info_brain()
    brain.config = SimpleNamespace(brain=SimpleNamespace(models_file=str(path)))
    assert brain.list_models() == {
        "ok": {"model": "m"},
        "shorthand": {"model": "just-a-model-id"},
    }


def test_select_model_requires_configured_models_file():
    brain = _bare_info_brain()
    brain.config = SimpleNamespace(brain=SimpleNamespace())
    with pytest.raises(ValueError, match="models_file"):
        brain.select_model("anything")


def test_select_model_missing_file_raises(tmp_path):
    brain = _bare_info_brain()
    brain.config = SimpleNamespace(
        brain=SimpleNamespace(models_file=str(tmp_path / "nope.json")))
    with pytest.raises(FileNotFoundError):
        brain.select_model("anything")


def test_select_model_invalid_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    brain = _bare_info_brain()
    brain.config = SimpleNamespace(brain=SimpleNamespace(models_file=str(path)))
    with pytest.raises(ValueError, match="Invalid JSON"):
        brain.select_model("anything")


def test_select_model_unknown_name_lists_available(monkeypatch, tmp_path):
    brain = _switch_brain(monkeypatch, tmp_path, {"a": {"model": "m-a"}})
    with pytest.raises(KeyError, match="Available: a"):
        brain.select_model("nope")
    assert brain._model_name == "m-a"  # current model untouched


def test_select_model_switches_client_endpoint_and_key(monkeypatch, tmp_path):
    brain = _switch_brain(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},  # first entry: boots the session
        "other": {"api_key_env": "OTHER_API_KEY",
                  "base_url": "https://api.other/v1", "model": "other-model"},
    })
    monkeypatch.setenv("OTHER_API_KEY", "key-other")
    assert brain._model_name == "boot-model"  # sanity: first entry booted

    message = brain.select_model("other")

    assert brain._model_name == "other-model"
    assert brain._base_url == "https://api.other/v1"
    assert brain._client.api_key == "key-other"
    assert brain._model_entry == "other"
    assert "other" in message and "other-model" in message


def test_select_model_defaults_to_brain_connection_fields(monkeypatch, tmp_path):
    """Entry without api_key_env/base_url keeps the [brain] connection values."""
    brain = _switch_brain(monkeypatch, tmp_path, {"plain": {"model": "plain-model"}})

    brain.select_model("plain")

    assert brain._model_name == "plain-model"
    assert brain._base_url == "https://api.test/v1"
    assert brain._client.api_key == "key-a"  # from TEST_API_KEY


def test_select_model_missing_api_key_raises_before_switch(monkeypatch, tmp_path):
    brain = _switch_brain(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "locked": {"api_key_env": "MISSING_API_KEY",
                   "base_url": "https://api.x/v1", "model": "m"}})
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    monkeypatch.setattr(live_ai_mod, "_load_env_file", lambda: None)

    with pytest.raises(ValueError, match="MISSING_API_KEY"):
        brain.select_model("locked")
    assert brain._model_name == "boot-model"  # the old model stays active


def test_select_model_entry_without_any_base_url_raises(tmp_path):
    """Neither the entry nor [brain] carries a base_url -> refuse to switch."""
    path = _write_models_file(tmp_path, {"orphan": {"model": "m"}})
    brain = _bare_info_brain()
    brain.config = SimpleNamespace(brain=SimpleNamespace(models_file=str(path)))
    with pytest.raises(ValueError, match="base_url"):
        brain.select_model("orphan")


def test_select_model_resets_metadata_and_recomputes_caps(monkeypatch, tmp_path):
    """Cached metadata is dropped; context window / vision / cap re-resolve."""
    brain = _switch_brain(monkeypatch, tmp_path, {"big": {"model": "big-model"}},
                          context_window=1000, max_response_tokens=0.2)
    assert brain._model_name == "big-model"  # first entry booted the session
    brain._model_meta = {"owned_by": "stale"}  # cached by a previous turn

    brain.select_model("big")

    assert not hasattr(brain, "_model_meta")  # cache dropped for the new server
    assert brain._context_window == 1000
    assert brain._max_tokens == 200  # ratio re-applied to the (new) window
    assert brain._vision is False


# ------------------------------------------------- boot from the models file

def test_boot_uses_first_models_file_entry(monkeypatch, tmp_path):
    """With models_file set, its first entry boots: no [brain] duplicates."""
    brain = _switch_brain(monkeypatch, tmp_path, {
        "first": {"api_key_env": "TEST_API_KEY",
                  "base_url": "https://api.first/v1", "model": "first-model"},
        "second": {"api_key_env": "TEST_API_KEY",
                   "base_url": "https://api.second/v1", "model": "second-model"},
    })
    assert brain._model_name == "first-model"
    assert brain._base_url == "https://api.first/v1"
    assert brain._client.api_key == "key-a"
    assert brain._model_entry == "first"


def test_boot_first_entry_falls_back_to_brain_connection(monkeypatch, tmp_path):
    """A first entry without base_url/api_key_env inherits them from [brain]."""
    brain = _switch_brain(monkeypatch, tmp_path, {"only": {"model": "only-model"}})
    assert brain._model_name == "only-model"
    assert brain._base_url == "https://api.test/v1"
    assert brain._client.api_key == "key-a"


def test_boot_first_entry_missing_key_raises(monkeypatch, tmp_path):
    """A non-resolvable key of the first entry fails startup (before .env)."""
    path = _write_models_file(tmp_path, {
        "locked": {"api_key_env": "MISSING_BOOT_KEY",
                   "base_url": "https://api.x/v1", "model": "m"}})
    monkeypatch.delenv("MISSING_BOOT_KEY", raising=False)
    monkeypatch.setattr(live_ai_mod, "_load_env_file", lambda: None)
    with pytest.raises(ValueError, match="MISSING_BOOT_KEY"):
        live_ai_mod.Brain(_remote_config(models_file=str(path)))


def test_boot_without_models_file_keeps_brain_keys(monkeypatch):
    """Legacy setup: no models_file -> [brain] model/base_url/api_key_env boot."""
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "key-a")
    agent = live_ai_mod.Brain(_remote_config())
    assert agent._model_name == "test-model"
    assert agent._base_url == "https://api.test/v1"
    assert agent._client.api_key == "key-a"
    assert not hasattr(agent, "_model_entry")


def test_boot_without_models_file_requires_model_and_base_url(monkeypatch):
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    with pytest.raises(ValueError, match="base_url"):
        live_ai_mod.Brain(_remote_config(base_url=""))
    with pytest.raises(ValueError, match="model"):
        live_ai_mod.Brain(_remote_config(model=""))


## Chain-of-thought visualization ([ui] show_thinking / max_thinking_tokens) ##

class _ThinkingRecorder:
    """StreamPrinter stand-in that records thinking vs. answer text."""

    def __init__(self):
        self.thinking: list[str] = []
        self.answer: list[str] = []

    def feed_thinking(self, text):
        self.thinking.append(text)

    def feed(self, text):
        self.answer.append(text)

    def flush(self):
        pass

    def note(self, text):
        pass

    def start_spinner(self, note="working"):
        pass

    def stop_spinner(self):
        pass

    def mark_line_clean(self):
        pass


def _delta(delta_content=None, reasoning=None, reasoning_content=None):
    """One OpenAI-style stream chunk with the given delta fields."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=delta_content, reasoning=reasoning,
        reasoning_content=reasoning_content, tool_calls=None))])


def _thinking_agent(monkeypatch, chunks, **brain_overrides):
    """Brain wired to a scripted completion stream (offline, via the stub)."""
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    agent = live_ai_mod.Brain(_remote_config(**brain_overrides))
    agent._stream = _ThinkingRecorder()
    client = agent._client
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: iter(chunks)))
    return agent


def _fragments(agent):
    return [f.content for f in agent.model_fragments("hi")]


def test_reasoning_content_streams_to_thinking_display(monkeypatch):
    """DeepSeek-style reasoning_content deltas go to the display, not the answer."""
    agent = _thinking_agent(monkeypatch, [
        _delta(reasoning_content="let me "),
        _delta(reasoning_content="think"),
        _delta(delta_content="The answer"),
        _delta(delta_content=" is 4"),
    ], show_thinking=True)
    assert _fragments(agent) == ["The answer", " is 4"]
    assert "".join(agent._stream.thinking) == "\n💭 let me think"
    # Reasoning never enters the chat history.
    assert agent._messages[-1] == {"role": "assistant", "content": "The answer is 4"}


def test_reasoning_field_openrouter_style(monkeypatch):
    agent = _thinking_agent(monkeypatch, [
        _delta(reasoning="pondering"),
        _delta(delta_content="ok"),
    ], show_thinking=True)
    assert _fragments(agent) == ["ok"]
    assert "".join(agent._stream.thinking) == "\n💭 pondering"


def test_think_tags_split_across_chunks(monkeypatch):
    """Raw istine-tagged content: tags may split at any chunk boundary."""
    agent = _thinking_agent(monkeypatch, [
        _delta(delta_content="<thi"),
        _delta(delta_content="nk>deep "),
        _delta(delta_content="thou"),
        _delta(delta_content="ght</thi"),
        _delta(delta_content="nk>final"),
    ], show_thinking=True)
    assert _fragments(agent) == ["final"]
    assert "".join(agent._stream.thinking) == "\n💭 deep thought"
    assert agent._messages[-1] == {"role": "assistant", "content": "final"}


def test_show_thinking_off_keeps_tags_in_answer(monkeypatch):
    """Feature off: content passes through unchanged (historical behavior)."""
    agent = _thinking_agent(monkeypatch, [
        _delta(delta_content="istinehiine"),
    ])
    assert _fragments(agent) == ["istinehiine"]
    assert agent._stream.thinking == []


def test_max_thinking_tokens_caps_display(monkeypatch):
    """The cap bounds what is visualized; the marker is shown exactly once."""
    agent = _thinking_agent(monkeypatch, [
        _delta(reasoning_content="x" * 1000),
        _delta(delta_content="answer"),
    ], show_thinking=True, max_thinking_tokens=10)  # 10 tokens ~ 40 chars
    assert _fragments(agent) == ["answer"]
    shown = "".join(agent._stream.thinking)
    assert shown.count("… (thinking truncated)") == 1
    assert "x" * 1000 not in shown


def test_thinking_token_cap_parsing():
    cap = live_ai_mod._thinking_token_cap
    assert cap(None) == 0      # unset -> unlimited
    assert cap(0) == 0
    assert cap(-5) == 0        # negative -> unlimited
    assert cap(25) == 25
    assert cap("40") == 40
    assert cap("not-a-number") == 0  # bad value must not fail the session


def test_thinking_state_resets_between_turns(monkeypatch):
    """A second turn restarts the counter, tag state and header."""
    agent = _thinking_agent(monkeypatch, [
        _delta(reasoning_content="r1"), _delta(delta_content="a1"),
    ], show_thinking=True)
    assert _fragments(agent) == ["a1"]
    first_turn_thinking = len(agent._stream.thinking)
    assert agent._thinking_chars > 0
    agent._client.chat.completions.create = lambda **kw: iter([
        _delta(reasoning_content="r2"), _delta(delta_content="a2"),
    ])
    assert _fragments(agent) == ["a2"]
    assert "".join(agent._stream.thinking[first_turn_thinking:]) == "\n💭 r2"
    assert agent._thinking_chars == len("r2")


def test_null_printer_drops_thinking():
    """Quiet (pooled) agents never display reasoning."""
    printer = live_ai_mod.NullPrinter()
    printer.feed_thinking("secret thoughts")
    assert printer.buffer == []


def test_stream_printer_feed_thinking_prints_dim():
    """feed_thinking prints without touching the answer-stream line state."""
    from src.common import StreamPrinter
    printer = StreamPrinter()
    sink = io.StringIO()
    printer.console.file = sink
    printer.feed_thinking("thinking hard")
    assert "thinking hard" in sink.getvalue()
    # The answer state machinery is untouched: no fence pending, line empty.
    assert printer._in_code is False
    assert printer._line == ""


def _tool_call_delta(name="model_info", arguments="{}"):
    """One OpenAI-style stream chunk carrying a single tool call."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, reasoning=None, reasoning_content=None,
        tool_calls=[SimpleNamespace(index=0, id="c1", type="function",
                                    function=SimpleNamespace(name=name,
                                                             arguments=arguments))]))])


def _thinking_act_agent(monkeypatch, rounds, **brain_overrides):
    """Agent (act loop) wired to scripted per-round completion streams."""
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    agent = live_ai_mod.Agent(_remote_config(**brain_overrides))
    agent._stream = _ThinkingRecorder()
    scripted = iter(rounds)
    agent._client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: iter(next(scripted))))
    return agent


def test_truncation_marker_printed_once(monkeypatch):
    """After the cap, further reasoning chunks must be dropped silently."""
    agent = _thinking_agent(monkeypatch, [
        _delta(reasoning_content="a" * 100),
        _delta(reasoning_content="b" * 100),
        _delta(reasoning_content="c" * 100),
        _delta(delta_content="answer"),
    ], show_thinking=True, max_thinking_tokens=10)  # ~40 chars
    assert _fragments(agent) == ["answer"]
    shown = "".join(agent._stream.thinking)
    assert shown.count("… (thinking truncated)") == 1
    assert shown.count("💭") == 1


def test_thinking_budget_resets_per_round(monkeypatch):
    """Each act() round gets a fresh visualization budget (regression).

    The per-turn budget made round 1 exhaust the cap so all later rounds of
    a long tool-loop turn printed only truncation markers.
    """
    agent = _thinking_act_agent(monkeypatch, [
        # Round 1: blows the cap, then requests a tool call.
        [_delta(reasoning_content="x" * 1000), _tool_call_delta()],
        # Round 2: fresh budget -> this reasoning must be visible again.
        [_delta(reasoning_content="round two thinking"),
         _delta(delta_content="final answer")],
    ], show_thinking=True, max_thinking_tokens=10)
    agent.act("hi")
    shown = "".join(agent._stream.thinking)
    assert shown.count("… (thinking truncated)") == 1  # once for round 1
    assert "round two thinking" in shown               # round 2 budget fresh
    assert shown.count("💭") == 2                      # header once per round
    assert agent._messages[-1] == {"role": "assistant", "content": "final answer"}


# ------------------------------------------------ MCP tool injection (PLAN 5)


def _mcp_manager(monkeypatch, results=None):
    """Bind a fake MCP manager to src.tools.mcp.get_manager for an Agent."""
    import src.tools.mcp as mcp_mod

    class FakeMcpManager:
        status_value = {
            "test": {"transport": "http", "status": "connected",
                     "tools": ["echo"]},
        }
        tools = [{"name": "echo", "description": "Echo a payload",
                  "inputSchema": {"type": "object",
                                  "properties": {"text": {"type": "string"}},
                                  "required": ["text"]}}]
        calls: list[tuple[str, str, dict]] = []
        refreshed = 0

        def connect_all(self):
            return {"test": ["echo"]}

        def refresh(self):
            FakeMcpManager.refreshed += 1
            return {"test": ["echo"]}

        def status(self):
            return self.status_value

        def list_tools(self, server):
            return list(self.tools)

        def call(self, server, tool, arguments):
            FakeMcpManager.calls.append((server, tool, arguments))
            return (results or {}).get((server, tool), f"ran {tool}")

    manager = FakeMcpManager()
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    return manager


def test_agent_init_appends_mcp_tools(monkeypatch):
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    _mcp_manager(monkeypatch)
    agent = live_ai_mod.Agent(_remote_config())
    names = [tool.__name__ for tool in agent.tools]
    assert "mcp_test_echo" in names
    assert agent._mcp_tool_names == ["mcp_test_echo"]
    assert "mcp_test_echo" in agent._tool_impls
    definition = next(d for d in agent.tool_definitions
                      if d["function"]["name"] == "mcp_test_echo")
    assert definition["function"]["description"] == "Echo a payload"
    assert definition["function"]["parameters"]["required"] == ["text"]


def test_agent_without_mcp_manager_unchanged(monkeypatch):
    """G3 regression guard: no manager -> identical tool set as before."""
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    agent = live_ai_mod.Agent(_remote_config())
    names = [tool.__name__ for tool in agent.tools]
    assert "mcp_test_echo" not in names
    assert agent._mcp_tool_names == []
    # The two always-present bridge tools are the only MCP-related additions.
    assert "mcp_list_tools" in names and "mcp_call_tool" in names


def test_act_loop_calls_mcp_tool(monkeypatch):
    """Round 1 requests the MCP tool, round 2 answers from its result."""
    manager = _mcp_manager(monkeypatch, results={("test", "echo"): "pong"})
    agent = _thinking_act_agent(monkeypatch, [
        [_tool_call_delta(name="mcp_test_echo",
                          arguments='{"text": "ping"}')],
        [_delta(delta_content="the echo said pong")],
    ])
    agent.act("use the echo tool")
    assert manager.calls == [("test", "echo", {"text": "ping"})]
    assert agent._messages[3] == {"role": "tool", "tool_call_id": "c1",
                                  "content": "pong"}
    assert agent._messages[-1] == {"role": "assistant",
                                   "content": "the echo said pong"}


def test_refresh_mcp_tools_replaces_entries(monkeypatch):
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    manager = _mcp_manager(monkeypatch)
    agent = live_ai_mod.Agent(_remote_config())
    before_tools = list(agent.tools)
    before_definitions = list(agent.tool_definitions)

    message = agent.refresh_mcp_tools()

    assert "refreshed" in message and "mcp_test_echo" in message
    assert manager.refreshed == 1
    # The MCP tool is a fresh wrapper object at the same name.
    names = [tool.__name__ for tool in agent.tools]
    assert names.count("mcp_test_echo") == 1
    assert len(agent.tools) == len(before_tools)
    assert len(agent.tool_definitions) == len(before_definitions)
    assert agent._tool_impls["mcp_test_echo"].__name__ == "mcp_test_echo"


def test_refresh_mcp_tools_without_manager(monkeypatch):
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    agent = live_ai_mod.Agent(_remote_config())
    message = agent.refresh_mcp_tools()
    assert "not configured" in message


# ------------------------------------------------------- tool topic grouping


def test_tool_topics_groups_known_tools_in_curated_order():
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    tools = [
        live_ai_mod.web_search, live_ai_mod.recall_memory,
        live_ai_mod.agent_activity, brain.model_info,
    ]
    grouped = live_ai_mod.tool_topics(tools)
    topics = [topic for topic, _ in grouped]
    # curated order wins regardless of the input order
    assert topics == ["model", "monitoring", "memory", "web"]


def test_tool_topics_preserves_input_order_within_topic():
    tools = [live_ai_mod.web_search, live_ai_mod.extract_webpage,
             live_ai_mod.web_news]
    grouped = dict(live_ai_mod.tool_topics(tools))
    assert [t.__name__ for t in grouped["web"]] == [
        "web_search", "extract_webpage", "web_news"]


def test_tool_topics_unknown_tools_land_in_other_last():
    def mystery_tool():
        """Unknown."""

    tools = [mystery_tool, live_ai_mod.recall_memory]
    grouped = live_ai_mod.tool_topics(tools)
    assert [topic for topic, _ in grouped] == ["memory", "other"]
    assert grouped[-1][1] == [mystery_tool]


def test_tool_topics_flattened_is_deterministic():
    """Grouping reorders by topic, but the flattened order is deterministic
    (same input -> same display order), which is what /tools numbering uses."""
    tools = [
        live_ai_mod.web_search, live_ai_mod.recall_memory,
        live_ai_mod.agent_activity, live_ai_mod.exit_gracefully,
        live_ai_mod.write_text_file,
    ]
    first = [tool for _, group in live_ai_mod.tool_topics(tools) for tool in group]
    second = [tool for _, group in live_ai_mod.tool_topics(list(reversed(tools)))
              for tool in group]
    assert first == second  # stable regardless of input order
    assert set(first) == set(tools)  # no tool lost or duplicated


def test_tool_topics_empty_list():
    assert live_ai_mod.tool_topics([]) == []


# --------------------------------------------- JSON schema for bare dict


def test_json_schema_bare_dict_is_object():
    """A bare ``dict`` annotation maps to {"type": "object"}.

    Regression: get_origin(dict) is None, so a bare dict fell through to
    {"type": "string"} and models were told that e.g. schedule_task's
    task objects were plain strings.
    """
    assert live_ai_mod.OpenAICore._json_schema(dict) == {"type": "object"}
    assert live_ai_mod.OpenAICore._json_schema(dict[str, str]) == {"type": "object"}


def test_json_schema_list_of_dict_items_are_objects():
    assert live_ai_mod.OpenAICore._json_schema(list[dict]) == {
        "type": "array", "items": {"type": "object"}}


def test_schedule_task_schema_offers_task_objects():
    """The model-visible schema must describe task dicts as objects, not strings."""
    definition = live_ai_mod.OpenAICore._openai_tool_definition(
        live_ai_mod.schedule_task)
    assert "tasks" not in definition["function"]["parameters"]["properties"] or \
        definition["function"]["parameters"]["properties"]["tasks"] == {
            "type": "array", "items": {"type": "object"}}
