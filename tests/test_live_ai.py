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
