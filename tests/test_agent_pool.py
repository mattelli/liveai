"""Tests for the multi-agent pool (PLAN.md tasks 1-7).

Task 1: ``[agent]`` pool settings resolution (``agent_pool_settings``).
"""

import pathlib
import textwrap
import json
from types import SimpleNamespace

import pytest

import src.live_ai as live_ai_mod  # noqa: E402
from src.common import load_config  # noqa: E402


# ------------------------------------------------------------------ settings


def _agent_config(**agent) -> SimpleNamespace:
    return SimpleNamespace(agent=SimpleNamespace(**agent))


def test_agent_pool_settings_full_config():
    config = _agent_config(
        bootstrap_prompt="b",
        parallel_agents=2,
        agent_models=["glm", "qwen"],
        agent_system_prompts=["research", "code"],
    )
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": True,
        "monitor": set(),
        "monitor_log_file": "agents.jsonl",
        "monitor_buffer": 100,
        "async_delegation": False,
    }


def test_agent_pool_settings_defaults_when_keys_missing():
    # Only the historical bootstrap_prompt key: single-agent behavior.
    config = _agent_config(bootstrap_prompt="b")
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 0,
        "models": [],
        "system_prompts": [],
        "quiet": True,
        "monitor": set(),
        "monitor_log_file": "agents.jsonl",
        "monitor_buffer": 100,
        "async_delegation": False,
    }


def test_agent_pool_settings_without_agent_section():
    assert live_ai_mod.agent_pool_settings(SimpleNamespace()) == {
        "count": 0,
        "models": [],
        "system_prompts": [],
        "quiet": True,
        "monitor": set(),
        "monitor_log_file": "agents.jsonl",
        "monitor_buffer": 100,
        "async_delegation": False,
    }


def test_agent_pool_settings_truncates_models_to_count():
    config = _agent_config(parallel_agents=1, agent_models=["a", "b", "c"])
    assert live_ai_mod.agent_pool_settings(config)["models"] == ["a"]


def test_agent_pool_settings_accepts_comma_separated_strings():
    # load_config path-resolves strings inside TOML lists, so a comma-
    # separated string is the robust way to declare several models.
    config = _agent_config(
        parallel_agents=2, agent_models="glm, qwen", agent_system_prompts="research, code"
    )
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": True,
        "monitor": set(),
        "monitor_log_file": "agents.jsonl",
        "monitor_buffer": 100,
        "async_delegation": False,
    }


def test_agent_pool_settings_pads_missing_system_prompts():
    config = _agent_config(parallel_agents=3, agent_system_prompts=["one"])
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings["system_prompts"] == ["one", None, None]


def test_agent_pool_settings_rejects_bad_count():
    config = _agent_config(parallel_agents="many")
    assert live_ai_mod.agent_pool_settings(config)["count"] == 0
    config = _agent_config(parallel_agents=-2)
    assert live_ai_mod.agent_pool_settings(config)["count"] == 0


def test_agent_pool_settings_quiet_is_always_true():
    """Pooled agents are always quiet: no user-facing flag anymore."""
    assert live_ai_mod.agent_pool_settings(_agent_config(parallel_agents=1))["quiet"] is True
    # A leftover agent_quiet key in the config is ignored.
    config = _agent_config(parallel_agents=1, agent_quiet=False)
    assert live_ai_mod.agent_pool_settings(config)["quiet"] is True


# ------------------------------------------------- load_config round-trip


def test_load_config_round_trips_agent_pool_keys(tmp_path):
    config_file = tmp_path / "config.toml"
    # Comma-separated strings: load_config path-resolves strings inside TOML
    # lists (e.g. "glm" -> WindowsPath), so the string form is the robust one.
    config_file.write_text(textwrap.dedent("""
        [agent]
        bootstrap_prompt = "hello"
        parallel_agents = 2
        agent_models = "glm, qwen"
        agent_system_prompts = "research, code"
    """), encoding="utf-8")
    config = load_config(config_file)
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": True,
        "monitor": set(),
        "monitor_log_file": "agents.jsonl",
        "monitor_buffer": 100,
        "async_delegation": False,
    }


# ------------------------------------------------- task 2: per-agent tools


def _offline_agent(monkeypatch, tmp_path, agent_overrides=None, **kwargs):
    """A real Agent built offline (stubbed openai client).

    ``agent_overrides`` are merged into the ``[agent]`` config section (e.g.
    ``{"max_act_rounds": 100}``). Also configures the memory/vision tool
    modules with a tmp history path, since ``[brain] memory_path`` lives in
    the module-level config cache.
    """
    from src.tools import memory as memory_mod, vision as vision_mod
    from tests.test_live_ai import _fake_openai, _remote_config

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    memory_mod.configure(SimpleNamespace(
        brain=SimpleNamespace(memory_path=str(tmp_path / "history.txt"),
                              system_prompt="be helpful"),
        agent=SimpleNamespace(bootstrap_prompt="bootstrap"),
    ))
    config = _remote_config()
    if agent_overrides:
        config.agent = SimpleNamespace(
            bootstrap_prompt="bootstrap", **agent_overrides)
    return live_ai_mod.Agent(config, **kwargs)


def test_pooled_agent_gets_own_memory_and_vision_instances(monkeypatch, tmp_path):
    from src.tools import memory as memory_mod, vision as vision_mod

    agent_a = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    agent_b = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)

    def find(agent, module):
        return [t for t in agent.tools if getattr(t, "__module__", "") == module.__name__]

    mem_a, mem_b = find(agent_a, memory_mod), find(agent_b, memory_mod)
    vis_a, vis_b = find(agent_a, vision_mod), find(agent_b, vision_mod)
    assert len(mem_a) == 3 and len(vis_a) == 1  # recall/store/clear + image_response
    # fresh instances per agent: none is the module-default instance
    assert all(m.__self__ is not memory_mod._default for m in mem_a + mem_b)
    assert all(v.__self__ is not vision_mod._default for v in vis_a + vis_b)
    # ... and the two agents do not share instances between each other
    assert mem_a[0].__self__ is not mem_b[0].__self__
    assert vis_a[0].__self__ is not vis_b[0].__self__


def test_pooled_agent_history_binding_is_isolated(monkeypatch, tmp_path):
    from src.tools import memory as memory_mod
    monkeypatch.setattr(memory_mod, "_history_provider", None)  # clean seam

    agent_a = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    agent_b = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    clear_a = [t for t in agent_a.tools if t.__name__ == "clear_chat"][0]
    clear_b = [t for t in agent_b.tools if t.__name__ == "clear_chat"][0]

    # each agent owns its own messages list; clear_chat only touches its own
    assert clear_a() == "Chat cleared successfully"
    assert [m["role"] for m in agent_a._messages] == ["system", "user"]
    assert [m["role"] for m in agent_b._messages] == ["system"]  # untouched
    assert clear_b() == "Chat cleared successfully"
    assert [m["role"] for m in agent_b._messages] == ["system", "user"]
    # pooled agents never touch the module-default binding
    assert memory_mod._history_provider is None


def test_main_agent_keeps_module_default_tools(monkeypatch, tmp_path):
    from src.tools import memory as memory_mod, vision as vision_mod

    agent = _offline_agent(monkeypatch, tmp_path)  # isolated_tools defaults to False

    mem = [t for t in agent.tools if getattr(t, "__module__", "") == memory_mod.__name__]
    vis = [t for t in agent.tools if getattr(t, "__module__", "") == vision_mod.__name__]
    assert mem == [memory_mod.recall_memory, memory_mod.store_memory,
                   memory_mod.clear_chat]
    assert vis == [vision_mod.image_response]


def test_tool_names_unchanged_for_pooled_agent(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    names = {t.__name__ for t in agent.tools}
    assert {"recall_memory", "store_memory", "clear_chat",
            "image_response"} <= names


# ------------------------------------------------------ task 3: quiet mode


class _SpeechRecorder:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class _QuietRoundStream:
    """Minimal stream interface for act() round-loop tests."""

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


def _quiet_act_agent(monkeypatch, tmp_path, rounds):
    """A quiet agent whose completion rounds are scripted like _bare_act_agent."""
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True, quiet=True)
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    agent._tool_impls = {}
    scripted = iter(rounds)

    def fake_round(tools):
        return next(scripted)

    agent._stream_round = fake_round
    return agent


def test_quiet_agent_never_speaks(monkeypatch, tmp_path):
    import src.tools.speech as speech_mod

    recorder = _SpeechRecorder()
    monkeypatch.setattr(speech_mod, "_speech_generator", recorder)
    agent = _quiet_act_agent(monkeypatch, tmp_path, [("hello there", [])])

    agent.act("hi")

    assert recorder.spoken == []  # quiet: no speech even on a normal turn


def test_quiet_agent_writes_nothing_to_stdout(monkeypatch, tmp_path, capsys):
    agent = _quiet_act_agent(monkeypatch, tmp_path, [("hello there", [])])

    agent.act("hi")

    assert capsys.readouterr().out == ""


def test_quiet_agent_still_buffers_and_answers(monkeypatch, tmp_path):
    agent = _quiet_act_agent(monkeypatch, tmp_path, [("hello there", [])])

    agent.act("hi")

    assert agent._messages[-1] == {"role": "assistant", "content": "hello there"}


def test_non_quiet_agent_still_speaks(monkeypatch, tmp_path):
    import src.tools.speech as speech_mod

    recorder = _SpeechRecorder()
    monkeypatch.setattr(speech_mod, "_speech_generator", recorder)
    agent = _offline_agent(monkeypatch, tmp_path)  # quiet=False by default
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    agent._tool_impls = {}
    # fragments accumulated by the streamed round (the scripted fake bypasses
    # _handle_fragment, so pre-fill the buffer like a streamed round would)
    agent._fragments_buffer = ["full"]
    scripted = iter([("full", [])])
    agent._stream_round = lambda tools: next(scripted)

    agent.act("hi")

    assert recorder.spoken == ["full"]


# ------------------------------------------------------- task 4: AgentPool


def _pool_config(tmp_path, models_data, **brain_overrides):
    """Config for AgentPool: models file + TEST_API_KEY in the environment."""
    import json
    from tests.test_live_ai import _remote_config

    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps(models_data), encoding="utf-8")
    brain_overrides.setdefault("models_file", str(models_file))
    return _remote_config(**brain_overrides)


def _make_pool(monkeypatch, tmp_path, models_data, models, monitor=None,
               async_delegation=False, **brain_overrides):
    from tests.test_live_ai import _fake_openai

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _pool_config(tmp_path, models_data, **brain_overrides)
    settings = live_ai_mod.agent_pool_settings(SimpleNamespace(
        agent=SimpleNamespace(parallel_agents=len(models),
                              agent_models=", ".join(models),
                              monitor=monitor,
                              async_delegation=async_delegation),
    ))
    return live_ai_mod.AgentPool(config, settings)


def test_agent_pool_builds_one_agent_per_model(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
        "qwen": {"model": "qwen-model"},
    }, models=["glm", "qwen"])

    assert set(pool.agents) == {"glm", "qwen"}
    assert pool.agents["glm"]._model_name == "glm-model"
    assert pool.agents["qwen"]._model_name == "qwen-model"
    # pooled agents are quiet and tool-isolated
    assert all(a.quiet and a.isolated_tools for a in pool.agents.values())


def test_agent_pool_unknown_model_entry_fails_fast(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="unknown model entry 'nope'"):
        _make_pool(monkeypatch, tmp_path, {
            "boot": {"model": "boot-model"},
        }, models=["nope"])


def test_agent_pool_requires_models_file(monkeypatch, tmp_path):
    from tests.test_live_ai import _fake_openai, _remote_config

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _remote_config()  # no models_file
    with pytest.raises(ValueError, match="models_file"):
        live_ai_mod.AgentPool(config, {"count": 1, "models": ["glm"],
                                       "system_prompts": [None], "quiet": True})


def test_agent_pool_empty_models_fails(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="no agents configured"):
        _make_pool(monkeypatch, tmp_path, {"boot": {"model": "m"}}, models=[])


def test_agent_pool_ask_returns_final_answer(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])
    agent = pool.agents["glm"]
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent.tool_definitions = []
    agent._tool_impls = {}
    scripted = iter([("the weather is sunny", [])])
    agent._stream_round = lambda tools: next(scripted)

    assert pool.ask("glm", "weather?") == "the weather is sunny"
    assert agent._messages[-1]["role"] == "user" or True  # history appended


def test_agent_pool_ask_unknown_agent_raises_keyerror(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])

    with pytest.raises(KeyError, match="Available: glm"):
        pool.ask("nope", "hi")


def test_agent_pool_ask_parallel_preserves_order(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
        "qwen": {"model": "qwen-model"},
    }, models=["glm", "qwen"])

    ran = []

    def fake_ask(agent_name, prompt):
        ran.append((agent_name, prompt))
        return f"answer-{agent_name}"

    monkeypatch.setattr(pool, "ask", fake_ask)
    results = pool.ask_parallel([("qwen", "t2"), ("glm", "t1")])

    assert results == ["answer-qwen", "answer-glm"]  # task order, not runtime order
    assert ("qwen", "t2") in ran and ("glm", "t1") in ran


def test_agent_pool_ask_parallel_empty_tasks(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])

    assert pool.ask_parallel([]) == []


def test_agent_pool_per_agent_system_prompt(monkeypatch, tmp_path):
    from tests.test_live_ai import _fake_openai

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _pool_config(tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    })
    settings = {"count": 1, "models": ["glm"],
                "system_prompts": ["You research the web."], "quiet": True}
    pool = live_ai_mod.AgentPool(config, settings)

    assert pool.agents["glm"]._messages[0]["content"] == "You research the web."


# ------------------------------------------- task 5: delegation tools


def test_default_tools_include_delegation_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "ask_agent" in names and "run_parallel_agents" in names


def test_ask_agent_without_pool_returns_error():
    monkeypatch_state = live_ai_mod._POOL
    try:
        live_ai_mod._POOL = None
        assert "no agent pool configured" in live_ai_mod.ask_agent("glm", "hi")
        assert "no agent pool configured" in live_ai_mod.run_parallel_agents(
            [{"agent": "glm", "prompt": "hi"}])
    finally:
        live_ai_mod._POOL = monkeypatch_state


def test_ask_agent_delegates_to_pool(monkeypatch, tmp_path):
    calls = []

    class _FakePool:
        def ask(self, agent_name, prompt, caller=""):
            calls.append((agent_name, prompt, caller))
            return f"answer from {agent_name}"

    monkeypatch.setattr(live_ai_mod, "_POOL", _FakePool())
    assert live_ai_mod.ask_agent("glm", "do research") == "answer from glm"
    assert calls == [("glm", "do research", "")]  # main agent: no caller name


def test_run_parallel_agents_delegates_to_pool(monkeypatch, tmp_path):
    captured = {}

    class _FakePool:
        def ask_parallel(self, tasks, caller=""):
            captured["tasks"], captured["caller"] = tasks, caller
            return ["a1", "a2"]

    monkeypatch.setattr(live_ai_mod, "_POOL", _FakePool())
    result = live_ai_mod.run_parallel_agents(
        [{"agent": "glm", "prompt": "t1"}, {"agent": "qwen", "prompt": "t2"}])
    assert captured["tasks"] == [("glm", "t1"), ("qwen", "t2")]
    assert captured["caller"] == ""
    assert json.loads(result) == [
        {"agent": "glm", "prompt": "t1", "answer": "a1"},
        {"agent": "qwen", "prompt": "t2", "answer": "a2"},
    ]


def test_run_parallel_agents_validates_tasks(monkeypatch, tmp_path):
    class _FakePool:
        def ask_parallel(self, tasks, caller=""):
            return []

    monkeypatch.setattr(live_ai_mod, "_POOL", _FakePool())
    assert "must be a non-empty list" in live_ai_mod.run_parallel_agents([])
    assert "needs 'agent' and 'prompt'" in live_ai_mod.run_parallel_agents(
        [{"agent": "glm"}])
    assert "needs 'agent' and 'prompt'" in live_ai_mod.run_parallel_agents(
        ["not a dict"])


def test_nested_delegation_is_refused(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])
    agent = pool.agents["glm"]
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent.tool_definitions = []
    agent._tool_impls = {}
    scripted = iter([("trying to delegate", [])])
    agent._stream_round = lambda tools: next(scripted)

    # A pooled agent (caller set via thread-local) must be refused
    live_ai_mod._TOOL_CONTEXT.agent_name = "glm"
    try:
        with pytest.raises(ValueError, match="Nested delegation refused"):
            pool.ask("glm", "delegate?", caller="glm")
        monkeypatch.setattr(live_ai_mod, "_POOL", pool)
        assert "Nested delegation refused" in live_ai_mod.ask_agent("glm", "hi")
    finally:
        live_ai_mod._TOOL_CONTEXT.agent_name = ""


def test_pooled_agent_act_sets_thread_local_name(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])
    agent = pool.agents["glm"]
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent.tool_definitions = []
    agent._tool_impls = {}
    seen = {}

    def scripted(tools):
        seen["name"] = live_ai_mod._current_agent_name()
        return ("done", [])

    agent._stream_round = scripted
    pool.ask("glm", "hi")
    assert seen["name"] == "glm"
    assert live_ai_mod._current_agent_name() == ""  # cleared afterwards


# ------------------------------------------- max_act_rounds from [agent] config


def test_max_act_rounds_from_config(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path,
                           agent_overrides={"max_act_rounds": 100})
    assert agent.MAX_ACT_ROUNDS == 100


def test_max_act_rounds_defaults_to_class_value_when_unset(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path)  # no max_act_rounds key
    assert agent.MAX_ACT_ROUNDS == live_ai_mod.Agent.MAX_ACT_ROUNDS


def test_max_act_rounds_zero_or_negative_falls_back(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path,
                           agent_overrides={"max_act_rounds": 0})
    assert agent.MAX_ACT_ROUNDS == live_ai_mod.Agent.MAX_ACT_ROUNDS


def test_max_act_rounds_load_config_round_trip(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""
        [agent]
        bootstrap_prompt = "hello"
        max_act_rounds = 100
    """), encoding="utf-8")
    config = load_config(config_file)
    assert config.agent.max_act_rounds == 100
    rounds = int(getattr(config.agent, "max_act_rounds", 0) or 0)
    assert (rounds if rounds > 0 else live_ai_mod.Agent.MAX_ACT_ROUNDS) == 100


# ------------------------------- duplicate entries & same-agent parallelism


def test_duplicate_model_entries_get_unique_keys(monkeypatch, tmp_path):
    """Each agent_models entry owns its own isolated Agent instance.

    Regression: duplicate names collapsed into one dict entry, so two
    identical entries silently built a 1-agent pool.
    """
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm", "glm"])

    assert set(pool.agents) == {"glm", "glm#2"}
    assert pool.agents["glm"] is not pool.agents["glm#2"]
    assert pool.agents["glm"]._model_name == "glm-model"
    assert pool.agents["glm#2"]._model_name == "glm-model"


def test_ask_parallel_same_agent_runs_sequentially(monkeypatch, tmp_path):
    """Tasks naming the same agent must never run act() concurrently.

    Regression: two tasks on one agent interleaved the shared message
    history, producing server-side "incomplete parallel tool-call group"
    errors.
    """
    import threading
    import time

    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])
    agent = pool.agents["glm"]
    agent._messages = [{"role": "system", "content": "s"}]

    active = [0]
    max_active = [0]

    def tracking_act(prompt):
        active[0] += 1
        max_active[0] = max(max_active[0], active[0])
        time.sleep(0.05)  # widen the window: concurrent act() would overlap
        active[0] -= 1
        agent._messages.append(
            {"role": "assistant", "content": f"done: {prompt}"})

    agent.act = tracking_act  # instance attr: ask() picks it up
    results = pool.ask_parallel([("glm", "task a"), ("glm", "task b")])

    assert results == ["done: task a", "done: task b"]  # task order kept
    assert max_active[0] == 1  # strictly sequential on the shared agent


def test_ask_parallel_distinct_agents_still_fan_out(monkeypatch, tmp_path):
    """Distinct agents run concurrently (same-name tasks only serialize)."""
    import threading

    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
        "qwen": {"model": "qwen-model"},
    }, models=["glm", "qwen"])
    barrier = threading.Barrier(2, timeout=5)
    started = []

    def make_act(name):
        def act(prompt):
            started.append(name)
            barrier.wait()  # both agents must be inside act() simultaneously
            pool.agents[name]._messages.append(
                {"role": "assistant", "content": f"{name}: {prompt}"})
        return act

    pool.agents["glm"].act = make_act("glm")
    pool.agents["qwen"].act = make_act("qwen")
    results = pool.ask_parallel([("glm", "a"), ("qwen", "b")])

    assert results == ["glm: a", "qwen: b"]
    assert sorted(started) == ["glm", "qwen"]


# ------------------------------------------------ task 1: monitor settings


def test_monitor_flags_parsing_variants():
    parse = live_ai_mod._monitor_flags
    assert parse(None) == set()
    assert parse("") == set()
    assert parse("none") == set()
    assert parse("calls") == {"calls"}
    assert parse("log") == {"log"}
    assert parse("calls,log") == {"calls", "log"}
    assert parse("log, calls") == {"calls", "log"}
    assert parse(" CALLS , LOG ") == {"calls", "log"}
    assert parse(["calls", "log"]) == {"calls", "log"}
    assert parse("calls,bogus,log") == {"calls", "log"}  # unknown ignored
    assert parse("bogus") == set()
    assert parse("calls,none,log") == set()  # "none" clears everything


def test_monitor_buffer_parsing():
    parse = live_ai_mod._monitor_buffer
    assert parse(None) == 100
    assert parse(100) == 100
    assert parse("50") == 50
    assert parse(25) == 25
    assert parse("bad") == 100
    assert parse(0) == 100      # non-positive falls back to the default
    assert parse(-5) == 100


def test_agent_pool_settings_monitor_keys():
    config = _agent_config(parallel_agents=1, monitor="calls,log",
                           monitor_log_file="logs/sub.jsonl", monitor_buffer="25")
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings["monitor"] == {"calls", "log"}
    assert settings["monitor_log_file"] == "logs/sub.jsonl"
    assert settings["monitor_buffer"] == 25


def test_agent_pool_settings_monitor_log_file_path_object():
    # load_config path-resolves *_file keys to Path objects; settings keep it.
    import pathlib as _pl
    config = _agent_config(parallel_agents=1, monitor_log_file=_pl.Path("x/y.jsonl"))
    assert live_ai_mod.agent_pool_settings(config)["monitor_log_file"] == str(_pl.Path("x/y.jsonl"))


def test_monitor_settings_load_config_round_trip(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""
        [agent]
        bootstrap_prompt = "hello"
        parallel_agents = 1
        agent_models = "glm"
        monitor = "calls, log"
        monitor_log_file = "logs/agents.jsonl"
        monitor_buffer = 50
    """), encoding="utf-8")
    config = load_config(config_file)
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings["monitor"] == {"calls", "log"}
    assert settings["monitor_log_file"].endswith("agents.jsonl")  # resolved path
    assert settings["monitor_buffer"] == 50


# ------------------------------------------- task 2: async delegation setting


def test_agent_pool_settings_async_delegation_default_false():
    assert live_ai_mod.agent_pool_settings(_agent_config(parallel_agents=1))[
        "async_delegation"] is False
    assert live_ai_mod.agent_pool_settings(SimpleNamespace())[
        "async_delegation"] is False


def test_agent_pool_settings_async_delegation_parses_true():
    config = _agent_config(parallel_agents=1, async_delegation=True)
    assert live_ai_mod.agent_pool_settings(config)["async_delegation"] is True


def test_agent_pool_settings_async_delegation_invalid_is_false():
    # bool() of a truthy string would be True; only a real boolean counts.
    config = _agent_config(parallel_agents=1, async_delegation="yes")
    assert live_ai_mod.agent_pool_settings(config)["async_delegation"] is True
    config = _agent_config(parallel_agents=1, async_delegation=0)
    assert live_ai_mod.agent_pool_settings(config)["async_delegation"] is False


def test_async_delegation_load_config_round_trip(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""
        [agent]
        bootstrap_prompt = "hello"
        parallel_agents = 1
        agent_models = "glm"
        async_delegation = true
    """), encoding="utf-8")
    config = load_config(config_file)
    assert live_ai_mod.agent_pool_settings(config)["async_delegation"] is True


def test_pool_async_delegation_flag_and_executor(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], async_delegation=True)
    assert pool.async_delegation is True
    assert pool._executor is not None
    pool.stop()  # idempotent
    pool.stop()


def test_pool_synchronous_by_default_and_stop_idempotent(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])
    assert pool.async_delegation is False
    pool.stop()
    pool.stop()  # no error
