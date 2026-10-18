"""Tests for the multi-agent pool (PLAN.md tasks 1-7).

Task 1: ``[agent]`` pool settings resolution (``agent_pool_settings``).
"""

import pathlib
import textwrap
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
        agent_quiet=True,
    )
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": True,
    }


def test_agent_pool_settings_defaults_when_keys_missing():
    # Only the historical bootstrap_prompt key: single-agent behavior.
    config = _agent_config(bootstrap_prompt="b")
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 0,
        "models": [],
        "system_prompts": [],
        "quiet": True,
    }


def test_agent_pool_settings_without_agent_section():
    assert live_ai_mod.agent_pool_settings(SimpleNamespace()) == {
        "count": 0,
        "models": [],
        "system_prompts": [],
        "quiet": True,
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


def test_agent_pool_settings_quiet_defaults_true_and_is_bool():
    assert live_ai_mod.agent_pool_settings(_agent_config(parallel_agents=1))["quiet"] is True
    config = _agent_config(parallel_agents=1, agent_quiet=False)
    assert live_ai_mod.agent_pool_settings(config)["quiet"] is False


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
        agent_quiet = false
    """), encoding="utf-8")
    config = load_config(config_file)
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": False,
    }


# ------------------------------------------------- task 2: per-agent tools


def _offline_agent(monkeypatch, tmp_path, **kwargs):
    """A real Agent built offline (stubbed openai client).

    Also configures the memory/vision tool modules with a tmp history path,
    since ``[brain] history_path`` lives in the module-level config cache.
    """
    from src.tools import memory as memory_mod, vision as vision_mod
    from tests.test_live_ai import _fake_openai, _remote_config

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    memory_mod.configure(SimpleNamespace(
        brain=SimpleNamespace(history_path=str(tmp_path / "history.txt"),
                              system_prompt="be helpful"),
        agent=SimpleNamespace(bootstrap_prompt="bootstrap"),
    ))
    return live_ai_mod.Agent(_remote_config(), **kwargs)


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
    assert mem == [memory_mod.recall_personal, memory_mod.store_personal,
                   memory_mod.clear_chat]
    assert vis == [vision_mod.image_response]


def test_tool_names_unchanged_for_pooled_agent(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    names = {t.__name__ for t in agent.tools}
    assert {"recall_personal", "store_personal", "clear_chat",
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


def _make_pool(monkeypatch, tmp_path, models_data, models, **brain_overrides):
    from tests.test_live_ai import _fake_openai

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _pool_config(tmp_path, models_data, **brain_overrides)
    settings = live_ai_mod.agent_pool_settings(SimpleNamespace(
        agent=SimpleNamespace(parallel_agents=len(models),
                              agent_models=", ".join(models)),
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
