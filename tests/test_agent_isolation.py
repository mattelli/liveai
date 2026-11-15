"""Tests for the isolated/quiet Agent capabilities kept after the pool removal.

The scheduler execution agent is built with Agent(config, isolated_tools=True,
quiet=True); these tests cover per-agent tool isolation, quiet mode and the
[agent] max_act_rounds setting.
"""

from types import SimpleNamespace

import textwrap

import src.live_ai as live_ai_mod  # noqa: E402
from src.common import load_config  # noqa: E402

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
        brain=SimpleNamespace(memory_path_cwd=str(tmp_path / "history.txt"),
                              system_prompt="be helpful"),
        agent=SimpleNamespace(bootstrap_prompt="bootstrap"),
    ))
    config = _remote_config()
    if agent_overrides:
        config.agent = SimpleNamespace(
            bootstrap_prompt="bootstrap", **agent_overrides)
    return live_ai_mod.Agent(config, **kwargs)


def test_isolated_agent_gets_own_memory_and_vision_instances(monkeypatch, tmp_path):
    from src.tools import memory as memory_mod, vision as vision_mod

    agent_a = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    agent_b = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)

    def find(agent, module):
        return [t for t in agent.tools if getattr(t, "__module__", "") == module.__name__]

    mem_a, mem_b = find(agent_a, memory_mod), find(agent_b, memory_mod)
    vis_a, vis_b = find(agent_a, vision_mod), find(agent_b, vision_mod)
    assert len(mem_a) == 4 and len(vis_a) == 1  # recall/store/clear/compact + image
    # fresh instances per agent: none is the module-default instance
    assert all(m.__self__ is not memory_mod._default for m in mem_a + mem_b)
    assert all(v.__self__ is not vision_mod._default for v in vis_a + vis_b)
    # ... and the two agents do not share instances between each other
    assert mem_a[0].__self__ is not mem_b[0].__self__
    assert vis_a[0].__self__ is not vis_b[0].__self__


def test_isolated_agent_history_binding_is_isolated(monkeypatch, tmp_path):
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
                   memory_mod.clear_chat, memory_mod.compact_chat]
    assert vis == [vision_mod.image_response]


def test_tool_names_unchanged_for_isolated_agent(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    names = {t.__name__ for t in agent.tools}
    assert {"recall_memory", "store_memory", "clear_chat", "compact_chat",
            "image_response"} <= names


def test_compact_chat_in_memory_topic():
    from src.live_ai import _TOOL_TOPICS
    assert "compact_chat" in _TOOL_TOPICS["memory"]


def test_isolated_agent_binds_its_own_summarizer(monkeypatch, tmp_path):
    """An isolated agent's compact_chat uses the LLM summarizer, not the fallback."""
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    compact = [t for t in agent.tools if t.__name__ == "compact_chat"][0]
    instance = compact.__self__

    assert instance._summarizer is not None
    assert instance._summarizer == agent._summarize_messages
    # the instance is also kept for auto-compaction
    assert agent._memory_tools is instance


def test_isolated_agent_compact_chat_uses_bound_summarizer(monkeypatch, tmp_path):
    """End to end: the per-agent summarizer produces the injected summary."""
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True)
    compact = [t for t in agent.tools if t.__name__ == "compact_chat"][0]
    instance = compact.__self__
    seen = []
    instance.bind_summarizer(lambda old: seen.append(list(old)) or "MODEL SUMMARY")
    agent._messages = (
        [{"role": "system", "content": "be helpful"}]
        + [{"role": "user", "content": f"old {i}"} for i in range(6)]
        + [{"role": "user", "content": "recent"}]
    )

    result = compact()

    assert result.startswith("Chat compacted:")
    assert "(extractive fallback)" not in result
    assert "MODEL SUMMARY" in agent._messages[2]["content"]
    assert seen  # the bound summarizer was actually called


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


