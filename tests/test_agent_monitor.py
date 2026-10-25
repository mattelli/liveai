"""Tests for the sub-agent activity monitor (PLAN_MONITOR.md task 2)."""

import json
import threading

import pytest

import src.live_ai as live_ai_mod
from src.live_ai import AgentMonitor


@pytest.fixture(autouse=True)
def _reset_process_monitor():
    """Keep the module-level monitor from leaking between tests."""
    previous = live_ai_mod.get_monitor()
    live_ai_mod.bind_monitor(None)
    yield
    live_ai_mod.bind_monitor(previous)


# ------------------------------------------------------------- ring buffer


def test_record_appends_and_snapshot_is_oldest_first():
    monitor = AgentMonitor(maxlen=10)
    monitor.record("glm", "start", prompt="hi")
    monitor.record("glm", "answer", answer="hello")
    events = monitor.snapshot()
    assert [e["event"] for e in events] == ["start", "answer"]
    assert events[0]["agent"] == "glm"
    assert len(monitor) == 2


def test_ring_buffer_drops_oldest_beyond_maxlen():
    monitor = AgentMonitor(maxlen=3)
    for i in range(5):
        monitor.record("glm", "tool_call", tool=f"t{i}")
    events = monitor.snapshot()
    assert len(events) == 3
    assert [e["tool"] for e in events] == ["t2", "t3", "t4"]


def test_snapshot_limit_returns_most_recent():
    monitor = AgentMonitor(maxlen=10)
    for i in range(5):
        monitor.record("glm", "tool_call", tool=f"t{i}")
    assert [e["tool"] for e in monitor.snapshot(limit=2)] == ["t3", "t4"]
    assert len(monitor.snapshot(limit=100)) == 5


def test_snapshot_returns_a_copy():
    monitor = AgentMonitor(maxlen=5)
    monitor.record("glm", "start")
    snap = monitor.snapshot()
    snap.append({"bogus": True})
    assert len(monitor) == 1


def test_clear_empties_buffer():
    monitor = AgentMonitor(maxlen=5)
    monitor.record("glm", "start")
    monitor.clear()
    assert monitor.snapshot() == []
    assert len(monitor) == 0


def test_events_carry_utc_iso_timestamp():
    from datetime import datetime

    monitor = AgentMonitor(maxlen=5)
    monitor.record("glm", "start")
    ts = monitor.snapshot()[0]["ts"]
    parsed = datetime.fromisoformat(ts)  # raises on a non-ISO value
    assert parsed.tzinfo is not None  # UTC-aware


# ------------------------------------------------------------- JSONL sink


def test_log_file_written_one_json_object_per_line(tmp_path):
    log_file = tmp_path / "agents.jsonl"
    monitor = AgentMonitor(maxlen=10, log_file=str(log_file))
    monitor.record("glm", "start", prompt="hi")
    monitor.record("glm", "answer", answer="hello")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event"] == "start" and first["prompt"] == "hi"
    assert second["event"] == "answer" and second["answer"] == "hello"
    assert all("ts" in e and "agent" in e for e in (first, second))


def test_log_file_parent_directory_created_lazily(tmp_path):
    log_file = tmp_path / "nested" / "dir" / "agents.jsonl"
    monitor = AgentMonitor(maxlen=10, log_file=str(log_file))
    monitor.record("glm", "start")
    assert log_file.exists()
    assert len(json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])) >= 3


def test_no_log_file_no_file_created(tmp_path):
    before = set(p.name for p in tmp_path.iterdir())
    monitor = AgentMonitor(maxlen=10, log_file=None)
    monitor.record("glm", "start")
    monitor.record("glm", "answer")
    assert set(p.name for p in tmp_path.iterdir()) == before
    assert len(monitor) == 2  # buffer still populated


def test_log_write_failure_is_swallowed_and_sink_dropped(tmp_path):
    # A directory as log path: open() fails on every attempt.
    monitor = AgentMonitor(maxlen=10, log_file=str(tmp_path))
    monitor.record("glm", "start")   # write fails -> sink dropped
    monitor.record("glm", "answer")  # must not raise either
    assert len(monitor) == 2


def test_unicode_event_written_without_escaping(tmp_path):
    log_file = tmp_path / "agents.jsonl"
    monitor = AgentMonitor(maxlen=10, log_file=str(log_file))
    monitor.record("glm", "answer", answer="risposta con Ã ccenti âœ“")
    line = json.loads(log_file.read_text(encoding="utf-8"))
    assert line["answer"] == "risposta con Ã ccenti âœ“"


# ------------------------------------------------------------- concurrency


def test_concurrent_records_do_not_interleave_lines(tmp_path):
    log_file = tmp_path / "agents.jsonl"
    monitor = AgentMonitor(maxlen=1000, log_file=str(log_file))
    threads = 8
    per_thread = 25

    def worker(worker_id):
        for i in range(per_thread):
            monitor.record(f"agent-{worker_id}", "tool_call", tool=f"t{i}")

    runners = [threading.Thread(target=worker, args=(w,)) for w in range(threads)]
    for runner in runners:
        runner.start()
    for runner in runners:
        runner.join()

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == threads * per_thread  # exactly 200 complete lines
    parsed = [json.loads(line) for line in lines]  # every line parses
    assert len({(e["agent"], e["tool"]) for e in parsed}) == threads * per_thread
    assert len(monitor) == 200  # all events buffered (under the maxlen cap)


# ------------------------------------------------- module-level binding


def test_bind_monitor_and_get_monitor():
    previous = live_ai_mod.get_monitor()
    try:
        monitor = AgentMonitor(maxlen=5)
        live_ai_mod.bind_monitor(monitor)
        assert live_ai_mod.get_monitor() is monitor
        live_ai_mod.bind_monitor(None)
        assert live_ai_mod.get_monitor() is None
    finally:
        live_ai_mod.bind_monitor(previous)


def test_monitor_maxlen_never_zero():
    assert AgentMonitor(maxlen=0).maxlen >= 1
    assert AgentMonitor(maxlen=-3).maxlen >= 1


# ------------------------------------------- monitor settings + build_monitor


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


def test_monitor_settings_without_agent_section():
    from types import SimpleNamespace

    assert live_ai_mod.monitor_settings(SimpleNamespace()) == {
        "monitor": set(),
        "monitor_log_file": "agents.jsonl",
        "monitor_buffer": 100,
    }


def test_monitor_settings_keys():
    from types import SimpleNamespace

    config = SimpleNamespace(agent=SimpleNamespace(
        monitor="calls,log", monitor_log_file="logs/sub.jsonl",
        monitor_buffer="25"))
    settings = live_ai_mod.monitor_settings(config)
    assert settings["monitor"] == {"calls", "log"}
    assert settings["monitor_log_file"] == "logs/sub.jsonl"
    assert settings["monitor_buffer"] == 25


def test_monitor_settings_log_file_path_object():
    # load_config path-resolves *_file keys to Path objects; settings keep it.
    import pathlib as _pl
    from types import SimpleNamespace

    config = SimpleNamespace(agent=SimpleNamespace(
        monitor_log_file=_pl.Path("x/y.jsonl")))
    assert live_ai_mod.monitor_settings(config)["monitor_log_file"] == str(_pl.Path("x/y.jsonl"))


def test_monitor_settings_load_config_round_trip(tmp_path):
    from src.common import load_config
    import textwrap

    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""
        [agent]
        bootstrap_prompt = "hello"
        monitor = "calls, log"
        monitor_log_file = "logs/agents.jsonl"
        monitor_buffer = 50
    """), encoding="utf-8")
    config = load_config(config_file)
    settings = live_ai_mod.monitor_settings(config)
    assert settings["monitor"] == {"calls", "log"}
    assert settings["monitor_log_file"].endswith("agents.jsonl")  # resolved path
    assert settings["monitor_buffer"] == 50


def test_build_monitor_disabled_returns_none():
    from types import SimpleNamespace

    previous = live_ai_mod.get_monitor()
    try:
        assert live_ai_mod.build_monitor(SimpleNamespace()) is None
        assert live_ai_mod.get_monitor() is None
    finally:
        live_ai_mod.bind_monitor(previous)


def test_build_monitor_enabled_binds_ring_buffer(tmp_path):
    from types import SimpleNamespace

    previous = live_ai_mod.get_monitor()
    try:
        config = SimpleNamespace(agent=SimpleNamespace(
            monitor="calls", monitor_buffer="7"))
        monitor = live_ai_mod.build_monitor(config)
        assert monitor is live_ai_mod.get_monitor()
        assert monitor.maxlen == 7
        monitor.record("scheduler", "start", prompt="x")
        assert [e["agent"] for e in monitor.snapshot()] == ["scheduler"]
    finally:
        live_ai_mod.bind_monitor(previous)


def test_build_monitor_log_flag_only_creates_jsonl_sink(tmp_path):
    from types import SimpleNamespace

    previous = live_ai_mod.get_monitor()
    log_file = tmp_path / "agents.jsonl"
    try:
        config = SimpleNamespace(agent=SimpleNamespace(
            monitor="log", monitor_log_file=str(log_file)))
        monitor = live_ai_mod.build_monitor(config)
        monitor.record("scheduler", "answer", answer="done")
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["answer"] == "done"
    finally:
        live_ai_mod.bind_monitor(previous)


def test_build_monitor_calls_flag_no_log_file(tmp_path):
    from types import SimpleNamespace

    previous = live_ai_mod.get_monitor()
    before = set(p.name for p in tmp_path.iterdir())
    try:
        config = SimpleNamespace(agent=SimpleNamespace(monitor="calls"))
        monitor = live_ai_mod.build_monitor(config)
        monitor.record("scheduler", "start")
        assert set(p.name for p in tmp_path.iterdir()) == before  # no JSONL file
        assert len(monitor) == 1  # ring buffer still populated
    finally:
        live_ai_mod.bind_monitor(previous)


def test_main_setup_monitor_binds_when_enabled(monkeypatch):
    """main._setup_monitor builds + binds the monitor from [agent] settings."""
    import types
    import main

    captured = {}
    monkeypatch.setattr(
        "src.live_ai.build_monitor",
        lambda config: captured.setdefault("bound", config))
    config = types.SimpleNamespace(agent=types.SimpleNamespace(monitor="calls"))
    main._setup_monitor(config)
    assert captured["bound"] is config


def test_main_setup_monitor_noop_when_disabled(monkeypatch):
    import types
    import main
    import src.live_ai as live_ai

    previous = live_ai.get_monitor()
    try:
        main._setup_monitor(types.SimpleNamespace(agent=types.SimpleNamespace()))
        assert live_ai.get_monitor() is previous  # nothing was created/bound
    finally:
        live_ai.bind_monitor(previous)


# ------------------------------------------------- agent emits events (direct)


def _offline_agent(monkeypatch, tmp_path):
    """A real Agent built offline (stubbed openai client), quiet + isolated."""
    from types import SimpleNamespace
    from src.tools import memory as memory_mod
    from tests.test_live_ai import _fake_openai, _remote_config

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    memory_mod.configure(SimpleNamespace(
        brain=SimpleNamespace(memory_path=str(tmp_path / "history.txt"),
                              system_prompt="be helpful"),
        agent=SimpleNamespace(bootstrap_prompt="bootstrap"),
    ))
    return live_ai_mod.Agent(_remote_config(), isolated_tools=True, quiet=True)


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


def test_agent_tool_calls_reported_to_monitor(monkeypatch, tmp_path):
    """A quiet/isolated agent's tool calls flow into the bound monitor."""
    agent = _offline_agent(monkeypatch, tmp_path)
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent.tool_definitions = []
    agent._tool_impls = {"echo": lambda **kw: f"echo:{kw.get('text', '')}"}
    scripted = iter([
        ("", [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}]),
        ("done", []),  # second round: no tool calls -> act() ends
    ])
    agent._stream_round = lambda tools: next(scripted)

    monitor = AgentMonitor(maxlen=10)
    live_ai_mod.bind_monitor(monitor)
    live_ai_mod._TOOL_CONTEXT.agent_name = "scheduler"
    try:
        agent.act("run the tool")
    finally:
        live_ai_mod._TOOL_CONTEXT.agent_name = ""
        live_ai_mod.bind_monitor(None)

    events = monitor.snapshot()
    assert [e["event"] for e in events] == ["tool_call", "tool_result"]
    assert events[0]["agent"] == "scheduler"
    assert events[0]["tool"] == "echo"
    assert events[1]["result"] == "echo:hi"


def test_main_agent_tool_calls_not_reported(monkeypatch, tmp_path):
    """Without a thread-local agent name, tool calls stay out of the monitor."""
    agent = _offline_agent(monkeypatch, tmp_path)
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent.tool_definitions = []
    agent._tool_impls = {"echo": lambda **kw: "ok"}
    agent._stream_round = lambda tools: ("done", [])

    monitor = AgentMonitor(maxlen=10)
    live_ai_mod.bind_monitor(monitor)
    try:
        agent.act("hi")
    finally:
        live_ai_mod.bind_monitor(None)

    assert monitor.snapshot() == []


def test_agent_activity_unbound_returns_empty_message():
    live_ai_mod.bind_monitor(None)
    assert live_ai_mod.agent_activity() == "No sub-agent activity recorded."


def test_agent_activity_formats_events_newest_last():
    monitor = AgentMonitor(maxlen=10)
    monitor.record("scheduler", "start", prompt="check the news")
    monitor.record("scheduler", "answer", answer="found it")
    live_ai_mod.bind_monitor(monitor)
    try:
        report = live_ai_mod.agent_activity()
    finally:
        live_ai_mod.bind_monitor(None)
    lines = report.splitlines()
    assert len(lines) == 2
    assert "scheduler" in lines[0] and "check the news" in lines[0]
    assert "found it" in lines[1]


def test_agent_activity_last_limits_to_most_recent():
    monitor = AgentMonitor(maxlen=10)
    for i in range(5):
        monitor.record("scheduler", "tool_call", tool=f"t{i}")
    live_ai_mod.bind_monitor(monitor)
    try:
        report = live_ai_mod.agent_activity(last=2)
    finally:
        live_ai_mod.bind_monitor(None)
    lines = report.splitlines()
    assert len(lines) == 2
    assert "t3" in lines[0] and "t4" in lines[1]


def test_agent_activity_bound_but_empty():
    live_ai_mod.bind_monitor(AgentMonitor(maxlen=10))
    try:
        assert live_ai_mod.agent_activity() == "No sub-agent activity recorded."
    finally:
        live_ai_mod.bind_monitor(None)


def test_default_tools_include_agent_activity():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "agent_activity" in names


def test_agent_activity_tool_definition_well_formed():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    tools = live_ai_mod._default_tools(agent)
    definitions = [live_ai_mod.Brain._openai_tool_definition(t) for t in tools]
    by_name = {d["function"]["name"]: d for d in definitions}
    definition = by_name["agent_activity"]
    assert definition["type"] == "function"
    assert definition["function"]["description"]
    assert definition["function"]["parameters"]["type"] == "object"
