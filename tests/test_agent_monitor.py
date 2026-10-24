"""Tests for the sub-agent activity monitor (PLAN_MONITOR.md task 2)."""

import json
import threading

import pytest

import src.live_ai as live_ai_mod
from src.live_ai import AgentMonitor


from tests.test_agent_pool import _make_pool, _QuietRoundStream  # noqa: E402


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
    monitor.record("glm", "answer", answer="risposta con àccenti ✓")
    line = json.loads(log_file.read_text(encoding="utf-8"))
    assert line["answer"] == "risposta con àccenti ✓"


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


# ------------------------------------------------ task 3: pool emits events


def test_pool_ask_records_start_and_answer(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], monitor={"calls"})
    agent = pool.agents["glm"]
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    agent._tool_impls = {}
    agent._stream_round = lambda tools: ("the answer", [])

    pool.ask("glm", "do a thing")

    events = pool.monitor.snapshot()
    assert [(e["event"], e["agent"]) for e in events] == [
        ("start", "glm"), ("answer", "glm")]
    assert events[0]["prompt"] == "do a thing"
    assert events[1]["answer"] == "the answer"


def test_pool_without_calls_flag_records_nothing(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], monitor=set())
    agent = pool.agents["glm"]
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    agent._tool_impls = {}
    agent._stream_round = lambda tools: ("the answer", [])

    pool.ask("glm", "do a thing")

    # Monitoring disabled: no monitor is created at all (no buffer, no log).
    assert pool.monitor is None
    assert live_ai_mod.get_monitor() is None


def test_pool_ask_failure_still_records_start(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], monitor={"calls"})
    agent = pool.agents["glm"]
    agent.act = lambda prompt: (_ for _ in ()).throw(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        pool.ask("glm", "do a thing")

    events = pool.monitor.snapshot()
    assert [e["event"] for e in events] == ["start"]  # no "answer" on failure


def test_subagent_tool_calls_reported_to_monitor(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], monitor={"calls"})
    agent = pool.agents["glm"]
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    agent._tool_impls = {"echo": lambda **kw: f"echo:{kw.get('text', '')}"}
    scripted = iter([
        ("", [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}]),
        ("done", []),  # second round: no tool calls -> act() ends
    ])
    agent._stream_round = lambda tools: next(scripted)

    live_ai_mod._TOOL_CONTEXT.agent_name = "glm"
    try:
        agent.act("run the tool")
    finally:
        live_ai_mod._TOOL_CONTEXT.agent_name = ""

    events = pool.monitor.snapshot()
    assert [e["event"] for e in events] == ["tool_call", "tool_result"]
    assert events[0]["tool"] == "echo" and events[0]["args"] == '{"text": "hi"}'
    assert events[1]["result"] == "echo:hi"


def test_main_agent_tool_calls_not_reported(monkeypatch, tmp_path):
    """Main-agent tool activity stays out of the sub-agent monitor."""
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], monitor={"calls"})
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    agent._tool_impls = {"echo": lambda **kw: "ok"}
    agent._stream = _QuietRoundStream()

    # no thread-local agent name: main agent context
    agent._execute_tool({"name": "echo", "arguments": "{}"})

    assert pool.monitor.snapshot() == []


def test_monitor_survives_pool_rebuild(monkeypatch, tmp_path):
    """Two pools built in one process share the process-wide monitor."""
    pool_a = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"], monitor={"calls"})
    pool_b = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "qwen": {"model": "qwen-model"},
    }, models=["qwen"], monitor={"calls"})
    assert pool_b.monitor is pool_a.monitor


# ------------------------------------------- task 4: agent_activity tool


def test_agent_activity_unbound_returns_empty_message():
    live_ai_mod.bind_monitor(None)
    assert live_ai_mod.agent_activity() == "No sub-agent activity recorded."


def test_agent_activity_formats_events_newest_last():
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    monitor.record("glm", "start", prompt="research X")
    monitor.record("glm", "tool_call", tool="web_search")
    monitor.record("glm", "answer", answer="found it")
    live_ai_mod.bind_monitor(monitor)
    try:
        report = live_ai_mod.agent_activity()
    finally:
        live_ai_mod.bind_monitor(None)
    lines = report.splitlines()
    assert len(lines) == 3
    assert "glm" in lines[0] and "research X" in lines[0]
    assert "web_search" in lines[1]
    assert "found it" in lines[2]


def test_agent_activity_last_limits_to_most_recent():
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    for i in range(5):
        monitor.record("glm", "tool_call", tool=f"t{i}")
    live_ai_mod.bind_monitor(monitor)
    try:
        report = live_ai_mod.agent_activity(last=2)
    finally:
        live_ai_mod.bind_monitor(None)
    lines = report.splitlines()
    assert len(lines) == 2
    assert "t3" in lines[0] and "t4" in lines[1]


def test_agent_activity_bound_but_empty():
    from src.live_ai import AgentMonitor

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
    definitions = {d["function"]["name"]: d for d in
                   (live_ai_mod.OpenAICore._openai_tool_definition(t) for t in tools)}
    definition = definitions["agent_activity"]["function"]
    assert definition["description"]
    assert definition["parameters"]["type"] == "object"


# ------------------------------------------------- task 5: e2e smoke


def _script_agent(agent, rounds):
    """Minimal quiet-agent plumbing + scripted completion rounds."""
    agent._stream = _QuietRoundStream()
    agent._context_window = 1024
    agent._messages = [{"role": "system", "content": "s"}]
    agent.tool_definitions = []
    scripted = iter(rounds)
    agent._stream_round = lambda tools: next(scripted)


def test_e2e_ask_and_parallel_logged_and_reported(monkeypatch, tmp_path):
    """One ask() + one ask_parallel() with two agents, monitor = "calls,log".

    The JSONL file must hold the full event sequence per agent (including a
    tool call) and agent_activity() must report both agent names.
    """
    from tests.test_agent_pool import _pool_config
    from tests.test_live_ai import _fake_openai

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _pool_config(tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
        "qwen": {"model": "qwen-model"},
    })
    log_file = tmp_path / "agents.jsonl"
    settings = {"count": 2, "models": ["glm", "qwen"],
                "system_prompts": [None, None], "quiet": True,
                "monitor": {"calls", "log"},
                "monitor_log_file": str(log_file), "monitor_buffer": 100}
    pool = live_ai_mod.AgentPool(config, settings)
    assert live_ai_mod.get_monitor() is pool.monitor  # bound for agent_activity

    # glm: one tool call, then the final answer (real _execute_tool path).
    # It is delegated twice (ask + ask_parallel), so four scripted rounds.
    glm = pool.agents["glm"]
    glm._tool_impls = {"echo": lambda **kw: f"echo:{kw.get('text', '')}"}
    _script_agent(glm, [
        ("", [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}]),
        ("glm answer", []),
        ("", [{"id": "c2", "name": "echo", "arguments": '{"text": "yo"}'}]),
        ("glm answer", []),
    ])
    qwen = pool.agents["qwen"]
    qwen._tool_impls = {}
    _script_agent(qwen, [("qwen answer", [])])

    assert pool.ask("glm", "task one") == "glm answer"
    results = pool.ask_parallel([("glm", "p1"), ("qwen", "p2")])
    assert results == ["glm answer", "qwen answer"]

    # JSONL: one JSON object per line, complete event sequence per agent.
    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    glm_events = [e for e in events if e["agent"] == "glm"]
    qwen_events = [e for e in events if e["agent"] == "qwen"]
    assert [e["event"] for e in glm_events] == [
        "start", "tool_call", "tool_result", "answer",
        "start", "tool_call", "tool_result", "answer"]
    assert glm_events[0]["prompt"] == "task one"
    assert glm_events[3]["answer"] == "glm answer"
    assert glm_events[2]["result"] == "echo:hi"
    assert [e["event"] for e in qwen_events] == ["start", "answer"]
    assert qwen_events[1]["answer"] == "qwen answer"
    assert all("ts" in e for e in events)

    # agent_activity() reports both agents from the shared ring buffer.
    report = live_ai_mod.agent_activity(last=50)
    assert "glm" in report and "qwen" in report
    assert "task one" in report and "p2" in report
