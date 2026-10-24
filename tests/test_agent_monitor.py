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
    }, models=["glm"])
    pool.monitor_flags = {"calls"}
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

    assert pool.monitor.snapshot() == []


def test_pool_ask_failure_still_records_start(monkeypatch, tmp_path):
    pool = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "glm": {"model": "glm-model"},
    }, models=["glm"])
    pool.monitor_flags = {"calls"}
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
    }, models=["glm"])
    pool.monitor_flags = {"calls"}
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
    }, models=["glm"])
    pool.monitor_flags = {"calls"}
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
    }, models=["glm"])
    pool_a.monitor_flags = {"calls"}
    pool_b = _make_pool(monkeypatch, tmp_path, {
        "boot": {"model": "boot-model"},
        "qwen": {"model": "qwen-model"},
    }, models=["qwen"])
    pool_b.monitor_flags = {"calls"}
    assert pool_b.monitor is pool_a.monitor
