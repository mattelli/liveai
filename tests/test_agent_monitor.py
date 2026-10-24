"""Tests for the sub-agent activity monitor (PLAN_MONITOR.md task 2)."""

import json
import threading

import pytest

import src.live_ai as live_ai_mod
from src.live_ai import AgentMonitor


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
