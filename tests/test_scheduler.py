"""Unit tests for the scheduler tool (src/tools/scheduler.py).

No LLM, no real threads where avoidable: the engine's loop logic is driven
via the injected ``_execute`` callback and very short waits.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.tools import scheduler as scheduler_module
from src.tools.scheduler import (
    ScheduledTask, Scheduler, next_occurrence, parse_when,
)


## parse_when ##

def test_parse_when_iso():
    assert parse_when("2026-01-01T09:00:00") == datetime(2026, 1, 1, 9, 0, 0)


def test_parse_when_compact():
    assert parse_when("2026-01-01 09:00") == datetime(2026, 1, 1, 9, 0)


def test_parse_when_invalid_raises():
    for bad in ("", "   ", "tomorrow", "2026-13-01 09:00", "01/02/2026"):
        with pytest.raises(ValueError):
            parse_when(bad)


## next_occurrence ##

def test_next_occurrence_daily():
    after = datetime(2026, 1, 1, 9, 0)
    assert next_occurrence("daily", after) == after + timedelta(days=1)


def test_next_occurrence_hourly():
    after = datetime(2026, 1, 1, 9, 0)
    assert next_occurrence("hourly", after) == after + timedelta(hours=1)


def test_next_occurrence_every():
    after = datetime(2026, 1, 1, 9, 0)
    assert next_occurrence("every 30m", after) == after + timedelta(minutes=30)
    assert next_occurrence("every 2h", after) == after + timedelta(hours=2)
    assert next_occurrence("every 15s", after) == after + timedelta(seconds=15)
    assert next_occurrence("every 7d", after) == after + timedelta(days=7)


def test_next_occurrence_cron():
    # 09:30 every day: from 09:00 the next match is the same day 09:30.
    after = datetime(2026, 1, 1, 9, 0)
    assert next_occurrence("30 9 * * *", after) == datetime(2026, 1, 1, 9, 30)
    # From 09:30 exactly (strictly after): next day 09:30.
    assert next_occurrence("30 9 * * *", datetime(2026, 1, 1, 9, 30)) == \
        datetime(2026, 1, 2, 9, 30)
    # Monday only: 2026-01-01 is a Thursday -> next Monday is 2026-01-05.
    assert next_occurrence("0 9 * * MON", datetime(2026, 1, 1, 9, 0)) == \
        datetime(2026, 1, 5, 9, 0)
    # Step values: */15 minutes from 09:07 -> 09:15.
    assert next_occurrence("*/15 9 * * *", datetime(2026, 1, 1, 9, 7)) == \
        datetime(2026, 1, 1, 9, 15)


def test_next_occurrence_unknown_raises():
    after = datetime(2026, 1, 1, 9, 0)
    for bad in ("", "weekly", "every", "every 30x", "0 9 * *"):
        with pytest.raises(ValueError):
            next_occurrence(bad, after)


## Scheduler: add / cancel / list round-trip ##

class _Recorder:
    """Collects executed tasks (thread-safe) for assertions."""

    def __init__(self):
        self.calls: list[ScheduledTask] = []
        self.lock = threading.Lock()

    def __call__(self, task: ScheduledTask) -> None:
        with self.lock:
            self.calls.append(task)


def _make_scheduler(tmp_path: Path, on_missed="skip", execute=None) -> Scheduler:
    return Scheduler(persist_file=tmp_path / "schedule.json",
                     on_missed=on_missed, _execute=execute)


def test_add_cancel_list_round_trip(tmp_path):
    s = _make_scheduler(tmp_path)
    task = s.add("hello", "2030-01-01 09:00")
    assert task.id == "t-1"
    assert [t.id for t in s.list()] == ["t-1"]
    assert s.cancel("t-1") is True
    assert s.list() == []
    assert s.cancel("t-1") is False  # unknown id


def test_add_rejects_past_one_shot(tmp_path):
    s = _make_scheduler(tmp_path)
    with pytest.raises(ValueError):
        s.add("late", "2000-01-01 09:00")


def test_list_ordered_by_next_run(tmp_path):
    s = _make_scheduler(tmp_path)
    s.add("later", "2030-06-01 09:00")
    s.add("sooner", "2030-01-01 09:00")
    assert [t.id for t in s.list()] == ["t-2", "t-1"]


## persistence ##

def test_persistence_round_trip(tmp_path):
    file = tmp_path / "schedule.json"
    s1 = Scheduler(persist_file=file)
    s1.add("persist me", "2030-01-01 09:00")
    s2 = Scheduler(persist_file=file)
    assert [t.prompt for t in s2.list()] == ["persist me"]
    # ids continue after a restart (no reuse of t-1)
    assert s2.add("another", "2030-02-01 09:00").id == "t-2"


def test_persistence_corrupt_file(tmp_path):
    file = tmp_path / "schedule.json"
    file.write_text("not json", encoding="utf-8")
    s = Scheduler(persist_file=file)  # must not raise
    assert s.list() == []


## on_missed handling ##

def test_on_missed_run_executes_once(tmp_path):
    recorder = _Recorder()
    file = tmp_path / "schedule.json"
    past = (datetime.now() - timedelta(minutes=5)).isoformat()
    file.write_text(json.dumps(
        [{"id": "t-9", "prompt": "missed", "next_run": past, "repeat": ""}]),
        encoding="utf-8")
    Scheduler(persist_file=file, on_missed="run", _execute=recorder)
    assert [t.prompt for t in recorder.calls] == ["missed"]
    # and it is dropped afterwards (not re-run on the next load)
    assert Scheduler(persist_file=file, on_missed="run", _execute=recorder).list() == []


def test_on_missed_skip_drops(tmp_path):
    recorder = _Recorder()
    file = tmp_path / "schedule.json"
    past = (datetime.now() - timedelta(minutes=5)).isoformat()
    file.write_text(json.dumps(
        [{"id": "t-9", "prompt": "missed", "next_run": past, "repeat": ""}]),
        encoding="utf-8")
    s = Scheduler(persist_file=file, on_missed="skip", _execute=recorder)
    assert recorder.calls == []
    assert s.list() == []


def test_on_missed_recurring_stays_scheduled(tmp_path):
    file = tmp_path / "schedule.json"
    past = (datetime.now() - timedelta(minutes=5)).isoformat()
    file.write_text(json.dumps(
        [{"id": "t-9", "prompt": "tick", "next_run": past, "repeat": "daily"}]),
        encoding="utf-8")
    s = Scheduler(persist_file=file, on_missed="skip")
    assert [t.id for t in s.list()] == ["t-9"]  # recurring tasks are kept


## loop execution (short real waits) ##

def test_loop_runs_due_one_shot(tmp_path):
    recorder = _Recorder()
    s = _make_scheduler(tmp_path, execute=recorder)
    soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
    s.add("run me", soon)
    s.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not recorder.calls:
            time.sleep(0.02)
        assert [t.prompt for t in recorder.calls] == ["run me"]
        assert s.list() == []  # one-shot removed after running
    finally:
        s.stop()


def test_loop_reschedules_recurring(tmp_path):
    recorder = _Recorder()
    s = _make_scheduler(tmp_path, execute=recorder)
    soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
    task = s.add("tick", soon, repeat="every 1s")
    s.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(recorder.calls) < 2:
            time.sleep(0.02)
        assert len(recorder.calls) >= 2
        first, second = recorder.calls[0], recorder.calls[1]
        assert datetime.fromisoformat(second.next_run) > datetime.fromisoformat(first.next_run)
        assert [t.id for t in s.list()] == [task.id]  # still scheduled
    finally:
        s.stop()


def test_loop_survives_executor_error(tmp_path):
    def boom(task):
        raise RuntimeError("executor exploded")

    s = _make_scheduler(tmp_path, execute=boom)
    soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
    s.add("boom", soon)
    s.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and s.list():
            time.sleep(0.02)
        assert s.list() == []  # task consumed despite the executor error
        # thread still alive: schedule another and it runs too
        s.add("again", (datetime.now() + timedelta(milliseconds=150)).isoformat())
        time.sleep(0.6)
        assert s.list() == []
    finally:
        s.stop()


def test_cancel_prevents_execution(tmp_path):
    recorder = _Recorder()
    s = _make_scheduler(tmp_path, execute=recorder)
    task = s.add("never", (datetime.now() + timedelta(milliseconds=200)).isoformat())
    assert s.cancel(task.id)
    s.start()
    time.sleep(0.5)
    assert recorder.calls == []
    s.stop()


## disabled scheduler (no config) ##

def test_disabled_tools_return_error(monkeypatch):
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": False})
    assert scheduler_module.schedule_task("x", "2030-01-01 09:00") == \
        scheduler_module._DISABLED_MESSAGE
    assert scheduler_module.list_scheduled_tasks() == \
        scheduler_module._DISABLED_MESSAGE
    assert scheduler_module.cancel_scheduled_task("t-1") == \
        scheduler_module._DISABLED_MESSAGE


## configure() ##

def test_configure_missing_section_disables():
    class _Cfg:  # no [scheduler] attribute
        pass

    scheduler_module.configure(_Cfg())
    assert scheduler_module._CONFIG["enabled"] is False


def test_configure_reads_section():
    from types import SimpleNamespace
    scheduler_module.configure(SimpleNamespace(scheduler=SimpleNamespace(
        enabled=True, persist_file="data/schedule.json", on_missed="run")))
    assert scheduler_module._CONFIG == {
        "enabled": True, "persist_file": "data/schedule.json", "on_missed": "run"}
