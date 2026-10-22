"""Unit tests for the scheduler agent tools (Task 2 of PLAN.md).

The tools are exercised through the module-level API with the
``_SCHEDULER_FACTORY`` seam (fake scheduler) — no LLM, no real threads.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.tools import scheduler as scheduler_module
from src.tools.scheduler import ScheduledTask, SchedulerTools


class _FakeScheduler:
    """In-memory scheduler double recording the tool calls."""

    def __init__(self):
        self.tasks: dict[str, ScheduledTask] = {}
        self.next_id = 1
        self.cancelled: list[str] = []

    def add(self, prompt, at, repeat=""):
        task = ScheduledTask(id=f"t-{self.next_id}", prompt=prompt,
                             next_run=at, repeat=repeat)
        self.next_id += 1
        self.tasks[task.id] = task
        return task

    def cancel(self, task_id):
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.cancelled.append(task_id)
            return True
        return False

    def list(self):
        return sorted(self.tasks.values(), key=lambda t: t.next_run)


@pytest.fixture
def fake_scheduler(monkeypatch):
    """Enable the scheduler and inject the fake via the factory seam."""
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": True})
    fake = _FakeScheduler()
    monkeypatch.setattr(scheduler_module, "_SCHEDULER_FACTORY", lambda: fake)
    monkeypatch.setattr(scheduler_module, "_SCHEDULER", None)  # fresh default
    return fake


def test_schedule_task_returns_confirmation_with_id(fake_scheduler):
    result = scheduler_module.schedule_task("do it", "2030-01-01 09:00")
    assert "t-1" in result
    assert "Scheduled task" in result
    assert "do it" in result
    assert "one-shot" in result


def test_schedule_task_recurring_label(fake_scheduler):
    result = scheduler_module.schedule_task("tick", "2030-01-01 09:00",
                                            repeat="daily")
    assert "daily" in result


def test_schedule_task_bad_at_returns_error(fake_scheduler):
    result = scheduler_module.schedule_task("x", "not-a-date")
    assert result.startswith("Error:")


def test_schedule_task_past_one_shot_returns_error(fake_scheduler):
    result = scheduler_module.schedule_task("x", "2000-01-01 09:00")
    assert result.startswith("Error:")


def test_list_scheduled_tasks_shows_added(fake_scheduler):
    scheduler_module.schedule_task("write report", "2030-01-01 09:00")
    listing = scheduler_module.list_scheduled_tasks()
    assert "t-1" in listing
    assert "write report" in listing
    assert "one-shot" in listing


def test_list_scheduled_tasks_empty(fake_scheduler):
    assert scheduler_module.list_scheduled_tasks() == "No scheduled tasks."


def test_cancel_scheduled_task_existing(fake_scheduler):
    scheduler_module.schedule_task("x", "2030-01-01 09:00")
    assert scheduler_module.cancel_scheduled_task("t-1") == "Cancelled task t-1"
    assert scheduler_module.list_scheduled_tasks() == "No scheduled tasks."


def test_cancel_scheduled_task_unknown(fake_scheduler):
    result = scheduler_module.cancel_scheduled_task("t-99")
    assert result.startswith("Error:")
    assert "t-99" in result


def test_tools_honor_scheduler_factory_seam(fake_scheduler):
    # The factory seam produced the fake: the module-default instance holds it.
    scheduler_module.schedule_task("do it", "2030-01-01 09:00")
    assert scheduler_module._default._scheduler is fake_scheduler


def test_tools_never_raise(fake_scheduler):
    # Even with a scheduler whose methods explode, tools return error strings.
    class _Boom(_FakeScheduler):
        def add(self, *a, **k):
            raise RuntimeError("boom")

    monkey_scheduler = _Boom()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(scheduler_module, "_SCHEDULER_FACTORY",
                            lambda: monkey_scheduler)
        monkeypatch.setattr(scheduler_module, "_SCHEDULER", None)
        result = scheduler_module.schedule_task("x", "2030-01-01 09:00")
        assert result.startswith("Error:")
    finally:
        monkeypatch.undo()


def test_per_owner_isolation():
    """for_owner() hands out fresh instances (pooled-agent isolation)."""
    tools = SchedulerTools()
    assert tools.for_owner(object()) is not tools


def test_scheduler_tools_disabled_without_config(monkeypatch):
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": False})
    tools = SchedulerTools()
    assert tools.schedule_task("x", "2030-01-01 09:00").startswith("Error:")
    assert tools.list_scheduled_tasks().startswith("Error:")


def test_scheduler_tools_use_bound_scheduler(monkeypatch):
    """bind_scheduler() bypasses the factory (main.py wiring path)."""
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": True})
    fake = _FakeScheduler()
    tools = SchedulerTools()
    tools.bind_scheduler(fake)
    assert "t-1" in tools.schedule_task("bound", "2030-01-01 09:00")
    assert fake.tasks["t-1"].prompt == "bound"
