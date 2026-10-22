"""End-to-end scheduler smoke tests (Task 5 of PLAN.md).

Full pipeline with a fake execution agent (no LLM): schedule -> background
thread wakes up -> prompt runs on the agent -> task list updates. Marked
``integration`` like the existing integration tests.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.tools import scheduler as scheduler_module
from src.tools.scheduler import Scheduler


class _FakeAgent:
    def __init__(self):
        self.prompts: list[str] = []

    def act(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"ok: {prompt}"


@pytest.fixture(autouse=True)
def _enabled_scheduler(monkeypatch):
    """Enable the scheduler for these tests; restore the disabled state after."""
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": True})
    yield
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": False})


def test_e2e_one_shot_runs_and_list_empties(tmp_path):
    agent = _FakeAgent()
    s = Scheduler(persist_file=tmp_path / "schedule.json", on_missed="skip",
                  _execute=scheduler_module._agent_executor)
    scheduler_module.reset_executor_agent()
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", lambda: agent)
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        # The module-level tools lazily build the engine via the factory seam:
        # point it at the test scheduler so the tool and the loop share it.
        monkey.setattr(scheduler_module, "_SCHEDULER_FACTORY", lambda: s)
        monkey.setattr(scheduler_module, "_SCHEDULER", None)
        soon = (datetime.now() + timedelta(seconds=1)).isoformat()
        result = scheduler_module.schedule_task("e2e reminder", soon)
        assert result.startswith("Scheduled task")
        s.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not agent.prompts:
            time.sleep(0.05)
        assert agent.prompts == ["e2e reminder"]
        assert s.list() == []  # task list empty afterwards
    finally:
        monkey.undo()
        scheduler_module.reset_executor_agent()
        s.stop()


def test_e2e_cancel_path_never_executes(tmp_path):
    agent = _FakeAgent()
    s = Scheduler(persist_file=tmp_path / "schedule.json", on_missed="skip",
                  _execute=scheduler_module._agent_executor)
    scheduler_module.reset_executor_agent()
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", lambda: agent)
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        monkey.setattr(scheduler_module, "_SCHEDULER_FACTORY", lambda: s)
        monkey.setattr(scheduler_module, "_SCHEDULER", None)
        ahead = (datetime.now() + timedelta(hours=1)).isoformat()
        scheduler_module.schedule_task("must never run", ahead)
        task_id = scheduler_module.list_scheduled_tasks().split(" |")[0]
        assert scheduler_module.cancel_scheduled_task(task_id).startswith("Cancelled")
        s.start()
        time.sleep(0.5)  # give the loop a chance to (wrongly) fire
        assert agent.prompts == []  # cancelled task never executed
        assert s.list() == []  # task list empty afterwards
    finally:
        monkey.undo()
        scheduler_module.reset_executor_agent()
        s.stop()


def test_e2e_recurring_survives_restart(tmp_path):
    """A recurring task persists and keeps firing after a fresh Scheduler."""
    agent = _FakeAgent()
    file = tmp_path / "schedule.json"
    s1 = Scheduler(persist_file=file, on_missed="skip",
                   _execute=scheduler_module._agent_executor)
    scheduler_module.reset_executor_agent()
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", lambda: agent)
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        soon = (datetime.now() + timedelta(milliseconds=200)).isoformat()
        s1.add("heartbeat", soon, repeat="every 1s")
        s1.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not agent.prompts:
            time.sleep(0.05)
        assert agent.prompts == ["heartbeat"]
        s1.stop()

        # Restart on the same persistence file: the task is still scheduled.
        s2 = Scheduler(persist_file=file, on_missed="skip",
                       _execute=scheduler_module._agent_executor)
        assert [t.id for t in s2.list()] == ["t-1"]
        s2.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(agent.prompts) < 2:
            time.sleep(0.05)
        assert agent.prompts == ["heartbeat", "heartbeat"]
        s2.stop()
    finally:
        monkey.undo()
        scheduler_module.reset_executor_agent()
        s1.stop()
        s2.stop() if "s2" in dir() else None
