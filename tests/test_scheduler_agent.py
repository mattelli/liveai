"""Unit tests for the scheduler execution-agent wiring (Task 3 of PLAN.md).

The scheduler runs due prompts on a dedicated isolated agent; tests inject a
fake agent via the ``_AGENT_FACTORY`` seam — no LLM, no real model calls.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.tools import scheduler as scheduler_module
from src.tools.scheduler import Scheduler


class _FakeAgent:
    """Execution-agent double recording the prompts it was asked to run."""

    def __init__(self):
        self.prompts: list[str] = []

    def act(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"done: {prompt}"


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    """Fresh scheduler module state + fake agent factory; returns the fake."""
    monkeypatch.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
    agent = _FakeAgent()
    monkeypatch.setattr(scheduler_module, "_AGENT_FACTORY", lambda: agent)
    yield agent
    scheduler_module.reset_executor_agent()


def _scheduler(tmp_path, execute=None) -> Scheduler:
    return Scheduler(persist_file=tmp_path / "schedule.json",
                     on_missed="skip", _execute=execute)


def test_due_one_shot_runs_on_agent(tmp_path, agent_env):
    ran = []
    s = _scheduler(tmp_path, execute=scheduler_module._agent_executor)
    soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
    s.add("remind me", soon)
    s.start()
    try:
        deadline = datetime.now() + timedelta(seconds=5)
        while datetime.now() < deadline and not agent_env.prompts:
            import time
            time.sleep(0.02)
        assert agent_env.prompts == ["remind me"]  # exactly one act() call
        assert ran == []
        assert s.list() == []  # one-shot consumed
    finally:
        s.stop()


def test_agent_is_built_once_and_reused(tmp_path, agent_env):
    builds = []
    factory_calls = []

    def counting_factory():
        factory_calls.append(1)
        return agent_env

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", counting_factory)
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        s = _scheduler(tmp_path, execute=scheduler_module._agent_executor)
        soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
        s.add("first", soon)
        s.add("second", (datetime.now() + timedelta(milliseconds=400)).isoformat())
        s.start()
        import time
        deadline = datetime.now() + timedelta(seconds=5)
        while datetime.now() < deadline and len(agent_env.prompts) < 2:
            time.sleep(0.02)
        assert agent_env.prompts == ["first", "second"]
        assert len(factory_calls) == 1  # agent built lazily exactly once
    finally:
        monkey.undo()
        s.stop()


def test_recurring_task_runs_repeatedly_on_agent(tmp_path, agent_env):
    s = _scheduler(tmp_path, execute=scheduler_module._agent_executor)
    soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
    s.add("tick", soon, repeat="every 1s")
    s.start()
    try:
        import time
        deadline = datetime.now() + timedelta(seconds=5)
        while datetime.now() < deadline and len(agent_env.prompts) < 2:
            time.sleep(0.02)
        assert agent_env.prompts == ["tick", "tick"]
        assert [t.id for t in s.list()]  # still scheduled
    finally:
        s.stop()


def test_agent_error_is_caught_and_thread_survives(tmp_path, agent_env, caplog):
    class _ExplodingAgent:
        def act(self, prompt):
            raise RuntimeError("model endpoint down")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", lambda: _ExplodingAgent())
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        s = _scheduler(tmp_path, execute=scheduler_module._agent_executor)
        soon = (datetime.now() + timedelta(milliseconds=150)).isoformat()
        s.add("boom", soon)
        s.start()
        import time
        with caplog.at_level(logging.ERROR, logger="src.tools.scheduler.Scheduler"):
            deadline = datetime.now() + timedelta(seconds=5)
            while datetime.now() < deadline and s.list():
                time.sleep(0.02)
        assert s.list() == []  # task consumed despite the agent error
        assert "boom" in caplog.text or "failed" in caplog.text
        # thread still alive: another task runs fine on a fresh agent
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", lambda: _FakeAgent())
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        s.add("again", (datetime.now() + timedelta(milliseconds=150)).isoformat())
        deadline = datetime.now() + timedelta(seconds=5)
        while datetime.now() < deadline and s.list():
            time.sleep(0.02)
        assert s.list() == []
    finally:
        monkey.undo()
        s.stop()


def test_no_agent_available_logs_prompt_only(tmp_path, caplog):
    """Missing config / failed agent build: the executor logs and returns."""
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(scheduler_module, "_AGENT_FACTORY", lambda: None)
        monkey.setattr(scheduler_module, "_EXECUTOR_AGENT", None)
        monkey.setattr(scheduler_module, "_CONFIG", {"enabled": True})  # no "config"
        with caplog.at_level(logging.WARNING, logger="src.tools.scheduler"):
            scheduler_module._agent_executor(
                type("T", (), {"id": "t-1", "prompt": "logged only"})())
        assert "logged only" in caplog.text
    finally:
        monkey.undo()


def test_default_agent_factory_without_config_returns_none(monkeypatch):
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": True})
    assert scheduler_module._default_agent_factory() is None


def test_default_agent_factory_builds_isolated_agent(monkeypatch):
    """With a config present, the factory builds Agent(config, isolated, quiet)."""
    captured = {}

    class _FakeAgentModule:
        class Agent:  # noqa: N801 - minimal stand-in
            def __init__(self, config, isolated_tools=False, quiet=False):
                captured["config"] = config
                captured["isolated_tools"] = isolated_tools
                captured["quiet"] = quiet

    import types
    fake_module = types.SimpleNamespace(Agent=_FakeAgentModule.Agent)
    monkeypatch.setitem(__import__("sys").modules, "src.live_ai", fake_module)
    config = object()
    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": True,
                                                      "config": config})
    agent = scheduler_module._default_agent_factory()
    assert agent is not None
    assert captured == {"config": config, "isolated_tools": True, "quiet": True}


def test_main_setup_scheduler_starts_and_stops(monkeypatch):
    """_setup_scheduler starts the engine when enabled; _stop_scheduler stops it."""
    import main

    started, stopped = [], []

    class _FakeScheduler:
        def start(self):
            started.append(True)

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": True})
    monkeypatch.setattr(scheduler_module, "_SCHEDULER", None)
    monkeypatch.setattr(scheduler_module, "_default_scheduler_factory",
                        lambda: _FakeScheduler())
    monkeypatch.setattr(scheduler_module, "_SCHEDULER", None)

    main._setup_scheduler(object())  # config shape is irrelevant here
    assert started == [True]
    assert scheduler_module._SCHEDULER is not None

    main._stop_scheduler()
    assert stopped == [True]


def test_main_setup_scheduler_noop_when_disabled(monkeypatch):
    import main

    monkeypatch.setattr(scheduler_module, "_CONFIG", {"enabled": False})
    monkeypatch.setattr(scheduler_module, "_SCHEDULER", None)
    main._setup_scheduler(object())
    assert scheduler_module._SCHEDULER is None  # nothing was created/started
    main._stop_scheduler()  # must be a safe no-op
