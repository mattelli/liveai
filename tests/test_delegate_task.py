"""Unit tests for delegate_task (src.delegation) — no real LLM calls.

Child agents are stubbed via the ``_CHILD_RUNNER`` / ``_AGENT_FACTORY`` seams;
the background-results queue is drained between tests (conftest autouse
fixture) so results never leak across tests.
"""

import threading
from types import SimpleNamespace

import pytest

import src.delegation as delegation_mod
import src.live_ai as live_ai_mod
import src.sub_agents as sub_agents_mod
from src.agent_registry import AgentSpec, load_agent_registry
from src.sub_agents import SubAgentBuildError, build_sub_agent


@pytest.fixture
def wired():
    """Reset module state and wire a fake config; cleans up afterwards."""
    config = SimpleNamespace(
        brain=SimpleNamespace(system_prompt="s", model="m",
                              base_url="https://x", api_key_env="K"),
        agent=SimpleNamespace(tool_groups="all", max_act_rounds=0),
    )
    delegation_mod._APP_CONFIG = config
    delegation_mod.reset_task_ids()
    yield config
    delegation_mod._APP_CONFIG = None
    delegation_mod._CHILD_RUNNER = None
    delegation_mod._SUMMARIZER = None
    sub_agents_mod._AGENT_FACTORY = None
    live_ai_mod.drain_agent_results()  # keep the queue clean between tests


def _register(name: str, **spec_kw):
    load_agent_registry(SimpleNamespace(agents={
        name: {"tool_groups": "web", **spec_kw}}))


class _FakeChild:
    """Child-agent double: canned text (optionally after a delay/failure)."""

    def __init__(self, text="child result", delay=0.0, fail=False):
        self.text = text
        self.delay = delay
        self.fail = fail
        self.tasks = []

    def act(self, task):
        self.tasks.append(task)
        if self.fail:
            raise RuntimeError("child exploded")
        if self.delay:
            done = threading.Event()
            done.wait(self.delay)
        return SimpleNamespace(text=self.text)


def _drain(timeout_s: float = 2.0):
    """Poll the background queue for delegation results (bounded)."""
    for _ in range(int(timeout_s / 0.01)):
        got = live_ai_mod.drain_agent_results_of_kind("delegation")
        if got:
            return got
        threading.Event().wait(0.01)
    return []


# ------------------------------------------------------------------ registry

def test_no_registry_raises(wired):
    load_agent_registry(SimpleNamespace())  # no [agents]
    with pytest.raises(LookupError, match="no sub-agents configured"):
        delegation_mod.delegate_task("researcher", "do things")


def test_unknown_agent_lists_configured(wired):
    _register("researcher")
    with pytest.raises(LookupError) as excinfo:
        delegation_mod.delegate_task("writer", "do things")
    assert "unknown sub-agent 'writer'" in str(excinfo.value)
    assert "researcher" in str(excinfo.value)


def test_empty_agent_name_raises(wired):
    _register("researcher")
    with pytest.raises(LookupError, match="agent_name is required"):
        delegation_mod.delegate_task("", "do things")


def test_empty_task_raises(wired):
    _register("researcher")
    with pytest.raises(LookupError, match="task is required"):
        delegation_mod.delegate_task("researcher", "  ")


# --------------------------------------------------------------- async spawn

def test_delegation_returns_ack_immediately(wired):
    _register("researcher")
    child = _FakeChild(delay=5.0)  # a synchronous call would block 5s here
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: child
    import time
    started = time.monotonic()
    ack = delegation_mod.delegate_task("researcher", "long task")
    elapsed = time.monotonic() - started
    assert "task_id=agent-1" in ack
    assert "'researcher'" in ack
    assert elapsed < 2.0  # returned while the child is still running


def test_child_result_lands_in_queue(wired):
    _register("researcher")
    child = _FakeChild(text="the answer")
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: child
    ack = delegation_mod.delegate_task("researcher", "task one")
    got = _drain()
    assert got == [("researcher (agent-1)", "the answer")]
    assert child.tasks == ["task one"]
    assert delegation_mod.pending_delegations() == 0
    assert "agent-1" in ack


def test_task_ids_increase(wired):
    _register("researcher")
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild()
    ack1 = delegation_mod.delegate_task("researcher", "one")
    ack2 = delegation_mod.delegate_task("researcher", "two")
    assert "agent-1" in ack1 and "agent-2" in ack2


def test_pending_tracked_while_running(wired):
    _register("researcher")
    release = threading.Event()

    def runner(agent, task):
        release.wait(2.0)
        return "late result"

    delegation_mod._CHILD_RUNNER = runner
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild()
    delegation_mod.delegate_task("researcher", "slow")
    deadline = threading.Event()
    for _ in range(200):  # wait until the background thread registered itself
        if delegation_mod.pending_delegations() == 1:
            break
        deadline.wait(0.01)
    assert delegation_mod.pending_delegations() == 1
    release.set()
    _drain()
    assert delegation_mod.pending_delegations() == 0


# ----------------------------------------------------------------- failures

def test_child_failure_queues_error_text(wired):
    _register("researcher")
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(fail=True)
    delegation_mod.delegate_task("researcher", "boom")
    got = _drain()
    assert got and "failed" in got[0][1] and "child exploded" in got[0][1]


def test_empty_child_answer_gets_marker(wired):
    _register("researcher")
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(text="")
    delegation_mod.delegate_task("researcher", "empty")
    got = _drain()
    assert got and "empty answer" in got[0][1]


def test_build_failure_surfaces_immediately(wired):
    _register("researcher")

    def factory(cfg, spec):
        raise SubAgentBuildError("bad model entry")

    sub_agents_mod._AGENT_FACTORY = factory
    with pytest.raises(LookupError, match="bad model entry"):
        delegation_mod.delegate_task("researcher", "task")
    assert delegation_mod.pending_delegations() == 0


# ------------------------------------------------------------- summarization

def _wire_summarize(summarize: bool, max_chars: int):
    load_agent_registry(SimpleNamespace(agents={
        "summarize_results": summarize,
        "summarize_max_chars": max_chars,
        "researcher": {"tool_groups": "web"}}))


def test_long_result_summarized_when_enabled(wired):
    _wire_summarize(summarize=True, max_chars=10)
    delegation_mod._SUMMARIZER = lambda text: f"summary of {len(text)} chars"
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(text="x" * 500)
    delegation_mod.delegate_task("researcher", "task")
    got = _drain()
    assert got == [("researcher (agent-1)", "summary of 500 chars")]


def test_short_result_passes_through(wired):
    _wire_summarize(summarize=True, max_chars=10)
    delegation_mod._SUMMARIZER = lambda text: "SHOULD NOT RUN"
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(text="tiny")
    delegation_mod.delegate_task("researcher", "task")
    got = _drain()
    assert got == [("researcher (agent-1)", "tiny")]


def test_summarizer_failure_keeps_original(wired):
    _wire_summarize(summarize=True, max_chars=10)
    long_text = "y" * 500

    def bad_summarizer(text):
        raise RuntimeError("summary service down")

    delegation_mod._SUMMARIZER = bad_summarizer
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(text=long_text)
    delegation_mod.delegate_task("researcher", "task")
    got = _drain()
    assert got == [("researcher (agent-1)", long_text)]


def test_summarization_disabled_by_default(wired):
    _register("researcher")  # summarize_results defaults to false
    delegation_mod._SUMMARIZER = lambda text: "SHOULD NOT RUN"
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(text="x" * 999)
    delegation_mod.delegate_task("researcher", "task")
    got = _drain()
    assert got == [("researcher (agent-1)", "x" * 999)]


# ------------------------------------------------------------ monitoring

def test_delegated_run_records_start_and_answer(wired):
    """A delegation emits start/answer events under the child's name."""
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    live_ai_mod.bind_monitor(monitor)
    try:
        _register("researcher")
        sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(
            text="the answer")
        delegation_mod.delegate_task("researcher", "task one")
        got = _drain()
        assert got == [("researcher (agent-1)", "the answer")]
        events = monitor.snapshot()
        assert [(e["agent"], e["event"]) for e in events] == [
            ("researcher", "start"), ("researcher", "answer")]
        assert events[0]["prompt"] == "task one"
        assert events[0]["task_id"] == "agent-1"
        assert events[1]["answer"] == "the answer"
    finally:
        live_ai_mod.bind_monitor(None)


def test_delegated_failure_records_answer_with_error(wired):
    """A failing child still records an answer event (with the error text)."""
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    live_ai_mod.bind_monitor(monitor)
    try:
        _register("researcher")
        sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(fail=True)
        delegation_mod.delegate_task("researcher", "task one")
        got = _drain()
        assert "Error: sub-agent 'researcher' failed" in got[0][1]
        events = monitor.snapshot()
        assert [e["event"] for e in events] == ["start", "answer"]
        assert "failed" in events[1]["answer"]
    finally:
        live_ai_mod.bind_monitor(None)


def test_delegated_tool_calls_reported_to_monitor(wired):
    """Tool calls during a delegation flow into the monitor (thread-local)."""
    import src.live_ai as live_ai
    from src.live_ai import AgentMonitor

    class _ToolChild:
        _tool_impls = {"echo": lambda **kw: "echo-ok"}
        tool_definitions = []
        _disabled_tools = set()
        _cancel_requested = False

        def __init__(self):
            self.messages = []

        def act(self, prompt):
            live_ai.Agent._execute_tool(
                self, {"name": "echo", "arguments": "{}"})
            self.messages.append({"role": "assistant", "content": "ok"})
            return SimpleNamespace(text="ok")

    monitor = AgentMonitor(maxlen=10)
    live_ai_mod.bind_monitor(monitor)
    try:
        _register("researcher")
        sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _ToolChild()
        delegation_mod.delegate_task("researcher", "task one")
        got = _drain()
        assert got == [("researcher (agent-1)", "ok")]
        events = monitor.snapshot()
        kinds = [e["event"] for e in events]
        assert kinds == ["start", "tool_call", "tool_result", "answer"]
        assert events[1]["tool"] == "echo"
        assert events[2]["result"] == "echo-ok"
    finally:
        live_ai_mod.bind_monitor(None)


def test_delegation_not_monitored_when_monitor_off(wired):
    """No monitor bound: delegation still completes, no events recorded."""
    _register("researcher")
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(text="ok")
    delegation_mod.delegate_task("researcher", "task one")
    got = _drain()
    assert got == [("researcher (agent-1)", "ok")]


def test_delegated_events_visible_in_agent_activity(wired):
    """agent_activity() reports delegation runs alongside scheduled runs."""
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    monitor.record("scheduler", "start", prompt="check the news")
    live_ai_mod.bind_monitor(monitor)
    try:
        _register("researcher")
        sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: _FakeChild(
            text="research done")
        delegation_mod.delegate_task("researcher", "research X")
        _drain()
        report = live_ai_mod.agent_activity()
        assert "researcher" in report and "research X" in report
        assert "scheduler" in report and "check the news" in report
        assert "research done" in report
    finally:
        live_ai_mod.bind_monitor(None)


# ------------------------------------------------------------ flat topology

def test_child_toolset_never_contains_delegate_task(wired, monkeypatch):
    """Flat topology: children are built without the delegation tool."""
    from tests.test_live_ai import _fake_openai, _remote_config

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    _register("researcher")
    config = _remote_config()
    config.agent = SimpleNamespace(tool_groups="all", max_act_rounds=0,
                                   bootstrap_prompt="")
    delegation_mod.configure_delegation(config)
    # No _AGENT_FACTORY stub: exercise the real builder end-to-end.
    child = build_sub_agent(
        config, AgentSpec(name="researcher", tool_groups="all"))
    names = [getattr(t, "__name__", "") for t in child.tools]
    assert "delegate_task" not in names
