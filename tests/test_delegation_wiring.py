"""Wiring tests: delegation tool registration, prompt section, session drain.

No real LLM calls: agent construction is stubbed with the shared
``_fake_openai`` helper.
"""

import textwrap
from types import SimpleNamespace

import pytest

import src.live_ai as live_ai_mod
from src.agent_registry import load_agent_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    load_agent_registry(SimpleNamespace())  # empty registry before AND after
    yield
    load_agent_registry(SimpleNamespace())


def _offline_agent(monkeypatch, tmp_path, agent_overrides=None, agents=None,
                   **kwargs):
    from tests.test_live_ai import _fake_openai, _remote_config
    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _remote_config()
    config.agent = SimpleNamespace(bootstrap_prompt="", **(agent_overrides or {}))
    if agents is not None:
        # The [agents] section lives in the config the agent is built from
        # (production flow: config.toml -> Agent(config)).
        config.agents = SimpleNamespace(**agents)
    return live_ai_mod.Agent(config, **kwargs)


def _register(**agents):
    load_agent_registry(SimpleNamespace(agents=dict(agents)))


# ------------------------------------------------------- tool registration

def test_delegate_task_registered_with_agents(monkeypatch, tmp_path):
    agent = _offline_agent(
        monkeypatch, tmp_path, agents={"researcher": {"tool_groups": "web"}})
    assert "delegate_task" in [t.__name__ for t in agent.tools]


def test_delegate_task_absent_without_agents(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path)
    assert "delegate_task" not in [t.__name__ for t in agent.tools]


def test_delegate_task_absent_for_isolated_agents(monkeypatch, tmp_path):
    """Flat topology: isolated agents never get the delegation tool."""
    _register(researcher={"tool_groups": "web"})
    agent = _offline_agent(monkeypatch, tmp_path, isolated_tools=True,
                           quiet=True)
    assert "delegate_task" not in [t.__name__ for t in agent.tools]


def test_delegate_task_respects_tool_groups_filter(monkeypatch, tmp_path):
    """The tool sits in the 'delegation' topic: filtered out unless listed."""
    _register(researcher={"tool_groups": "web"})
    agent = _offline_agent(
        monkeypatch, tmp_path, agent_overrides={"tool_groups": "web, system"})
    assert "delegate_task" not in [t.__name__ for t in agent.tools]


def test_delegate_tool_schema_sent(monkeypatch, tmp_path):
    agent = _offline_agent(
        monkeypatch, tmp_path, agents={"researcher": {"tool_groups": "web"}})
    names = [d["function"]["name"] for d in agent.tool_definitions]
    assert "delegate_task" in names


def test_isolated_agent_registry_not_clobbered(monkeypatch, tmp_path):
    """A child built later must not empty the main agent's registry."""
    main = _offline_agent(
        monkeypatch, tmp_path, agents={"researcher": {"tool_groups": "web"}})
    child = _offline_agent(monkeypatch, tmp_path, isolated_tools=True,
                           quiet=True)
    assert "delegate_task" not in [t.__name__ for t in child.tools]
    from src.agent_registry import agent_registry_state
    assert agent_registry_state()  # registry still populated


# ---------------------------------------------------- managed-agents prompt

def test_managed_agents_section_in_system_prompt(monkeypatch, tmp_path):
    agent = _offline_agent(
        monkeypatch, tmp_path,
        agents={"researcher": {"tool_groups": "web",
                               "description": "does research"}})
    system = agent.messages[0]["content"]
    assert "delegate_task" in system
    assert "- researcher: does research" in system


def test_managed_agents_section_absent_without_agents(monkeypatch, tmp_path):
    agent = _offline_agent(monkeypatch, tmp_path)
    assert "delegate_task" not in agent.messages[0]["content"]


def test_managed_agents_section_absent_for_children(monkeypatch, tmp_path):
    _register(researcher={"tool_groups": "web"})
    child = _offline_agent(monkeypatch, tmp_path, isolated_tools=True,
                           quiet=True)
    assert "delegate_task" not in child.messages[0]["content"]


# -------------------------------------------------------- session injection

def test_session_injects_delegation_results(monkeypatch, tmp_path):
    from src.session import ConversationSession

    live_ai_mod.queue_agent_result(
        "delegation", "researcher (agent-1)", "research answer")
    live_ai_mod.queue_agent_result(
        "delegation", "writer (agent-2)", "writer answer")

    session = ConversationSession.__new__(ConversationSession)
    session.agent = SimpleNamespace(messages=[])
    session._inject_scheduled_results()
    contents = [m["content"] for m in session.agent.messages]
    assert len(contents) == 2
    assert contents[0] == ("Sub-agent task researcher (agent-1) completed. "
                           "Result:\nresearch answer")
    assert contents[1] == ("Sub-agent task writer (agent-2) completed. "
                           "Result:\nwriter answer")
