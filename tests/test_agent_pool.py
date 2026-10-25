"""Tests for the agent-pool removal (PLAN_REMOVE_AGENT_POOL.md).

The pool is gone: these tests verify the delegation tools, the AgentPool
class and the pool settings no longer exist."""

import pytest

import src.live_ai as live_ai_mod  # noqa: E402

# ------------------------------------------- delegation tools removed


def test_delegation_tools_removed():
    """ask_agent / run_parallel_agents are gone; agent_activity stays."""
    import src.live_ai as live_ai_mod

    assert not hasattr(live_ai_mod, "ask_agent")
    assert not hasattr(live_ai_mod, "run_parallel_agents")
    assert not hasattr(live_ai_mod, "bind_agent_pool")
    assert not hasattr(live_ai_mod, "_POOL")
    assert hasattr(live_ai_mod, "agent_activity")  # kept for the scheduler

    # ... and from the default tool set of a fresh agent.
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "ask_agent" not in names
    assert "run_parallel_agents" not in names
    assert "agent_activity" in names


def test_agent_pool_class_removed():
    """The AgentPool class no longer exists in src.live_ai."""
    assert not hasattr(live_ai_mod, "AgentPool")
    with pytest.raises(ImportError):
        from src.live_ai import AgentPool  # noqa: F401


