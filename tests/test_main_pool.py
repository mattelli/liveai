"""Tests for main.py agent-pool wiring (_setup_agent_pool)."""
import types
import pytest


class _FakeAgent:
    """Minimal Agent stand-in: records construction, no network."""

    instances = []

    def __init__(self, config):
        self.config = config
        _FakeAgent.instances.append(self)


@pytest.fixture
def pool_env(monkeypatch):
    """Patch Agent + bind_agent_pool; return the captured bindings."""
    import src.live_ai as live_ai

    _FakeAgent.instances = []
    captured = []
    monkeypatch.setattr(live_ai, "Agent", _FakeAgent)
    monkeypatch.setattr(
        live_ai, "AgentPool",
        lambda config, models: types.SimpleNamespace(config=config, models=models))
    monkeypatch.setattr(
        live_ai, "bind_agent_pool",
        lambda pool: captured.append(pool))
    return captured


def _config(parallel_agents, models):
    return types.SimpleNamespace(agent=types.SimpleNamespace(
        parallel_agents=parallel_agents, agent_models=models))


def test_noop_when_parallel_agents_missing(pool_env):
    from main import _setup_agent_pool

    cfg = types.SimpleNamespace(agent=types.SimpleNamespace())  # no key at all
    _setup_agent_pool(cfg)
    assert pool_env == []  # nothing bound, single-agent setup untouched


def test_noop_when_zero(pool_env):
    from main import _setup_agent_pool

    _setup_agent_pool(_config(0, ["a", "b"]))
    assert pool_env == []


def test_builds_pool_from_config(pool_env):
    from main import _setup_agent_pool

    _setup_agent_pool(_config(2, ["glm-5.3-flash", "qwen2.5-coder", "extra"]))
    assert len(pool_env) == 1
    pool = pool_env[0]
    assert pool.models == ["glm-5.3-flash", "qwen2.5-coder"]  # truncated to n
    assert pool.config is not None


def test_raises_when_fewer_models_than_agents(pool_env):
    from main import _setup_agent_pool

    with pytest.raises(ValueError, match="agent_models has only 1"):
        _setup_agent_pool(_config(2, ["glm-5.3-flash"]))
    assert pool_env == []  # nothing bound on failure


def test_exact_count_ok(pool_env):
    from main import _setup_agent_pool

    _setup_agent_pool(_config(3, ["a", "b", "c"]))
    assert pool_env[0].models == ["a", "b", "c"]
