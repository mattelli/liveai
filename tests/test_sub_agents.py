"""Unit tests for the sub-agent factory (src.sub_agents)."""

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.sub_agents as sub_agents_mod
from src.agent_registry import AgentSpec
from src.sub_agents import SubAgentBuildError, build_sub_agent, child_config


def _config(tmp_path: Path, models: dict | None = None) -> SimpleNamespace:
    models_file = None
    if models is not None:
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps({"models": models}), encoding="utf-8")
    return SimpleNamespace(
        brain=SimpleNamespace(
            system_prompt="main prompt",
            model="main-model",
            base_url="https://api.test/v1",
            api_key_env="TEST_API_KEY",
            models_file=str(models_file) if models_file else None,
            temperature=0.6,
        ),
        agent=SimpleNamespace(
            tool_groups="all", max_act_rounds=200, bootstrap_prompt="",
        ),
    )


# ------------------------------------------------------------- child_config

def test_child_config_overrides_tool_groups_and_rounds(tmp_path):
    config = _config(tmp_path)
    spec = AgentSpec(name="r", tool_groups="web, files", max_act_rounds=50)
    child = child_config(config, spec)
    assert child.agent.tool_groups == "web, files"
    assert child.agent.max_act_rounds == 50
    # untouched settings survive the copy
    assert child.brain.system_prompt == "main prompt"
    assert child.brain.temperature == 0.6


def test_child_config_keeps_shared_settings_for_siblings(tmp_path):
    """The original config must not be mutated by building a child."""
    config = _config(tmp_path)
    spec = AgentSpec(name="r", tool_groups="web", max_act_rounds=7,
                     system_prompt="child prompt")
    child = child_config(config, spec)
    assert child.agent.tool_groups == "web"
    assert child.brain.system_prompt == "child prompt"
    # parent untouched:
    assert config.agent.tool_groups == "all"
    assert config.agent.max_act_rounds == 200
    assert config.brain.system_prompt == "main prompt"


def test_child_config_zero_rounds_keeps_parent_value(tmp_path):
    config = _config(tmp_path)
    spec = AgentSpec(name="r", tool_groups="all", max_act_rounds=0)
    child = child_config(config, spec)
    assert child.agent.max_act_rounds == 200  # parent value kept


def test_child_config_missing_sections_tolerated(tmp_path):
    config = SimpleNamespace()  # no [brain], no [agent]
    spec = AgentSpec(name="r", tool_groups="web")
    child = child_config(config, spec)
    assert child.agent.tool_groups == "web"


# -------------------------------------------------------------- _models_entry

def test_models_entry_lookup_case_insensitive(tmp_path):
    config = _config(tmp_path, models=[{"name": "Fast", "model": "m-fast",
                                        "api_key_env": "TEST_API_KEY"}])
    spec = AgentSpec(name="r", model="fast")
    entry = sub_agents_mod._models_entry(config, spec)
    assert entry["model"] == "m-fast"


def test_models_entry_none_without_model(tmp_path):
    config = _config(tmp_path)
    assert sub_agents_mod._models_entry(
        config, AgentSpec(name="r")) is None


def test_models_entry_unknown_raises(tmp_path):
    config = _config(tmp_path, models=[{"name": "Fast", "model": "m"}])
    with pytest.raises(SubAgentBuildError, match="not found"):
        sub_agents_mod._models_entry(config, AgentSpec(name="r", model="nope"))


def test_models_entry_without_models_file_raises(tmp_path):
    config = _config(tmp_path)  # models_file=None
    with pytest.raises(SubAgentBuildError, match="models_file"):
        sub_agents_mod._models_entry(config, AgentSpec(name="r", model="fast"))


def test_models_entry_bare_string_shape(tmp_path):
    config = _config(tmp_path, models=["fast-model"])
    entry = sub_agents_mod._models_entry(config, AgentSpec(name="r", model="fast-model"))
    assert entry["model"] == "fast-model"


# ------------------------------------------------------------- build_sub_agent

def test_build_sub_agent_uses_factory_seam(tmp_path):
    config = _config(tmp_path)
    sentinel = object()
    seen = []

    def factory(cfg, spec):
        seen.append((cfg, spec))
        return sentinel

    sub_agents_mod._AGENT_FACTORY = factory
    try:
        spec = AgentSpec(name="r", tool_groups="web")
        assert build_sub_agent(config, spec) is sentinel
        assert seen and seen[0][1] is spec
    finally:
        sub_agents_mod._AGENT_FACTORY = None


def test_build_sub_agent_factory_error_wrapped(tmp_path):
    config = _config(tmp_path)

    def factory(cfg, spec):
        raise RuntimeError("boom")

    sub_agents_mod._AGENT_FACTORY = factory
    try:
        with pytest.raises(SubAgentBuildError, match="boom"):
            build_sub_agent(config, AgentSpec(name="r"))
    finally:
        sub_agents_mod._AGENT_FACTORY = None


def test_build_sub_agent_factory_none_raises(tmp_path):
    config = _config(tmp_path)
    sub_agents_mod._AGENT_FACTORY = lambda cfg, spec: None
    try:
        with pytest.raises(SubAgentBuildError):
            build_sub_agent(config, AgentSpec(name="r"))
    finally:
        sub_agents_mod._AGENT_FACTORY = None


def test_build_sub_agent_real_agent_offline(tmp_path, monkeypatch):
    """A real isolated, quiet Agent is built (openai client stubbed)."""
    from tests.test_live_ai import _fake_openai, _remote_config

    _fake_openai(monkeypatch)
    monkeypatch.setenv("TEST_API_KEY", "k")
    config = _remote_config()
    config.agent = SimpleNamespace(
        tool_groups="model, system", max_act_rounds=0, bootstrap_prompt="")
    agent = build_sub_agent(config, AgentSpec(name="r", tool_groups="model, system"))
    assert agent.isolated_tools is True
    assert agent.quiet is True
    names = [getattr(t, "__name__", "") for t in agent.tools]
    assert "model_info" in names      # topic "model"
    assert "system_info" in names     # topic "system"
    assert "web_search" not in names  # topic "web" excluded
    assert "delegate_task" not in names  # flat topology: no delegation tool
