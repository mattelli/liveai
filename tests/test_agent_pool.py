"""Tests for the multi-agent pool (PLAN.md tasks 1-7).

Task 1: ``[agent]`` pool settings resolution (``agent_pool_settings``).
"""

import pathlib
import textwrap
from types import SimpleNamespace

import pytest

import src.live_ai as live_ai_mod  # noqa: E402
from src.common import load_config  # noqa: E402


# ------------------------------------------------------------------ settings


def _agent_config(**agent) -> SimpleNamespace:
    return SimpleNamespace(agent=SimpleNamespace(**agent))


def test_agent_pool_settings_full_config():
    config = _agent_config(
        bootstrap_prompt="b",
        parallel_agents=2,
        agent_models=["glm", "qwen"],
        agent_system_prompts=["research", "code"],
        agent_quiet=True,
    )
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": True,
    }


def test_agent_pool_settings_defaults_when_keys_missing():
    # Only the historical bootstrap_prompt key: single-agent behavior.
    config = _agent_config(bootstrap_prompt="b")
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 0,
        "models": [],
        "system_prompts": [],
        "quiet": True,
    }


def test_agent_pool_settings_without_agent_section():
    assert live_ai_mod.agent_pool_settings(SimpleNamespace()) == {
        "count": 0,
        "models": [],
        "system_prompts": [],
        "quiet": True,
    }


def test_agent_pool_settings_truncates_models_to_count():
    config = _agent_config(parallel_agents=1, agent_models=["a", "b", "c"])
    assert live_ai_mod.agent_pool_settings(config)["models"] == ["a"]


def test_agent_pool_settings_accepts_comma_separated_strings():
    # load_config path-resolves strings inside TOML lists, so a comma-
    # separated string is the robust way to declare several models.
    config = _agent_config(
        parallel_agents=2, agent_models="glm, qwen", agent_system_prompts="research, code"
    )
    assert live_ai_mod.agent_pool_settings(config) == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": True,
    }


def test_agent_pool_settings_pads_missing_system_prompts():
    config = _agent_config(parallel_agents=3, agent_system_prompts=["one"])
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings["system_prompts"] == ["one", None, None]


def test_agent_pool_settings_rejects_bad_count():
    config = _agent_config(parallel_agents="many")
    assert live_ai_mod.agent_pool_settings(config)["count"] == 0
    config = _agent_config(parallel_agents=-2)
    assert live_ai_mod.agent_pool_settings(config)["count"] == 0


def test_agent_pool_settings_quiet_defaults_true_and_is_bool():
    assert live_ai_mod.agent_pool_settings(_agent_config(parallel_agents=1))["quiet"] is True
    config = _agent_config(parallel_agents=1, agent_quiet=False)
    assert live_ai_mod.agent_pool_settings(config)["quiet"] is False


# ------------------------------------------------- load_config round-trip


def test_load_config_round_trips_agent_pool_keys(tmp_path):
    config_file = tmp_path / "config.toml"
    # Comma-separated strings: load_config path-resolves strings inside TOML
    # lists (e.g. "glm" -> WindowsPath), so the string form is the robust one.
    config_file.write_text(textwrap.dedent("""
        [agent]
        bootstrap_prompt = "hello"
        parallel_agents = 2
        agent_models = "glm, qwen"
        agent_system_prompts = "research, code"
        agent_quiet = false
    """), encoding="utf-8")
    config = load_config(config_file)
    settings = live_ai_mod.agent_pool_settings(config)
    assert settings == {
        "count": 2,
        "models": ["glm", "qwen"],
        "system_prompts": ["research", "code"],
        "quiet": False,
    }
