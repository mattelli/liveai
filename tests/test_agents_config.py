"""Unit tests for the [agents] registry parsing (src.agent_registry)."""

import textwrap
from pathlib import Path

import pytest

from src.agent_registry import (
    AgentSpec, agent_registry, load_agent_registry, summarize_settings,
)
from src.common import load_config


def _config_with(toml: str, tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent(toml), encoding="utf-8")
    return load_config(config_file)


def test_valid_registry(tmp_path):
    config = _config_with("""
        [agents.researcher]
        tool_groups = "web, files"
        model = "fast"
        system_prompt = "research things"
        max_act_rounds = 50
        description = "does research"

        [agents.writer]
        tool_groups = "files, memory"
    """, tmp_path)
    specs = load_agent_registry(config)
    assert set(specs) == {"researcher", "writer"}
    researcher = specs["researcher"]
    assert isinstance(researcher, AgentSpec)
    assert researcher.tool_groups == "web, files"
    assert researcher.model == "fast"
    assert researcher.system_prompt == "research things"
    assert researcher.max_act_rounds == 50
    assert researcher.description == "does research"
    assert specs["writer"].tool_groups == "files, memory"
    assert specs["writer"].model == ""
    assert agent_registry() == specs  # module-level registry populated


def test_shorthand_string_entry(tmp_path):
    config = _config_with("""
        [agents]
        researcher = "web, files"
    """, tmp_path)
    specs = load_agent_registry(config)
    assert specs["researcher"].tool_groups == "web, files"
    assert specs["researcher"].max_act_rounds == 0


def test_missing_section_disables_feature(tmp_path):
    config = _config_with("""
        [brain]
        system_prompt = "s"
    """, tmp_path)
    specs = load_agent_registry(config)
    assert specs == {}
    assert agent_registry() == {}


def test_empty_section_disables_feature(tmp_path):
    config = _config_with("""
        [agents]
    """, tmp_path)
    assert load_agent_registry(config) == {}


def test_unknown_keys_logged_and_ignored(tmp_path, caplog):
    config = _config_with("""
        [agents.researcher]
        tool_groups = "web"
        typo_key = "oops"
    """, tmp_path)
    specs = load_agent_registry(config)
    assert specs["researcher"].tool_groups == "web"
    assert "typo_key" in caplog.text


def test_case_variations_are_distinct_entries(tmp_path):
    # TOML table names are case-sensitive, so "Researcher" and "researcher"
    # are two entries. (A true duplicate can never reach the loader: Python
    # dicts collapse duplicate keys before we see them.)
    config = _config_with("""
        [agents.Researcher]
        tool_groups = "web"

        [agents.researcher]
        tool_groups = "files"
    """, tmp_path)
    specs = load_agent_registry(config)
    assert set(specs) == {"Researcher", "researcher"}
    assert specs["Researcher"].tool_groups == "web"
    assert specs["researcher"].tool_groups == "files"


def test_invalid_max_act_rounds_falls_back(tmp_path, caplog):
    config = _config_with("""
        [agents.researcher]
        tool_groups = "web"
        max_act_rounds = "lots"
    """, tmp_path)
    specs = load_agent_registry(config)
    assert specs["researcher"].max_act_rounds == 0
    assert "max_act_rounds" in caplog.text


def test_non_table_entry_ignored(tmp_path, caplog):
    config = _config_with("""
        [agents]
        researcher = 42
    """, tmp_path)
    assert load_agent_registry(config) == {}
    assert "not a table" in caplog.text


def test_shared_flags_parsed(tmp_path):
    config = _config_with("""
        [agents]
        summarize_results = true
        summarize_max_chars = 500

        [agents.researcher]
        tool_groups = "web"
    """, tmp_path)
    load_agent_registry(config)
    assert summarize_settings() == (True, 500)


def test_shared_flags_defaults(tmp_path):
    config = _config_with("""
        [agents.researcher]
        tool_groups = "web"
    """, tmp_path)
    load_agent_registry(config)
    assert summarize_settings() == (False, 800)


def test_invalid_summarize_max_chars_falls_back(tmp_path):
    config = _config_with("""
        [agents]
        summarize_max_chars = "many"

        [agents.researcher]
        tool_groups = "web"
    """, tmp_path)
    load_agent_registry(config)
    assert summarize_settings() == (False, 800)


def test_empty_registry_resets_module_state(tmp_path):
    config = _config_with("""
        [agents.researcher]
        tool_groups = "web"
    """, tmp_path)
    load_agent_registry(config)
    assert agent_registry()
    config2 = _config_with("""
        [brain]
        system_prompt = "s"
    """, tmp_path)
    load_agent_registry(config2)
    assert agent_registry() == {}  # cleared when the section disappears
