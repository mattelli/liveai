"""Unit tests for scheduler tool registration + config (Task 4 of PLAN.md)."""

import pytest

from src.common import load_config
from src.live_ai import OpenAICore
from src.tools import scheduler as scheduler_module


def test_default_tools_include_scheduler_tools():
    from src.live_ai import _default_tools

    class _DummyBrain:  # only .model_info is accessed from the agent
        def model_info(self):
            return ""

    names = [getattr(t, "__name__", "") for t in _default_tools(_DummyBrain())]
    assert "schedule_task" in names
    assert "list_scheduled_tasks" in names
    assert "cancel_scheduled_task" in names


def test_scheduler_tool_definitions_are_well_formed():
    """OpenAI function-calling definitions: name, description, parameters."""
    from src.live_ai import _default_tools

    class _DummyBrain:
        def model_info(self):
            return ""

    tools = {getattr(t, "__name__", ""): t for t in _default_tools(_DummyBrain())}
    for name in ("schedule_task", "list_scheduled_tasks", "cancel_scheduled_task"):
        definition = OpenAICore._openai_tool_definition(tools[name])
        function = definition["function"]
        assert function["name"] == name
        assert function["description"].strip()  # docstring carried over
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        # schedule_task/cancel_scheduled_task take required arguments;
        # list_scheduled_tasks is parameterless (empty required/properties
        # is correct for OpenAI function calling).
        if name != "list_scheduled_tasks":
            assert parameters["required"], f"{name} must have required parameters"
            assert parameters["properties"]


def test_schedule_task_definition_parameters():
    from src.live_ai import _default_tools

    class _DummyBrain:
        def model_info(self):
            return ""

    tools = {getattr(t, "__name__", ""): t for t in _default_tools(_DummyBrain())}
    definition = OpenAICore._openai_tool_definition(tools["schedule_task"])
    properties = definition["function"]["parameters"]["properties"]
    assert "prompt" in properties and "at" in properties
    assert "repeat" in properties  # optional (has a default)


def test_config_parses_scheduler_section(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[scheduler]\n"
        'enabled = true\n'
        'persist_file = "data/schedule.json"\n'
        'on_missed = "run"\n',
        encoding="utf-8")
    config = load_config(config_file)
    section = config.scheduler
    assert section.enabled is True
    assert str(section.persist_file).endswith("schedule.json")  # path resolved
    assert section.on_missed == "run"


def test_config_without_scheduler_section_has_no_attribute(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[brain]\nmodel = 'x'\n", encoding="utf-8")
    config = load_config(config_file)
    assert not hasattr(config, "scheduler")  # configure() then disables


def test_configure_with_real_config_shape(tmp_path):
    """End-to-end config injection: [scheduler] enabled -> tools work."""
    from types import SimpleNamespace

    config = SimpleNamespace(scheduler=SimpleNamespace(
        enabled=True, persist_file=str(tmp_path / "schedule.json"),
        on_missed="skip"))
    scheduler_module.configure(config)
    try:
        assert scheduler_module._enabled() is True
        # A tool call goes through the factory seam: with a fake scheduler
        # bound the tools operate normally (no disabled error).
        scheduler_module.bind_scheduler(
            type("S", (), {
                "add": lambda self, p, a, r="": type("T", (), {
                    "id": "t-1", "prompt": p, "next_run": a, "repeat": r})(),
                "cancel": lambda self, tid: tid == "t-1",
                "list": lambda self: [],
            })())
        result = scheduler_module.schedule_task("hello", "2030-01-01 09:00")
        assert result.startswith("Scheduled task t-1")
    finally:
        scheduler_module.configure(SimpleNamespace())  # restore disabled state
