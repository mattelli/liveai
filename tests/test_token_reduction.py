"""Unit tests for the token-reduction features (PLAN_IMPROVE_TOKENS.md)."""
import json
from logging import getLogger
from types import SimpleNamespace

import pytest

import src.live_ai as live_ai_mod


# ------------------------------------------------------- tool result capping


def test_truncate_tool_result_unchanged_under_limit():
    assert live_ai_mod._truncate_tool_result("hello", 4000) == "hello"


def test_truncate_tool_result_unchanged_when_disabled():
    text = "x" * 50_000
    assert live_ai_mod._truncate_tool_result(text, 0) == text
    assert live_ai_mod._truncate_tool_result(text, -5) == text


def test_truncate_tool_result_keeps_head_and_tail():
    text = "HEAD" + "m" * 10_000 + "TAIL"
    result = live_ai_mod._truncate_tool_result(text, 4000)
    assert len(result) < len(text)
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert f"[truncated, {len(text)} chars total]" in result


def test_truncate_tool_result_head_tail_split_is_three_quarters():
    text = "A" * 10_000 + "Z" * 10_000
    result = live_ai_mod._truncate_tool_result(text, 4000)
    assert result.startswith("A" * 3000)      # head = limit * 3 // 4
    assert result.endswith("Z" * 1000)        # tail = limit // 4
    assert "A" * 3001 not in result           # middle content dropped


def test_truncate_tool_result_reports_original_length():
    text = "y" * 12_345
    result = live_ai_mod._truncate_tool_result(text, 1000)
    assert "[truncated, 12345 chars total]" in result


class _FakeStream:
    """Minimal StreamPrinter stand-in for bare-agent tests."""

    def __init__(self):
        self.fragments = []

    def feed(self, content):
        self.fragments.append(content)

    def flush(self):
        pass

    def mark_line_clean(self):
        pass

    def note(self, text):
        pass

    def start_spinner(self, note="working"):
        pass

    def stop_spinner(self):
        pass


class _BareAgent(live_ai_mod.Agent):
    """Agent built without __init__ (no SDK connection) for unit tests."""

    def __init__(self):  # noqa: D107 - deliberately bypasses Brain.__init__
        self.logger = getLogger("test")
        self.config = SimpleNamespace(brain=SimpleNamespace(model="m"))
        self._model_name = "m"
        self._cancel_requested = False
        self._active_stream = None
        self._fragments_buffer = []
        self._stream = None


def _agent_with_tool_result(result: str, limit: int) -> _BareAgent:
    agent = _BareAgent()
    agent._messages = [{"role": "system", "content": "sys"}]
    agent._context_window = 4096
    agent._max_tool_result_chars = limit
    agent._turn_started = None
    agent._show_thinking = False
    agent._reset_thinking()
    agent._stream = _FakeStream()
    agent.tool_definitions = []
    agent._tool_impls = {"big_tool": lambda: result}
    rounds = [[{"id": "call_1", "name": "big_tool", "arguments": "{}"}]]

    def fake_stream_round(tools):
        calls = rounds.pop(0) if rounds else []
        return "", calls

    agent._stream_round = fake_stream_round
    agent._stream_completion = lambda tools=None: iter(())
    return agent


def test_act_caps_tool_result_in_history():
    agent = _agent_with_tool_result("R" * 50_000, limit=4000)
    agent.act("do it")
    tool_messages = [m for m in agent._messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert len(content) < 50_000
    assert "[truncated, 50000 chars total]" in content


def test_act_keeps_full_result_when_limit_zero():
    agent = _agent_with_tool_result("R" * 50_000, limit=0)
    agent.act("do it")
    tool_messages = [m for m in agent._messages if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "R" * 50_000


def test_act_monitor_event_still_gets_first_200_chars():
    agent = _agent_with_tool_result("R" * 50_000, limit=4000)
    recorded = []

    class FakeMonitor:
        def record(self, *args, **kwargs):
            recorded.append(kwargs)

    live_ai_mod.bind_monitor(FakeMonitor())
    try:
        import src.live_ai as la
        la._TOOL_CONTEXT.agent_name = "worker"
        agent.act("do it")
    finally:
        la._TOOL_CONTEXT.agent_name = ""
        live_ai_mod.bind_monitor(None)
    results = [kw.get("result", "") for kw in recorded if "result" in kw]
    assert results and results[0].startswith("R" * 200)
    assert len(results[0]) == 200


# -------------------------------------------------------- lean tool schemas


def _default_tool_definitions():
    import types as _types
    dummy = _types.SimpleNamespace(model_info=live_ai_mod.Brain.model_info)
    tools = live_ai_mod._default_tools(dummy)
    return tools, [live_ai_mod.Brain._openai_tool_definition(t) for t in tools]


def test_tool_definitions_contain_no_default_keys():
    _, defs = _default_tool_definitions()
    for definition in defs:
        assert '"default"' not in json.dumps(definition)


def test_tool_definition_preserves_properties_and_required():
    # generate_image has many optional (defaulted) params; agent_activity one.
    tools, _ = _default_tool_definitions()
    by_name = {t.__name__: t for t in tools}
    definition = live_ai_mod.Brain._openai_tool_definition(by_name["generate_image"])
    props = definition["function"]["parameters"]["properties"]
    assert "prompt" in props and props["prompt"]["type"] == "string"
    assert "prompt" in definition["function"]["parameters"]["required"]
    assert "seed" in props  # optional param still documented, no default value

    definition = live_ai_mod.Brain._openai_tool_definition(by_name["agent_activity"])
    assert definition["function"]["parameters"]["required"] == []
    assert "last" in definition["function"]["parameters"]["properties"]


def test_tool_definition_doc_cap_zero_keeps_full_docstring():
    tools, _ = _default_tool_definitions()
    tool = tools[0]
    full_doc = (tool.__doc__ or "").strip()
    definition = live_ai_mod.Brain._openai_tool_definition(tool)
    assert definition["function"]["description"] == full_doc


def test_tool_definition_doc_cap_truncates_descriptions():
    tools, _ = _default_tool_definitions()
    for tool in tools:
        definition = live_ai_mod.Brain._openai_tool_definition(tool, max_doc_chars=300)
        description = definition["function"]["description"]
        assert len(description) <= 300
        assert definition["function"]["name"] == tool.__name__
        assert definition["function"]["parameters"]["type"] == "object"


def test_tool_definitions_payload_budget_with_doc_cap():
    tools, _ = _default_tool_definitions()
    total = sum(
        len(json.dumps(live_ai_mod.Brain._openai_tool_definition(t, max_doc_chars=300)))
        for t in tools
    )
    # Measured baseline: 21_942 (full docs + defaults) -> 19_126 (doc300).
    assert total < 20_000
