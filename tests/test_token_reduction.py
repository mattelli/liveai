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


# ------------------------------------------------------- tool group filtering


def _bare_build_agent(tool_groups):
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    agent.config = SimpleNamespace(
        agent=SimpleNamespace(tool_groups=tool_groups),
        brain=SimpleNamespace(model="m"),
    )
    agent.isolated_tools = False
    return agent


def test_select_tools_all_and_empty_keep_everything():
    tools = live_ai_mod._default_tools(
        SimpleNamespace(model_info=live_ai_mod.Brain.model_info))
    assert live_ai_mod._select_tools(tools, None) == tools
    assert live_ai_mod._select_tools(tools, "all") == tools
    assert live_ai_mod._select_tools(tools, "") == tools
    assert live_ai_mod._select_tools(tools, []) == tools


def test_select_tools_keeps_only_listed_topics():
    tools = live_ai_mod._default_tools(
        SimpleNamespace(model_info=live_ai_mod.Brain.model_info))
    selected = live_ai_mod._select_tools(tools, "files, web")
    names = [t.__name__ for t in selected]
    assert "read_text_file" in names and "web_search" in names
    assert "fetch_financial_data" not in names
    assert "recall_memory" not in names
    assert len(selected) < len(tools)


def test_select_tools_list_and_string_forms_equivalent():
    tools = live_ai_mod._default_tools(
        SimpleNamespace(model_info=live_ai_mod.Brain.model_info))
    as_string = live_ai_mod._select_tools(tools, "finance, images")
    as_list = live_ai_mod._select_tools(tools, ["finance", "images"])
    assert [t.__name__ for t in as_string] == [t.__name__ for t in as_list]
    names = [t.__name__ for t in as_string]
    assert "fetch_financial_data" in names and "generate_image" in names
    assert "run_command" not in names


def test_select_tools_unknown_group_ignored_without_crash():
    tools = live_ai_mod._default_tools(
        SimpleNamespace(model_info=live_ai_mod.Brain.model_info))
    selected = live_ai_mod._select_tools(tools, "files, banana")
    names = [t.__name__ for t in selected]
    assert "read_text_file" in names  # known group still honored
    assert "web_search" not in names


def test_agent_build_tools_honors_tool_groups():
    names = [t.__name__ for t in _bare_build_agent("finance")._build_tools()]
    assert "fetch_financial_data" in names
    assert "read_text_file" not in names


def test_agent_build_tools_default_keeps_all_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    agent.config = SimpleNamespace(brain=SimpleNamespace(model="m"))
    agent.isolated_tools = False
    names = [t.__name__ for t in agent._build_tools()]
    assert len(names) == 44
    assert "summarize_doc" in names and "exit_gracefully" in names


# ------------------------------------------------------- context pruning


def _prune_agent(messages, window, ratio, keep=40):
    agent = _BareAgent()
    agent._messages = messages
    agent._context_window = window
    agent._context_trim_ratio = ratio
    agent._context_keep_messages = keep
    return agent


def _tool_round(ids, result="r"):
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": i, "type": "function",
                         "function": {"name": "t", "arguments": "{}"}} for i in ids]},
        *[{"role": "tool", "tool_call_id": i, "content": result} for i in ids],
    ]


def test_prune_history_disabled_when_ratio_zero():
    messages = [{"role": "system", "content": "s"}] + _tool_round(["c1"]) * 50
    agent = _prune_agent(messages, window=1000, ratio=0.0)
    assert agent._prune_history() == 0
    assert len(agent._messages) == 101


def test_prune_history_untouched_under_ratio():
    messages = [{"role": "system", "content": "s"}] + _tool_round(["c1"]) * 3
    agent = _prune_agent(messages, window=100_000, ratio=0.7)
    assert agent._prune_history() == 0
    assert len(agent._messages) == 7


def test_prune_history_drops_complete_tool_rounds():
    # 20 tool rounds of ~1500 chars each in a 8k window -> ~75% usage.
    messages = [{"role": "system", "content": "s"}]
    for n in range(20):
        messages += _tool_round([f"c{n}"], result="r" * 1500)
    total = len(messages)  # 41
    agent = _prune_agent(messages, window=8000, ratio=0.7, keep=4)
    dropped = agent._prune_history()
    assert dropped > 0
    assert len(agent._messages) < total
    roles = [m["role"] for m in agent._messages]
    # No orphan tool messages: every tool follows its assistant(tool_calls).
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] == "assistant"
    # System prompt untouched (except the prune marker), recent messages kept.
    assert agent._messages[0]["role"] == "system"
    assert agent._messages[0]["content"].startswith("s")
    assert agent._messages[-1]["role"] == "tool"


def test_prune_history_never_enters_keep_window():
    messages = [{"role": "system", "content": "s"}]
    for n in range(30):
        messages += _tool_round([f"c{n}"], result="r" * 1500)
    agent = _prune_agent(messages, window=8000, ratio=0.5, keep=10)
    agent._prune_history()
    assert len(agent._messages) >= 11  # system + keep window intact
    assert agent._messages[0]["role"] == "system"


def test_prune_history_appends_marker_once():
    messages = [{"role": "system", "content": "s"}]
    for n in range(20):
        messages += _tool_round([f"c{n}"], result="r" * 1500)
    agent = _prune_agent(messages, window=8000, ratio=0.7, keep=4)
    agent._prune_history()
    assert "[history pruned:" in agent._messages[0]["content"]
    agent._prune_history()  # second call: usage already low, marker replaced
    assert agent._messages[0]["content"].count("[history pruned:") <= 1


def test_prune_history_stops_when_under_ratio():
    messages = [{"role": "system", "content": "s"}]
    for n in range(20):
        messages += _tool_round([f"c{n}"], result="r" * 1500)
    agent = _prune_agent(messages, window=8000, ratio=0.7, keep=4)
    agent._prune_history()
    usage, _ = agent._context_usage()
    assert usage <= 70.0


def test_prune_history_skips_group_crossing_keep_window():
    # A tool round whose tool result would cross into the keep window must
    # not be split: pruning stops instead of creating an orphan tool message.
    messages = [{"role": "system", "content": "s"}]
    for n in range(10):
        messages += _tool_round([f"c{n}"], result="r" * 1500)
    messages.append({"role": "user", "content": "recent question"})
    keep = 3  # keep-window starts inside the last tool round
    agent = _prune_agent(messages, window=8000, ratio=0.1, keep=keep)
    agent._prune_history()
    roles = [m["role"] for m in agent._messages]
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] == "assistant"  # never an orphan tool result


def test_act_calls_prune_history_once_per_turn(monkeypatch):
    agent = _agent_with_tool_result("ok", limit=0)
    agent._context_trim_ratio = 0.7
    agent._context_keep_messages = 40
    calls = []
    monkeypatch.setattr(
        type(agent), "_prune_history",
        lambda self: calls.append(1) or 0)
    agent.act("do it")
    assert len(calls) == 1


# --------------------------------------------- e2e turn + config round-trip


def test_e2e_turn_with_all_features_enabled():
    """Full act() turn: capped result + pruning + final answer, valid history."""
    agent = _BareAgent()
    agent._messages = [{"role": "system", "content": "sys prompt"}]
    agent._context_window = 8000
    agent._max_tool_result_chars = 4000
    agent._context_trim_ratio = 0.7
    agent._context_keep_messages = 4
    agent._turn_started = None
    agent._show_thinking = False
    agent._reset_thinking()
    agent._stream = _FakeStream()
    agent.tool_definitions = []
    agent._tool_impls = {"big_tool": lambda: "B" * 50_000}
    rounds = [[{"id": "c1", "name": "big_tool", "arguments": "{}"}]]

    def fake_stream_round(tools):
        calls = rounds.pop(0) if rounds else []
        return "final answer" if not calls else "", calls

    agent._stream_round = fake_stream_round
    agent._stream_completion = lambda tools=None: iter(())
    agent.act("use the tool")
    roles = [m["role"] for m in agent._messages]
    # system, user, assistant(tool_calls), tool(capped), assistant(answer)
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_content = agent._messages[3]["content"]
    assert "[truncated, 50000 chars total]" in tool_content
    assert len(tool_content) <= 4000 + 100  # cap + marker
    assert agent._messages[-1]["content"] == "final answer"


def test_config_round_trip_parses_new_keys(tmp_path):
    from src.common import load_config
    config_text = """
[brain]
model = "m"
system_prompt = "s"
max_tool_result_chars = 4000
context_trim_ratio = 0.7
context_keep_messages = 50

[agent]
tool_doc_chars = 300
tool_groups = "files, web"
"""
    path = tmp_path / "config.toml"
    path.write_text(config_text, encoding="utf-8")
    config = load_config(path)
    assert config.brain.max_tool_result_chars == 4000
    assert config.brain.context_trim_ratio == 0.7
    assert config.brain.context_keep_messages == 50
    assert config.agent.tool_doc_chars == 300
    assert config.agent.tool_groups == "files, web"


def test_config_round_trip_missing_keys_use_defaults():
    agent = _BareAgent()
    agent.config = SimpleNamespace(
        brain=SimpleNamespace(model="m"),
        agent=SimpleNamespace(),
    )
    # Same resolution logic as Agent.__init__ (exercised via getattr defaults).
    assert int(getattr(agent.config.brain, "max_tool_result_chars", 0) or 0) == 0
    assert float(getattr(agent.config.brain, "context_trim_ratio", 0) or 0) == 0.0
    assert int(getattr(agent.config.agent, "tool_doc_chars", 0) or 0) == 0
    assert getattr(agent.config.agent, "tool_groups", "all") == "all"


# ---------------------------------------------------- runtime tool toggling


def _toggle_agent():
    """Bare agent with a small fake tool set for toggle tests."""
    agent = _BareAgent()
    agent._disabled_tools = set()

    def tool_a() -> str:
        return "a"

    def tool_b() -> str:
        return "b"

    agent.tools = [tool_a, tool_b]
    agent.tool_definitions = [
        live_ai_mod.Brain._openai_tool_definition(t) for t in agent.tools
    ]
    agent._tool_impls = {t.__name__: t for t in agent.tools}
    return agent


def test_set_tool_enabled_disables_and_restores():
    agent = _toggle_agent()
    message = agent.set_tool_enabled("tool_a", False)
    assert "disabled" in message
    names = [d["function"]["name"] for d in agent.enabled_tool_definitions()]
    assert names == ["tool_b"]
    message = agent.set_tool_enabled("tool_a", True)
    assert "enabled" in message
    names = [d["function"]["name"] for d in agent.enabled_tool_definitions()]
    assert names == ["tool_a", "tool_b"]


def test_set_tool_enabled_unknown_name_returns_error():
    agent = _toggle_agent()
    message = agent.set_tool_enabled("nope", False)
    assert message.startswith("Error:")
    assert agent.enabled_tool_definitions() == agent.tool_definitions


def test_disabled_tool_execution_refused():
    agent = _toggle_agent()
    agent.set_tool_enabled("tool_a", False)
    result = agent._execute_tool({"name": "tool_a", "arguments": "{}"})
    assert result.startswith("Error:") and "disabled" in result
    # Enabled tools still run.
    assert agent._execute_tool({"name": "tool_b", "arguments": "{}"}) == "b"


def test_toggle_tools_by_topic():
    import types as _types
    dummy = _types.SimpleNamespace(model_info=live_ai_mod.Brain.model_info)
    agent = _toggle_agent()
    agent.tools = live_ai_mod._default_tools(dummy)
    agent.tool_definitions = [
        live_ai_mod.Brain._openai_tool_definition(t) for t in agent.tools]
    agent._tool_impls = {t.__name__: t for t in agent.tools}
    agent.toggle_tools(["finance"], enabled=False)
    names = [d["function"]["name"] for d in agent.enabled_tool_definitions()]
    assert "fetch_financial_data" not in names
    assert "save_financial_chart" not in names
    assert "web_search" in names
    agent.toggle_tools(["finance"], enabled=True)
    names = [d["function"]["name"] for d in agent.enabled_tool_definitions()]
    assert "fetch_financial_data" in names


def test_stream_round_sends_filtered_definitions():
    agent = _toggle_agent()
    agent._context_window = 4096
    agent._turn_started = None
    agent._show_thinking = False
    agent._reset_thinking()
    agent._stream = _FakeStream()
    sent = {}

    def fake_completion(tools=None):
        sent["tools"] = tools
        return iter(())

    agent._stream_completion = fake_completion
    agent.set_tool_enabled("tool_a", False)
    agent._stream_round(agent.tool_definitions)
    names = [d["function"]["name"] for d in sent["tools"]]
    assert names == ["tool_b"]


def test_mcp_refresh_keeps_disabled_mcp_tools(monkeypatch):
    import sys, types as _types
    agent = _toggle_agent()
    agent._context_window = 4096
    agent._mcp_tool_names = []
    fake_mcp = _types.ModuleType("src.tools.mcp")

    def mcp_tool_bindings(existing_names, manager=None):
        def mcp_x() -> str:
            return "x"
        return ([live_ai_mod.Brain._openai_tool_definition(mcp_x)], {"mcp_x": mcp_x})

    def get_manager():
        return _types.SimpleNamespace(refresh=lambda: [])

    fake_mcp.mcp_tool_bindings = mcp_tool_bindings
    fake_mcp.get_manager = get_manager
    monkeypatch.setitem(sys.modules, "src.tools.mcp", fake_mcp)
    import src.tools.mcp as real_mcp  # the agent imports the package module
    monkeypatch.setattr(real_mcp, "mcp_tool_bindings", mcp_tool_bindings)
    monkeypatch.setattr(real_mcp, "get_manager", get_manager)
    agent.refresh_mcp_tools()  # injects mcp_x
    assert "mcp_x" in agent._tool_impls
    agent.set_tool_enabled("mcp_x", False)
    agent.refresh_mcp_tools()  # re-injection must keep it disabled
    assert "mcp_x" not in agent._tool_impls
    names = [d["function"]["name"] for d in agent.enabled_tool_definitions()]
    assert "mcp_x" not in names
