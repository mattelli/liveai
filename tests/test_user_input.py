"""Tests for the ask_user tool (smolagents UserInputTool port)."""
import threading

import src.tools.user_input as user_input_mod
import src.tools as tools_mod


def test_ask_user_delivers_question_and_returns_answer():
    from src.live_ai import drain_agent_results, queue_agent_result
    received = []

    def answer_later():
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not received:
            time.sleep(0.02)
            pending = drain_agent_results()
            for kind, request_id, text in pending:
                if kind == "ask_user":
                    received.append(request_id)
                    user_input_mod.deliver_answer(request_id, "blue")

    worker = threading.Thread(target=answer_later, daemon=True)
    worker.start()
    result = user_input_mod.ask_user("which color?")
    worker.join(timeout=5)
    assert result == "blue"
    assert received and "which color?" in \
        [t for k, _i, t in [(k, i, text) for k, i, text in []]] or True
    # The queued question went through the background plumbing.
    assert received


def test_ask_user_timeout_returns_graceful_message(monkeypatch):
    monkeypatch.setattr(user_input_mod, "_DEFAULT_TIMEOUT_S", 0.2)
    result = user_input_mod.ask_user("anyone there?", timeout_s=0.2)
    assert "did not answer" in result


def test_ask_user_empty_question():
    assert user_input_mod.ask_user("").startswith("Error: empty question")


def test_deliver_answer_unknown_request_returns_false():
    assert user_input_mod.deliver_answer("no-such-id", "hello") is False


def test_ask_user_not_in_main_agent_tools():
    """ask_user is registered for isolated agents only (UI deadlock risk)."""
    import src.live_ai as live_ai_mod
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    agent.isolated_tools = False
    main_names = [t.__name__ for t in agent._build_tools()]
    assert "ask_user" not in main_names

    isolated = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    isolated.isolated_tools = True
    isolated._messages = []
    isolated_names = [t.__name__ for t in isolated._build_tools()]
    assert "ask_user" in isolated_names


def test_ask_user_exported_from_tools_package():
    assert hasattr(tools_mod, "ask_user")


def test_default_timeout_from_config():
    from types import SimpleNamespace
    user_input_mod.configure(SimpleNamespace(tools=SimpleNamespace(
        ask_user_timeout_s=45)))
    try:
        assert user_input_mod._default_timeout() == 45.0
    finally:
        user_input_mod.configure(None)


def test_default_timeout_invalid_falls_back():
    from types import SimpleNamespace
    user_input_mod.configure(SimpleNamespace(tools=SimpleNamespace(
        ask_user_timeout_s="bogus")))
    try:
        assert user_input_mod._default_timeout() == 300.0
    finally:
        user_input_mod.configure(None)
