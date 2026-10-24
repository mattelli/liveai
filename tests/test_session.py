import json

import pytest
from types import SimpleNamespace

import src.session as session_mod
from src.session import ConversationSession
from tests.conftest import FakeAgent


def make(values, use_tools=True):
    it = iter(values)
    return ConversationSession(FakeAgent(), lambda: next(it), use_tools=use_tools)


# ------------------------------------------------------------------- T3 core


def test_run_dispatches_to_agent(capsys):
    it = iter(["hello", "/quit"])
    agent = FakeAgent()
    s = ConversationSession(agent, lambda: next(it))
    assert s.run() == 0
    assert agent.acted == ["hello"]
    # 1 for the turn + 1 from shutdown() (consistent with T10's 2 turns -> 3 flushes)
    assert agent.flush_count == 2


def test_run_skips_empty_input():
    it = iter(["   ", "hi", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == ["hi"]


def test_run_stops_on_eof():
    it = iter([None])
    agent = FakeAgent()
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == []


def test_tools_off_uses_stream_path():
    it = iter(["hi", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it), use_tools=False).run()
    assert agent.acted == [] and agent.printed == ["stream"]


# -------------------------------------------------------------------- T4 help


def test_help_lists_commands(capsys):
    s = make(["/help", "/quit"])
    s.run()
    out = capsys.readouterr().out
    for cmd in ["/help", "/quit", "/clear", "/voice", "/tools", "/info", "/model", "/session"]:
        assert cmd in out
    assert s.agent.acted == []


# ------------------------------------------------------------------- T5 clear


def test_clear_calls_tool(monkeypatch):
    calls = []
    monkeypatch.setattr(session_mod, "clear_chat", lambda: calls.append(1))
    it = iter(["/clear", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == [1]


# ------------------------------------------------------------------- T6 voice


def test_voice_on_off(monkeypatch):
    calls = []
    monkeypatch.setattr(session_mod, "allow_speech", lambda enable: calls.append(enable))
    it = iter(["/voice on", "/voice OFF", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == [True, False]


def test_voice_usage(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(session_mod, "allow_speech", lambda enable: calls.append(enable))
    it = iter(["/voice maybe", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == []
    assert "usage" in capsys.readouterr().out.lower()


# ------------------------------------------------------------------- T7 tools


def test_tools_argument_shows_usage(capsys):
    it = iter(["/tools on", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert "usage" in capsys.readouterr().out.lower()
    assert agent.acted == []


class ToolAgent(FakeAgent):
    """FakeAgent exposing a small set of tools for the /tools launcher."""

    def __init__(self):
        super().__init__()
        self.launched = []

        def greet():
            """Say hello."""
            self.launched.append("greet")
            return "hello"

        def echo(text: str):
            """Echo text back."""
            self.launched.append(("echo", text))
            return text

        self.tools = [greet, echo]


def test_tools_lists_available_tools(capsys):
    agent = ToolAgent()
    it = iter(["/tools", "", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "greet" in out and "echo" in out
    assert "Say hello." in out


def test_tools_runs_tool_without_params():
    agent = ToolAgent()
    it = iter(["/tools", "1", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.launched == ["greet"]


def test_tools_prompts_required_param_by_name():
    agent = ToolAgent()
    it = iter(["/tools", "echo", "hi there", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.launched == [("echo", "hi there")]


def test_tools_blank_selection_cancels(capsys):
    agent = ToolAgent()
    it = iter(["/tools", "   ", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.launched == []
    assert "cancelled" in capsys.readouterr().out.lower()


def test_tools_unknown_selection(capsys):
    agent = ToolAgent()
    it = iter(["/tools", "nope", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.launched == []
    assert "unknown tool" in capsys.readouterr().out.lower()


def test_tools_empty_list(capsys):
    it = iter(["/tools", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "no tools available" in capsys.readouterr().out.lower()


# -------------------------------------------------------------------- T8 info


def test_info_prints_model_info(capsys):
    it = iter(["/info", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "fake-model-info" in capsys.readouterr().out


# ----------------------------------------------------------- T9 unknown command


def test_unknown_command_not_sent():
    it = iter(["/frobnicate", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == []


def test_unknown_command_message(capsys):
    it = iter(["/frobnicate", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "unknown command" in capsys.readouterr().out.lower()


# --------------------------------------------------------- T10 error recovery


def test_turn_error_is_recovered(capsys):
    class Boom(FakeAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def act(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("kaboom")
            super().act(prompt)

    agent = Boom()
    it = iter(["bad", "good", "/quit"])
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == ["good"]
    assert "kaboom" in capsys.readouterr().out
    assert agent.flush_count == 3  # flushed every turn, including the failed one


# ------------------------------------------------------ T11 KeyboardInterrupt


def test_ctrl_c_on_prompt_prompts_again(capsys):
    """Ctrl+C while waiting for input clears the line; EOF quits."""
    state = {"n": 0}

    def reader():
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt
        return "/quit"

    agent = FakeAgent()
    assert ConversationSession(agent, reader).run() == 0
    assert agent.acted == []


def test_ctrl_c_on_prompt_exits_in_non_interactive_mode():
    """Voice / non-interactive prompt sources keep the old semantics."""
    def reader():
        raise KeyboardInterrupt

    agent = FakeAgent()
    session = ConversationSession(agent, reader, quit_on_prompt_interrupt=True)
    assert session.run() == 0


def test_ctrl_c_during_turn_cancels_but_keeps_session(capsys):
    """Ctrl+C during a response stops the turn, not the session."""
    class Interrupting(FakeAgent):
        def act(self, prompt):
            raise KeyboardInterrupt

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = Interrupting()
    assert ConversationSession(agent, reader).run() == 0
    out = capsys.readouterr().out
    assert "cancelled" in out.lower()
    assert "Bye" in out  # session survived and exited via /quit


# ------------------------------------------------- T11b cancel_generation flow


def test_graceful_cancel_after_sigint_reports_and_continues(capsys):
    """SIGINT-driven cancel that ends the stream gracefully is reported."""
    from tests.conftest import Frag

    class CancelStreamAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self._cancel_requested = False

        @property
        def cancel_requested(self):
            return self._cancel_requested

        def cancel_generation(self):
            self._cancel_requested = True

        def model_fragments(self, prompt):
            # Simulate: SIGINT handler fired mid-stream and the prediction
            # stream ended gracefully after the server-side cancel.
            self.cancel_generation()
            return [Frag("partial")]

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = CancelStreamAgent()
    assert ConversationSession(agent, reader, use_tools=False).run() == 0
    out = capsys.readouterr().out
    assert agent.printed == ["partial"]
    assert "cancelled" in out.lower()
    assert "Bye" in out


def test_sigint_during_generation_hits_cancel_handler():
    """The temporary SIGINT handler routes Ctrl+C to agent.cancel_generation()."""
    import signal

    class SigintAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.cancel_calls = 0
            self._cancel_requested = False

        @property
        def cancel_requested(self):
            return self._cancel_requested

        def cancel_generation(self):
            self.cancel_calls += 1
            self._cancel_requested = True

        def act(self, prompt):
            handler = signal.getsignal(signal.SIGINT)
            assert handler is not signal.default_int_handler  # guard installed
            handler(signal.SIGINT, None)  # simulate Ctrl+C reaching the handler
            raise KeyboardInterrupt  # simulated abort from the stream callbacks

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = SigintAgent()
    previous = signal.getsignal(signal.SIGINT)
    try:
        assert ConversationSession(agent, reader).run() == 0
    finally:
        signal.signal(signal.SIGINT, previous)
    assert agent.cancel_calls == 1


# ------------------------------------------------------------- T12 quit aliases


@pytest.mark.parametrize("token", ["/quit", "exit", "EXIT"])
def test_quit_aliases(token):
    it = iter([token])
    agent = FakeAgent()
    assert ConversationSession(agent, lambda: next(it)).run() == 0


def test_backslash_x_is_not_a_command():
    """Backslash-x was removed as a quit alias; it is now plain text for the model."""
    it = iter([r"\x", "/quit"])
    agent = FakeAgent()
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == [r"\x"]


# ----------------------------------------------------------------- T13 shutdown


def test_shutdown_disables_voice_if_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(session_mod, "allow_speech", lambda enable: calls.append(enable))
    it = iter(["/voice on", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == [True, False]


def test_shutdown_no_voice_call_when_never_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(session_mod, "allow_speech", lambda enable: calls.append(enable))
    it = iter(["/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == []


def test_shutdown_idempotent():
    s = ConversationSession(FakeAgent(), lambda: "/quit")
    assert s.shutdown() == 0 and s.shutdown() == 0


# ------------------------------------------------------------------- T19 bootstrap


def test_session_calls_bootstrap_once(capsys):
    """Convention: bootstrap() increments a counter, it does NOT reuse act()."""
    class B(FakeAgent):
        def __init__(self):
            super().__init__()
            self.n = 0

        def bootstrap(self):
            self.n += 1

    it = iter(["hi", "/quit"])
    agent = B()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.n == 1
    assert agent.acted == ["hi"]  # bootstrap turn not counted as acted


def test_session_without_bootstrap_ok():
    it = iter(["hi", "/quit"])  # FakeAgent has no bootstrap()
    assert ConversationSession(FakeAgent(), lambda: next(it)).run() == 0


# ------------------------------------------ T21 Ctrl+C stops speech, not the turn


def test_ctrl_c_during_speech_stops_audio_only(capsys, monkeypatch, tmp_path):
    """Ctrl+C while the agent speaks: audio stops, turn and session continue."""
    pytest.importorskip("piper", reason="requires piper")
    pytest.importorskip("sounddevice", reason="requires sounddevice")
    import signal

    import src.audio_generator as audio_gen
    import src.tools.speech as speech_mod
    from tests.conftest import FakeOutputStream, FakePiperVoice

    class SigintSecondWriteStream(FakeOutputStream):
        """FakeOutputStream simulating Ctrl+C on the second audio write."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self._writes_this_playback = 0

        def start(self):
            super().start()
            self._writes_this_playback = 0

        def write(self, data):
            super().write(data)
            self._writes_this_playback += 1
            if self._writes_this_playback == 2:
                handler = signal.getsignal(signal.SIGINT)
                assert handler is not signal.default_int_handler  # speech guard
                handler(signal.SIGINT, None)

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not-a-real-model")
    generator = audio_gen.SpeechGenerator(
        model_path=model,
        voice=FakePiperVoice(chunks=(b"\x01\x02",) * 4),
        stream=SigintSecondWriteStream(),
    )
    monkeypatch.setattr(speech_mod, "_speech_generator", generator)
    monkeypatch.setattr(speech_mod, "_speech_stopped", False)

    class SpeakingAgent(FakeAgent):
        def act(self, prompt):
            super().act(prompt)
            speech_mod.speak_if_enabled("a spoken response")

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return ["hi", "again", "/quit"][counter["n"] - 1]

    agent = SpeakingAgent()
    assert ConversationSession(agent, reader).run() == 0
    assert agent.acted == ["hi", "again"]  # turn/session survived, not cancelled
    assert generator.stream.aborted == 2  # speech stopped on both turns
    out = capsys.readouterr().out.lower()
    assert out.count("speech stopped") == 2
    assert "generation cancelled" not in out


# ----------------------------------------------------------------- T20 sessions


class SessionAgent(FakeAgent):
    """FakeAgent carrying a live history and a [brain] config for /session."""

    def __init__(self, sessions_dir):
        super().__init__()
        self._messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        self.config = SimpleNamespace(brain=SimpleNamespace(sessions_dir=sessions_dir))
        self._model_name = "fake-model"

    @property
    def messages(self):
        return self._messages


def test_session_store_writes_json(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session store", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["messages"] == agent._messages
    assert data["model"] == "fake-model"
    assert "stored" in capsys.readouterr().out.lower()


def test_session_store_named_sanitizes_name(tmp_path):
    agent = SessionAgent(tmp_path)
    it = iter(["/session store my chat", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert (tmp_path / "my_chat.json").exists()


def test_session_store_refuses_overwrite(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session store dup", "/session store dup", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert "already exists" in capsys.readouterr().out.lower()


def test_session_store_without_dir(tmp_path, capsys):
    agent = SessionAgent(None)  # config present, sessions_dir unset
    it = iter(["/session store", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert "sessions_dir" in capsys.readouterr().out.lower()


def test_session_store_without_history(capsys):
    it = iter(["/session store", "/quit"])  # FakeAgent has no messages
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "no chat history" in capsys.readouterr().out.lower()


def test_session_list_shows_files(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session store alpha", "/session list", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "alpha.json" in out
    assert "  3  " in out  # message count column (right-aligned)


def test_session_list_empty(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session list", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert "no session files" in capsys.readouterr().out.lower()


def test_session_load_restores_history(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session store alpha", "/session load alpha", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent._messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert "loaded" in capsys.readouterr().out.lower()


def test_session_load_by_index(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session store one", "/session store two", "/session load 1", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert "one.json" in capsys.readouterr().out.lower()


def test_session_load_mutates_history_in_place(tmp_path):
    """Loading replaces the list contents (bind_history sees the same object)."""
    agent = SessionAgent(tmp_path)
    live = agent._messages
    it = iter(["/session store alpha", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    live.clear()
    live.append({"role": "user", "content": "junk"})
    it = iter(["/session load alpha", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert live is agent._messages
    assert "junk" not in [m.get("content") for m in live]
    assert [m.get("content") for m in live] == ["sys", "hi", "hello"]


def test_session_load_invalid_file(tmp_path, capsys):
    (tmp_path / "broken.json").write_text('{"messages": "nope"}', encoding="utf-8")
    agent = SessionAgent(tmp_path)
    it = iter(["/session load broken", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out.lower()
    assert "error loading session" in out
    assert agent._messages[0]["content"] == "sys"  # history untouched


def test_session_load_missing_file(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session load nope", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert "no session file" in capsys.readouterr().out.lower()


def test_session_load_requires_target(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session load", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert "usage" in capsys.readouterr().out.lower()


def test_session_usage_on_bare_and_bad_subcommand(tmp_path, capsys):
    agent = SessionAgent(tmp_path)
    it = iter(["/session", "/session frobnicate", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert capsys.readouterr().out.lower().count("usage") == 2


# ------------------------------------------------------------- /mcp command


class McpAgent(FakeAgent):
    """FakeAgent with a refresh_mcp_tools method (real Agent has one)."""

    def __init__(self):
        super().__init__()
        self.refresh_calls = 0
        self.refresh_message = "MCP refreshed (srv): 1 tool(s) injected: mcp_srv_echo"

    def refresh_mcp_tools(self):
        self.refresh_calls += 1
        return self.refresh_message


def _mcp_manager(monkeypatch, servers=None):
    import src.tools.mcp as mcp_mod

    class FakeMcpManager:
        def __init__(self):
            self.refreshed = 0

        def status(self):
            return servers or {
                "srv": {"transport": "stdio", "status": "connected",
                        "tools": ["echo"]},
                "down": {"transport": "http", "status": "error: unreachable",
                         "tools": []},
            }

        def refresh(self):
            self.refreshed += 1
            return {"srv": ["echo"]}

    manager = FakeMcpManager()
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    return manager


def test_mcp_command_listed_in_commands():
    assert "/mcp" in ConversationSession.COMMANDS


def test_cmd_mcp_lists_servers(monkeypatch, capsys):
    _mcp_manager(monkeypatch)
    it = iter(["/mcp", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "srv" in out and "down" in out
    assert "connected" in out and "error: unreachable" in out
    assert "stdio" in out and "http" in out


def test_cmd_mcp_refresh_calls_agent_refresh(monkeypatch, capsys):
    _mcp_manager(monkeypatch)
    agent = McpAgent()
    it = iter(["/mcp refresh", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.refresh_calls == 1
    assert "MCP refreshed" in capsys.readouterr().out


def test_cmd_mcp_refresh_falls_back_to_manager(monkeypatch, capsys):
    manager = _mcp_manager(monkeypatch)
    it = iter(["/mcp refresh", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()  # no refresh method
    assert manager.refreshed == 1
    assert "MCP refreshed" in capsys.readouterr().out


def test_cmd_mcp_refresh_without_manager(monkeypatch, capsys):
    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    it = iter(["/mcp refresh", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "not configured" in capsys.readouterr().out.lower()


def test_cmd_mcp_without_manager_shows_not_configured(monkeypatch, capsys):
    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    it = iter(["/mcp", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out.lower()
    assert "not configured" in out
    assert "usage" in out


def test_cmd_mcp_unknown_subcommand_shows_usage(monkeypatch, capsys):
    _mcp_manager(monkeypatch)
    it = iter(["/mcp frobnicate", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out.lower()
    assert "usage" in out
    assert "srv" not in out  # listing skipped for a bad sub-command


# ------------------------------------------------------------------- /monitor


def test_monitor_command_shows_recent_events(capsys):
    import src.live_ai as live_ai
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    monitor.record("glm", "start", prompt="research X")
    monitor.record("glm", "answer", answer="found it")
    live_ai.bind_monitor(monitor)
    try:
        s = make(["/monitor", "/quit"])
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "[glm] started: research X" in out
    assert "[glm] answered: found it" in out


def test_monitor_command_with_limit(capsys):
    import src.live_ai as live_ai
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    for i in range(5):
        monitor.record("glm", "tool_call", tool=f"t{i}")
    live_ai.bind_monitor(monitor)
    try:
        s = make(["/monitor 2", "/quit"])
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "t3" in out and "t4" in out
    assert "t0" not in out and "t1" not in out


def test_monitor_command_clear(capsys):
    import src.live_ai as live_ai
    from src.live_ai import AgentMonitor

    monitor = AgentMonitor(maxlen=10)
    monitor.record("glm", "start", prompt="x")
    live_ai.bind_monitor(monitor)
    try:
        agent = FakeAgent()
        agent.config = SimpleNamespace(agent=SimpleNamespace(
            bootstrap_prompt="b", parallel_agents=0, monitor="calls"))
        it = iter(["/monitor clear", "/monitor", "/quit"])
        s = ConversationSession(agent, lambda: next(it))
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "cleared" in out.lower()
    assert "No monitoring data available." in out
    assert monitor.snapshot() == []


def test_monitor_command_no_pool(capsys):
    import src.live_ai as live_ai
    live_ai.bind_monitor(None)
    s = make(["/monitor", "/quit"])
    s.run()
    out = capsys.readouterr().out
    assert "not active" in out


def test_monitor_command_bad_usage(capsys):
    import src.live_ai as live_ai
    from src.live_ai import AgentMonitor

    live_ai.bind_monitor(AgentMonitor(maxlen=10))
    try:
        s = make(["/monitor bogus", "/quit"])
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "Usage: /monitor [N|clear]" in out


def test_monitor_command_empty_buffer_hint(capsys):
    """Empty buffer (monitoring on or off) -> the no-data hint."""
    import src.live_ai as live_ai
    from src.live_ai import AgentMonitor

    live_ai.bind_monitor(AgentMonitor(maxlen=10))
    try:
        agent = FakeAgent()
        agent.config = SimpleNamespace(agent=SimpleNamespace(
            bootstrap_prompt="b", parallel_agents=0))
        it = iter(["/monitor", "/quit"])
        s = ConversationSession(agent, lambda: next(it))
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "No monitoring data available." in out


def test_monitor_in_help(capsys):
    s = make(["/help", "/quit"])
    s.run()
    assert "/monitor" in capsys.readouterr().out
