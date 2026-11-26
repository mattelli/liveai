import json

import pytest
from types import SimpleNamespace

import src.session as session_mod
from src.session import ConversationSession
from tests.conftest import FakeAgent, require_import


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


def test_quit_flag_from_tool_exits_after_turn(capsys):
    """exit_gracefully's quit flag ends the loop after the turn completes."""
    from src.common import quit_requested, reset_quit

    reset_quit()
    try:
        it = iter(["hi", "never-asked"])
        agent = FakeAgent()
        session = ConversationSession(agent, lambda: next(it))
        # Simulate the tool having run during the first turn.
        original_act = agent.act

        def act_and_request(prompt):
            original_act(prompt)
            from src.common import request_quit
            request_quit()

        agent.act = act_and_request
        assert session.run() == 0
        assert agent.acted == ["hi"]  # second prompt never consumed
        assert "Bye" in capsys.readouterr().out
        assert quit_requested() is True  # flag is not auto-cleared; loop broke
    finally:
        reset_quit()


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


# ------------------------------------------------------------- T5b compact


def test_compact_default_calls_tool(monkeypatch):
    calls = []
    monkeypatch.setattr("src.tools.compact_chat",
                        lambda keep_last=4: calls.append(keep_last) or "ok")
    it = iter(["/compact", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == [4]


def test_compact_with_keep_last_argument(monkeypatch):
    calls = []
    monkeypatch.setattr("src.tools.compact_chat",
                        lambda keep_last=4: calls.append(keep_last) or "ok")
    it = iter(["/compact 2", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == [2]


def test_compact_invalid_argument_falls_back_to_default(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("src.tools.compact_chat",
                        lambda keep_last=4: calls.append(keep_last) or "ok")
    it = iter(["/compact abc", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert calls == [4]
    assert "Invalid keep_last" in capsys.readouterr().out


def test_compact_error_is_reported_in_red(monkeypatch, capsys):
    monkeypatch.setattr("src.tools.compact_chat",
                        lambda keep_last=4: "Error: no chat session is bound.")
    it = iter(["/compact", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "Error: no chat session is bound." in out


def test_help_lists_compact(capsys):
    s = make(["/help", "/quit"])
    s.run()
    out = capsys.readouterr().out
    assert "/compact" in out
    assert "compact old history" in out


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


def test_voice_on_reports_unavailable_speech(monkeypatch, capsys):
    """/voice on surfaces the error and stays off when TTS is unavailable."""
    monkeypatch.setattr(
        session_mod, "allow_speech",
        lambda enable: "Error: speech generation unavailable: MME error 6",
    )
    it = iter(["/voice on", "/quit"])
    session = ConversationSession(FakeAgent(), lambda: next(it))
    session.run()
    out = capsys.readouterr().out.lower()
    assert "mme error 6" in out
    assert "voice enabled" not in out
    assert session._speech_enabled is False


def test_voice_on_succeeds_when_speech_available(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        session_mod, "allow_speech",
        lambda enable: calls.append(enable) or "Speech Generator initialized",
    )
    it = iter(["/voice on", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "voice enabled" in capsys.readouterr().out.lower()
    # shutdown() releases the generator because voice was considered enabled.
    assert calls == [True, False]


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


class ModelListAgent(FakeAgent):
    """FakeAgent exposing the models-file API for the /model command."""

    def __init__(self):
        super().__init__()
        self._model_name = "glm-flash"

    @property
    def model_name(self):
        return self._model_name

    def list_models(self):
        return {
            "glm-flash": {"model": "z-ai/glm-5.3-flash",
                          "base_url": "https://openrouter.ai/api/v1",
                          "provider": "wafer/us"},
            "plain": {"model": "plain-model"},
        }

    def select_model(self, name):
        self._model_name = name
        return f"Model switched to {name}"


def test_model_list_shows_provider_column(capsys):
    """/model lists each entry's provider (dash when unset)."""
    it = iter(["/model", "", "/quit"])
    ConversationSession(ModelListAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "glm-flash" in out
    assert "wafer/us" in out
    assert "plain" in out
    assert "z-ai/glm-5.3-flash" in out


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


def test_generation_cancelled_reported_as_cancel_not_error(capsys):
    """A GenerationCancelled from the agent prints the cancel banner, no ❌."""
    from src.live_ai import GenerationCancelled

    class CancellingAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self._cancel_requested = False

        @property
        def cancel_requested(self):
            return self._cancel_requested

        def cancel_generation(self):
            self._cancel_requested = True

        def act(self, prompt):
            self.cancel_generation()
            raise GenerationCancelled("generation cancelled by user")

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = CancellingAgent()
    assert ConversationSession(agent, reader).run() == 0
    out = capsys.readouterr().out
    assert "generation cancelled" in out.lower()
    assert "❌" not in out  # a cancel is never reported as an error
    assert "Bye" in out  # session survived and exited via /quit


def test_transport_error_during_cancel_is_not_an_error(capsys):
    """An httpx-style abort with the cancel flag set prints no ❌."""
    class AbortingAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self._cancel_requested = False

        @property
        def cancel_requested(self):
            return self._cancel_requested

        def cancel_generation(self):
            self._cancel_requested = True

        def act(self, prompt):
            # Closing the stream mid-read surfaces as a transport error.
            self.cancel_generation()
            raise RuntimeError("Connection error: stream closed")

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = AbortingAgent()
    assert ConversationSession(agent, reader).run() == 0
    out = capsys.readouterr().out
    assert "generation cancelled" in out.lower()
    assert "❌" not in out
    assert "Connection error" not in out  # no error text for a cancel
    assert "Bye" in out


def test_real_error_without_cancel_still_reported(capsys):
    """A genuine failure (no cancel pending) keeps the ❌ path."""
    class FailingAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self._cancel_requested = False

        @property
        def cancel_requested(self):
            return self._cancel_requested

        def act(self, prompt):
            raise RuntimeError("kaboom")

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = FailingAgent()
    assert ConversationSession(agent, reader).run() == 0
    out = capsys.readouterr().out
    assert "❌" in out and "kaboom" in out
    assert "generation cancelled" not in out.lower()


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
    require_import("piper", "sounddevice")
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
        self.config = SimpleNamespace(brain=SimpleNamespace(sessions_dir_cwd=sessions_dir))
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
    # /mcp refresh re-reads mcp.json via refresh_servers(): the fake manager
    # stands in for it (the agent-less fallback path).
    monkeypatch.setattr(mcp_mod, "refresh_servers", manager.refresh)
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


def test_cmd_mcp_reports_disabled(monkeypatch, capsys):
    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "enabled", lambda: False)
    it = iter(["/mcp", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "disabled" in capsys.readouterr().out.lower()


def test_cmd_mcp_refresh_reports_disabled(monkeypatch, capsys):
    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "enabled", lambda: False)
    agent = McpAgent()
    it = iter(["/mcp refresh", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert "disabled" in capsys.readouterr().out.lower()
    assert agent.refresh_calls == 0  # never reaches the agent


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
            bootstrap_prompt="b", monitor="calls"))
        it = iter(["/monitor clear", "/monitor", "/quit"])
        s = ConversationSession(agent, lambda: next(it))
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "cleared" in out.lower()
    assert "No sub-agent activity recorded." in out
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
            bootstrap_prompt="b"))
        it = iter(["/monitor", "/quit"])
        s = ConversationSession(agent, lambda: next(it))
        s.run()
    finally:
        live_ai.bind_monitor(None)
    out = capsys.readouterr().out
    assert "No sub-agent activity recorded." in out


def test_monitor_in_help(capsys):
    s = make(["/help", "/quit"])
    s.run()
    assert "/monitor" in capsys.readouterr().out


# ------------------------------------------------------- /tools grouped view


def test_tools_lists_tools_grouped_by_topic(capsys):
    agent = ToolAgent()
    it = iter(["/tools", "", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    # unknown tools (test doubles) are grouped under "other"
    assert "other:" in out
    assert "greet" in out and "Say hello." in out and "echo" in out


def test_tools_grouped_selection_by_number_runs_tool():
    """Number selection maps to the grouped display order."""
    agent = ToolAgent()
    it = iter(["/tools", "2", "hello", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.launched == [("echo", "hello")]


# ----------------------------------------------------------------- banner logo


def test_banner_shows_gradient_logo(capsys):
    """Default [ui] config: the ASCII-art logo appears inside the panel."""
    agent = FakeAgent()
    agent.config = SimpleNamespace(ui=SimpleNamespace(banner=True))
    it = iter(["/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "LiveAI" in out
    assert "_/\\_" in out or "_____" in out  # a row of the ASCII art


def test_banner_logo_hidden_when_disabled(capsys):
    """[ui] banner_logo = false: panel without the ASCII art."""
    agent = FakeAgent()
    agent.config = SimpleNamespace(ui=SimpleNamespace(
        banner=True, banner_logo=False))
    it = iter(["/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "LiveAI" in out          # panel still there
    assert "_/\\_" not in out               # but no art row


def test_banner_plain_line_when_banner_false(capsys):
    """[ui] banner = false: single-line banner, no panel, no logo."""
    agent = FakeAgent()
    agent.config = SimpleNamespace(ui=SimpleNamespace(
        banner=False, banner_logo=True))
    it = iter(["/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "Agent ready." in out
    assert "LiveAI" not in out


def test_gradient_logo_styles_each_line():
    """Every logo line carries an rgb style interpolating start -> end."""
    from src.session import _LOGO, _gradient_logo

    text = _gradient_logo()
    rows = _LOGO.strip("\n").splitlines()
    assert len(text.spans) >= len(rows)  # one styled span per row (at least)
    first, last = text.spans[0], text.spans[-1]
    assert "rgb(0,200,255)" in str(first.style)      # start color
    assert "rgb(180,0,255)" in str(last.style)       # end color


def test_gradient_logo_custom_art():
    from src.session import _gradient_logo

    text = _gradient_logo("AB\nCD")
    assert "AB" in text.plain and "CD" in text.plain


# ------------------------------------------- background results injection


def test_session_injects_scheduled_results_unchanged():
    import src.live_ai as live_ai

    agent = FakeAgent()
    agent.messages = []
    live_ai.queue_agent_result("scheduled", "t-1", "the reminder text")
    it = iter(["ok", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    contents = [m["content"] for m in agent.messages if m["role"] == "assistant"]
    assert any("Scheduled task t-1 completed" in c and "the reminder text" in c
               for c in contents)


def test_session_skips_unknown_background_kind():
    import src.live_ai as live_ai

    agent = FakeAgent()
    agent.messages = []
    live_ai.queue_agent_result("mystery", "x-1", "odd payload")
    it = iter(["ok", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()  # must not raise
    contents = [m["content"] for m in agent.messages if m["role"] == "assistant"]
    assert not any("odd payload" in c for c in contents)


# ---------------------------------------------------- runtime tool toggling


class ToggleAgent(FakeAgent):
    """FakeAgent exposing the real Agent toggle API (recorded calls)."""

    def __init__(self):
        super().__init__()
        self.toggles = []

    def toggle_tools(self, names, enabled):
        self.toggles.append((list(names), enabled))
        return f"toggled {names} -> {enabled}"


def test_cmd_tools_off_calls_agent_toggle(capsys):
    agent = ToggleAgent()
    it = iter(["/tools off web_search", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.toggles == [(["web_search"], False)]
    assert "toggled" in capsys.readouterr().out


def test_cmd_tools_on_multiple_names(capsys):
    agent = ToggleAgent()
    it = iter(["/tools on finance, zip_files", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.toggles == [(["finance", "zip_files"], True)]


def test_cmd_tools_bad_usage_shows_hint(capsys):
    agent = ToggleAgent()
    it = iter(["/tools frobnicate", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.toggles == []
    assert "usage" in capsys.readouterr().out.lower()


def test_cmd_tools_without_agent_support(capsys):
    it = iter(["/tools off web_search", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out.lower()
    assert "does not support" in out


# ------------------------------------------------------------- /tools list


class ListToolsAgent(FakeAgent):
    """FakeAgent with a real-shaped tool set for the /tools list tests."""

    def __init__(self):
        super().__init__()
        self._disabled_tools = set()

        def web_search(query: str):
            """Search web pages based on the query."""
            return "results"

        def schedule_task(prompt: str, at: str):
            """Schedule a prompt to run at a future time."""
            return "ok"

        self.tools = [web_search, schedule_task]


def test_tools_list_shows_all_tools_with_state(capsys):
    agent = ListToolsAgent()
    it = iter(["/tools list", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    # the agent's own tools are enabled (no marker); the default-set entries
    # of the same names win the display (module-default implementations)
    assert "  web_search - Search web pages" in out
    assert "  schedule_task - Schedule a prompt to run at a future time" in out
    # summary line with counts
    import re

    match = re.search(r"(\d+) of (\d+) tools enabled", out)
    assert match and int(match.group(1)) == 2


def test_tools_list_marks_runtime_disabled_tools(capsys):
    agent = ListToolsAgent()
    agent._disabled_tools = {"web_search"}
    it = iter(["/tools list", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    assert "web_search [off]" in out
    assert "schedule_task - Schedule" in out  # still enabled
    import re

    match = re.search(r"(\d+) of (\d+) tools enabled", out)
    assert match and int(match.group(1)) == 1


def test_tools_list_includes_default_tools_not_in_agent(capsys):
    """Tools excluded by [agent] tool_groups show up with an [off] marker."""
    agent = ListToolsAgent()
    it = iter(["/tools list", "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    out = capsys.readouterr().out
    # recall_memory is in the default set but not in the fake agent's tools
    assert "recall_memory [off]" in out
    assert "generate_image [off]" in out
    import re

    match = re.search(r"(\d+) of (\d+) tools enabled", out)
    assert match and int(match.group(1)) == 2 and int(match.group(2)) > 2


def test_tools_list_usage_hint_in_help(capsys):
    it = iter(["/help", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    out = capsys.readouterr().out
    # the rich table wraps the command cell; check both parts
    assert "/tools" in out and "list shows all tools + state" in out
