import pytest

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


def test_banner_mentions_help(capsys):
    it = iter(["/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).banner()
    assert "/help" in capsys.readouterr().out


# -------------------------------------------------------------------- T4 help


def test_help_lists_commands(capsys):
    s = make(["/help", "/quit"])
    s.run()
    out = capsys.readouterr().out
    for cmd in ["/help", "/quit", "/clear", "/voice", "/tools", "/info"]:
        assert cmd in out
    assert s.agent.acted == []


# ------------------------------------------------------------------- T5 clear


def test_clear_calls_agent():
    it = iter(["/clear", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.cleared == 1
    assert agent.acted == []


class NoClear:
    def flush_stream(self):
        pass


def test_clear_without_support(capsys):
    it = iter(["/clear", "/quit"])
    ConversationSession(NoClear(), lambda: next(it)).run()
    assert "not supported" in capsys.readouterr().out.lower()


# ------------------------------------------------------------------- T6 voice


def test_voice_on_off():
    it = iter(["/voice on", "/voice OFF", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == [True, False]


def test_voice_usage(capsys):
    it = iter(["/voice maybe", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == []
    assert "usage" in capsys.readouterr().out.lower()


# ------------------------------------------------------------------- T7 tools


def test_tools_toggle_runtime():
    it = iter(["/tools off", "hello", "/tools on", "world", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    # first prompt streamed, second acted
    assert agent.printed == ["stream"]
    assert agent.acted == ["world"]


def test_tools_usage(capsys):
    it = iter(["/tools", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "usage" in capsys.readouterr().out.lower()


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


def test_ctrl_c_on_prompt_exits():
    def reader():
        raise KeyboardInterrupt

    agent = FakeAgent()
    assert ConversationSession(agent, reader).run() == 0


def test_ctrl_c_during_turn_exits(capsys):
    class Interrupting(FakeAgent):
        def act(self, prompt):
            raise KeyboardInterrupt

    counter = {"n": 0}

    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"

    agent = Interrupting()
    assert ConversationSession(agent, reader).run() == 0
    assert "Bye" in capsys.readouterr().out


# ------------------------------------------------------------- T12 quit aliases


@pytest.mark.parametrize("token", ["/quit", "exit", "EXIT", r"\x"])
def test_quit_aliases(token):
    it = iter([token])
    agent = FakeAgent()
    assert ConversationSession(agent, lambda: next(it)).run() == 0


# ----------------------------------------------------------------- T13 shutdown


def test_shutdown_disables_voice_if_enabled():
    it = iter(["/voice on", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == [True, False]


def test_shutdown_no_voice_call_when_never_enabled():
    it = iter(["/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == []


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
