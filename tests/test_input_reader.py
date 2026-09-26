import pytest

from src.input_reader import InputReader
from src.session import ConversationSession
from tests.conftest import FakeAgent, make_session


# ------------------------------------------------------------------ P0 smoke


def test_prompt_toolkit_smoke(pt_input):
    pt_input.send_text("hello\n")
    assert make_session(pt_input).prompt("> ") == "hello"


# ------------------------------------------------------------------ P1 core


def test_read_returns_line(pt_input):
    pt_input.send_text("hello\n")
    r = InputReader(session=make_session(pt_input))
    assert r.read("You: ") == "hello"


def test_read_eof_returns_none(pt_input):
    pt_input.send_text("\x04")  # Ctrl-D on empty buffer
    r = InputReader(session=make_session(pt_input))
    assert r.read("You: ") is None


def test_read_ctrl_c_propagates(pt_input):
    pt_input.send_text("\x03")  # Ctrl-C
    r = InputReader(session=make_session(pt_input))
    with pytest.raises(KeyboardInterrupt):
        r.read("You: ")


def test_fallback_used_when_no_session():
    r = InputReader(session=None, fallback=lambda label: "fallback")
    assert r.read("You: ") == "fallback"


def test_close_is_idempotent():
    r1 = InputReader(session=None, fallback=lambda label: "")
    r1.close()
    r1.close()
    r2 = InputReader(session=None, fallback=lambda label: "")
    r2.close()


# ------------------------------------------------------------------ P2 history


def test_history_records_entries(pt_input):
    from prompt_toolkit.history import InMemoryHistory

    session = make_session(pt_input, history=InMemoryHistory())
    r = InputReader(session=session)
    pt_input.send_text("one\n")
    r.read("> ")
    pt_input.send_text("two\n")
    r.read("> ")
    assert session.history.get_strings() == ["one", "two"]


def test_file_history_persists_across_readers(pt_input, tmp_path):
    from prompt_toolkit.history import FileHistory

    hist = tmp_path / ".chat_history"
    session = make_session(pt_input, history=FileHistory(str(hist)))
    pt_input.send_text("remember me\n")
    InputReader(session=session).read("> ")
    # FileHistory appends immediately when the entry is stored.
    assert hist.exists()


def test_history_path_creates_parent(tmp_path):
    nested = tmp_path / "a" / "b" / ".chat_history"
    InputReader(history_path=nested)  # constructs FileHistory
    assert nested.parent.exists()


# ------------------------------------------------------------- P4 multi-line


def test_bracketed_paste_is_one_prompt(pt_input):
    # paste marker, two lines, closing marker, then Enter
    pt_input.send_text("\x1b[200~line one\nline two\x1b[201~\n")
    r = InputReader(session=make_session(pt_input))
    assert r.read("You: ") == "line one\nline two"


def test_multiline_prompt_is_not_a_command():
    it = iter(["/quit\nstill here", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == ["/quit\nstill here"]
    assert agent.cleared == 0


def test_single_line_command_still_dispatches():
    it = iter(["/clear", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.cleared == 1 and agent.acted == []


# --------------------------------------------------------- P5 explicit paste


def test_explicit_paste_block(pt_input):
    pt_input.send_text("/paste\n")
    pt_input.send_text("a\n")
    pt_input.send_text("b\n")
    pt_input.send_text(".\n")
    r = InputReader(session=make_session(pt_input), explicit_paste=True)
    assert r.read("You: ") == "a\nb"


def test_explicit_paste_eof_returns_partial(pt_input):
    pt_input.send_text("/paste\na\n\x04")
    r = InputReader(session=make_session(pt_input), explicit_paste=True)
    assert r.read("You: ") == "a"


def test_explicit_paste_disabled(pt_input):
    pt_input.send_text("/paste\n")
    r = InputReader(session=make_session(pt_input), explicit_paste=False)
    assert r.read("You: ") == "/paste"


# ----------------------------------------------------------- P6 completion


def test_completer_forwarded(pt_input):
    completer = object()
    session = make_session(pt_input, completer=completer)
    InputReader(session=session)
    assert session.completer is completer


def test_completer_installed_via_constructor(pt_input):
    class C:
        def get_completions(self, document, complete_event):
            return iter(())

    completer = C()
    session = make_session(pt_input, completer=completer)
    r = InputReader(session=session)
    pt_input.send_text("/he\t\n")
    # Best-effort smoke: driving the completion menu is layout-dependent, the
    # important contract is that the completer is installed and read does not
    # crash.
    r.read("You: ")
    assert session.completer is completer


# ----------------------------------------------------------- P8 end-to-end


def test_paste_end_to_end_reaches_agent_as_one_turn(pt_input):
    pt_input.send_text("\x1b[200~def f():\n    return 1\x1b[201~\n")
    r = InputReader(session=make_session(pt_input))
    agent = FakeAgent()
    it = iter([r.read("You: "), "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == ["def f():\n    return 1"]


def test_explicit_paste_end_to_end(pt_input):
    for chunk in ["/paste\n", "one\n", "two\n", ".\n", "/quit\n"]:
        pt_input.send_text(chunk)
    r = InputReader(session=make_session(pt_input), explicit_paste=True)
    agent = FakeAgent()
    ConversationSession(agent, lambda: r.read("You: ")).run()
    assert agent.acted == ["one\ntwo"]
