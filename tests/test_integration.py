import src.session as session_mod
from src.session import ConversationSession
from tests.conftest import FakeAgent

TRANSCRIPT = [
    "/help",
    "/voice on",
    "summarise the plan",
    "just answer",
    "/clear",
    "/info",
    "final answer",
    "/quit",
]


def test_full_session_transcript(capsys, monkeypatch):
    speech_calls, clear_calls = [], []
    monkeypatch.setattr(session_mod, "allow_speech", lambda enable: speech_calls.append(enable))
    monkeypatch.setattr(session_mod, "clear_chat", lambda: clear_calls.append(1))
    it = iter(TRANSCRIPT)
    agent = FakeAgent()
    session = ConversationSession(agent, lambda: next(it))
    assert session.run() == 0

    assert agent.acted == ["summarise the plan", "just answer", "final answer"]
    assert agent.printed == []
    assert speech_calls == [True, False]  # enabled then disabled on shutdown
    assert clear_calls == [1]
    # Per the §2.2 algorithm, command turns `continue` before the finally-flush,
    # so only the 3 model turns + shutdown() flush.
    assert agent.flush_count == 4
    out = capsys.readouterr().out
    assert "fake-model-info" in out
