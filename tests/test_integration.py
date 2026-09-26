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


def test_full_session_transcript(capsys):
    it = iter(TRANSCRIPT)
    agent = FakeAgent()
    session = ConversationSession(agent, lambda: next(it))
    assert session.run() == 0

    assert agent.acted == ["summarise the plan", "just answer", "final answer"]
    assert agent.printed == []
    assert agent.speech == [True, False]  # enabled then disabled on shutdown
    assert agent.cleared == 1
    # Per the §2.2 algorithm, command turns `continue` before the finally-flush,
    # so only the 3 model turns + shutdown() flush.
    assert agent.flush_count == 4
    out = capsys.readouterr().out
    assert "fake-model-info" in out
