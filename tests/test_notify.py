"""Tests for the [delegation.notify] terminal hint (PLAN-notify.md).

The hint is printed by the background thread right after a result is
queued: one line, task kind + id only — the result text is never shown.
"""

import re

import pytest
from types import SimpleNamespace

import src.live_ai as live_ai_mod


@pytest.fixture(autouse=True)
def _notify_off():
    """Reset the notifier before/after each test (hints off by default)."""
    live_ai_mod._NOTIFY["terminal"] = False
    yield
    live_ai_mod._NOTIFY["terminal"] = False


@pytest.fixture
def notify_on():
    live_ai_mod._NOTIFY["terminal"] = True
    return None


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_no_hint_when_disabled(capsys):
    """Missing [delegation.notify] / terminal=false: no output, queue still filled."""
    live_ai_mod.queue_agent_result("scheduled", "t-1", "secret answer")
    out = capsys.readouterr().out
    assert "completed" not in out
    assert live_ai_mod.drain_agent_results() == [("scheduled", "t-1", "secret answer")]


def test_hint_scheduled_kind_no_preview(notify_on, capsys):
    """scheduled: exact wording, and the result text never appears."""
    live_ai_mod.queue_agent_result("scheduled", "t-1", "SECRET RESULT TEXT")
    out = _ANSI.sub("", capsys.readouterr().out)
    assert out == ("🔔 Scheduled task t-1 completed.\n")
    assert "SECRET RESULT TEXT" not in out  # no preview, ever
    # delivery is unchanged: the full text is still queued for the next turn
    assert live_ai_mod.drain_agent_results() == [("scheduled", "t-1", "SECRET RESULT TEXT")]


def test_hint_delegation_kind(notify_on, capsys):
    live_ai_mod.queue_agent_result("delegation", "researcher (agent-1)", "research answer")
    out = _ANSI.sub("", capsys.readouterr().out)
    assert out == ("🔔 Sub-agent task researcher (agent-1) completed.\n")
    assert "research answer" not in out


def test_hint_unknown_kind_generic_label(notify_on, capsys):
    live_ai_mod.queue_agent_result("future-kind", "x-9", "some answer")
    out = _ANSI.sub("", capsys.readouterr().out)
    assert out == ("🔔 Background task x-9 completed.\n")


def test_no_hint_for_empty_result(notify_on, capsys):
    """The scheduler's 'produced no answer' case prints nothing."""
    live_ai_mod.queue_agent_result("scheduled", "t-2", "")
    live_ai_mod.queue_agent_result("scheduled", "t-3", "   \n  ")
    assert capsys.readouterr().out == ""


def test_configure_notify_missing_section_keeps_off():
    live_ai_mod.configure_notify(SimpleNamespace())
    assert live_ai_mod._NOTIFY["terminal"] is False


def test_configure_notify_enables_hints():
    live_ai_mod.configure_notify(SimpleNamespace(
        delegation=SimpleNamespace(notify=SimpleNamespace(terminal=True))))
    assert live_ai_mod._NOTIFY["terminal"] is True
    live_ai_mod._NOTIFY["terminal"] = False  # restore for other tests


def test_configure_notify_false_disables():
    live_ai_mod.configure_notify(SimpleNamespace(
        delegation=SimpleNamespace(notify=SimpleNamespace(terminal=False))))
    assert live_ai_mod._NOTIFY["terminal"] is False
