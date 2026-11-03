"""User-input tool for background agents (port of smolagents ``UserInputTool``).

Lets an isolated/background agent pause and ask the live session's user a
question mid-run. The question is queued through the existing background
result plumbing (``queue_agent_result`` with the ``ask_user`` kind) and the
call blocks on an answer queue with a timeout, so a silent user can never
hang the scheduler thread forever.

The tool is registered **only** for isolated agents: for the interactive
main agent it would deadlock the UI thread.
"""
from __future__ import annotations

import queue as queue_mod
import threading
import uuid
from logging import getLogger

__all__ = ["ask_user", "deliver_answer", "configure"]

_CONFIG = None          # [tools] section (ask_user_timeout_s override)
_ANSWER_QUEUES: dict[str, queue_mod.Queue] = {}
_QUEUES_LOCK = threading.Lock()
_DEFAULT_TIMEOUT_S = 300.0


def configure(config) -> None:
    """Keep the ``[tools]`` section (``ask_user_timeout_s`` override)."""
    global _CONFIG
    _CONFIG = getattr(config, "tools", None)


def _default_timeout() -> float:
    raw = getattr(_CONFIG, "ask_user_timeout_s", _DEFAULT_TIMEOUT_S)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return value if value > 0 else _DEFAULT_TIMEOUT_S


def ask_user(question: str, timeout_s: float | None = None) -> str:
    """
    Ask the live session's user a question and wait for the answer.

    Background agents call this when a task is ambiguous and proceeding
    blindly would be wrong. The question appears in the main session
    (kind ``ask_user``); the call blocks until the user answers through
    the session UI (which calls :func:`deliver_answer`) or the timeout
    expires.

    Args:
        question: What to ask the user (a concrete, self-contained question).
        timeout_s: How long to wait; defaults to ``[tools]
            ask_user_timeout_s`` (300s).

    Returns:
        The user's answer text, or a graceful "no answer" message on
        timeout (never raises to the model).
    """
    question = str(question or "").strip()
    if not question:
        return "Error: empty question"
    # Local import avoids a dependency cycle: src.live_ai imports src.tools.
    from src.live_ai import queue_agent_result

    request_id = uuid.uuid4().hex[:12]
    answer_queue: queue_mod.Queue = queue_mod.Queue()
    with _QUEUES_LOCK:
        _ANSWER_QUEUES[request_id] = answer_queue
    try:
        queue_agent_result("ask_user", request_id,
                           f"Background agent asks: {question}")
        getLogger(__name__).info(
            "ask_user(%s): question queued, waiting for an answer",
            request_id)
        if timeout_s is None:
            timeout_s = _default_timeout()
        try:
            answer = answer_queue.get(timeout=max(0.1, float(timeout_s)))
            answer = str(answer).strip()
            return answer if answer else "(no answer given)"
        except queue_mod.Empty:
            return ("(the user did not answer in time; proceed with your "
                    "best judgment and mention the assumption)")
    finally:
        with _QUEUES_LOCK:
            _ANSWER_QUEUES.pop(request_id, None)


def deliver_answer(request_id: str, answer: str) -> bool:
    """
    Deliver the user's answer for a pending ``ask_user`` request.

    Called by the session UI when the user replies to a background agent's
    question. Returns ``False`` when the request is unknown or already
    timed out.
    """
    with _QUEUES_LOCK:
        answer_queue = _ANSWER_QUEUES.get(str(request_id))
    if answer_queue is None:
        return False
    answer_queue.put(str(answer or ""))
    return True
