"""``delegate_task``: fire-and-forget task delegation to configured sub-agents.

The main session agent calls :func:`delegate_task` (registered as a tool) with
an ``[agents]`` registry name and a task prompt; the tool validates the name,
spawns a daemon thread running an isolated, quiet child agent, and returns an
acknowledgment immediately. On completion the child's final text is queued via
:func:`src.live_ai.queue_agent_result` under the ``"delegation"`` kind, so the
session delivers it exactly like scheduled-task results.

Flat topology (PLAN-multiagent goal 4): children are built without the
delegation tool, so they cannot spawn further agents.

Optional result summarization (PLAN-multiagent goal 5): with
``[agents] summarize_results = true``, child results longer than
``summarize_max_chars`` are condensed by an LLM summary pass before queuing
(fail-open: a summary failure queues the original text).

The tool logic lives here (not in ``src.tools``) because it needs
``src.live_ai`` internals (the Agent class, the results queue); importing
``src.live_ai`` from a ``src.tools`` submodule would be circular.
"""

from __future__ import annotations

import itertools
import threading
from logging import getLogger
from types import SimpleNamespace
from typing import Callable

from src.agent_registry import agent_registry, summarize_settings
from src.common import dump_func_call
from src.sub_agents import SubAgentBuildError, build_sub_agent

__all__ = [
    "delegate_task", "configure_delegation", "summarize_text",
    "pending_delegations",
]

log = getLogger(__name__)

# Application config injected via configure_delegation() (main.py pattern:
# tools get the config, live state is bound separately).
_APP_CONFIG: SimpleNamespace | None = None

# Test seams: patched by tests to avoid real LLM calls.
_CHILD_RUNNER: Callable | None = None   # runs one child turn; default = real
_SUMMARIZER: Callable | None = None     # summary pass; default = real

# Delegation task ids: agent-1, agent-2, ... (process-wide counter).
_TASK_IDS = itertools.count(1)


def reset_task_ids() -> None:
    """Restart the task-id counter at agent-1 (test isolation hook)."""
    global _TASK_IDS
    _TASK_IDS = itertools.count(1)

# Names of delegations still running (for pending_delegations introspection).
_RUNNING: set[str] = set()
_RUNNING_LOCK = threading.Lock()


def configure_delegation(config: SimpleNamespace) -> None:
    """Inject the application config (called at session startup)."""
    global _APP_CONFIG
    _APP_CONFIG = config


def pending_delegations() -> int:
    """Number of delegations currently running (introspection/tests)."""
    with _RUNNING_LOCK:
        return len(_RUNNING)


@dump_func_call
def delegate_task(agent_name: str, task: str) -> str:
    """Delegate a task to a configured sub-agent (fire-and-forget).

    ``agent_name`` must name a ``[agents.<name>]`` config entry. Returns an
    acknowledgment immediately; the child's result arrives later through the
    background-results queue (kind ``"delegation"``).
    """
    config = _APP_CONFIG
    specs = agent_registry()
    if config is None or not specs:
        raise LookupError(
            "Error: no sub-agents configured (set [agents.<name>] tables in "
            "config.toml)")
    name = str(agent_name or "").strip()
    if not name:
        raise LookupError("Error: agent_name is required")
    spec = specs.get(name) or specs.get(name.lower())
    if spec is None:
        raise LookupError(
            f"Error: unknown sub-agent {agent_name!r} "
            f"(configured: {', '.join(sorted(specs)) or 'none'})")
    task = str(task or "").strip()
    if not task:
        raise LookupError("Error: task is required")

    task_id = f"agent-{next(_TASK_IDS)}"
    try:
        # Build synchronously (cheap, no LLM call): a broken config must
        # surface as an immediate tool error, not as a silent thread crash.
        agent = build_sub_agent(config, spec)
    except SubAgentBuildError as exc:
        raise LookupError(f"Error: cannot start sub-agent: {exc}") from exc

    with _RUNNING_LOCK:
        _RUNNING.add(task_id)
    thread = threading.Thread(
        target=_run_child, args=(agent, spec.name, task_id, task),
        name=f"subagent-{spec.name}-{task_id}", daemon=True)
    thread.start()
    log.info("Delegated task %s to sub-agent '%s'", task_id, spec.name)
    return (f"Delegated to '{spec.name}' (task_id={task_id}). The result "
            "will arrive automatically when the sub-agent finishes.")


def _run_child(agent, agent_name: str, task_id: str, task: str) -> None:
    """One child ``act()`` turn on a background thread (never raises).

    The final text is queued for main-session delivery; failures queue an
    error text instead of raising into the thread (a raised exception in a
    daemon thread would be silently swallowed anyway).
    """
    runner = _CHILD_RUNNER or _default_child_runner
    try:
        result = runner(agent, task)
    except Exception as exc:
        log.error("Delegation %s ('%s') failed: %s", task_id, agent_name, exc)
        result = f"Error: sub-agent '{agent_name}' failed: {exc}"
    result = (result or "").strip() or \
        f"Sub-agent '{agent_name}' returned an empty answer."
    try:
        summarize, max_chars = summarize_settings()
        if summarize and len(result) > max_chars > 0:
            # Fail-open: a summarization failure queues the original text.
            result = summarize_text(agent, result) or result
    except Exception as exc:  # summarization must never lose the result
        log.warning("Delegation %s summarization failed: %s", task_id, exc)
    from src.live_ai import queue_agent_result
    queue_agent_result("delegation", f"{agent_name} ({task_id})", result)
    with _RUNNING_LOCK:
        _RUNNING.discard(task_id)
    log.info("Delegation %s ('%s') completed", task_id, agent_name)


def _default_child_runner(agent, task: str) -> str:
    """One child turn: ``act()`` and return the final text."""
    result = agent.act(task)
    return getattr(result, "text", "") or ""


def summarize_text(agent, text: str) -> str:
    """One LLM summary pass over a long child result (empty on failure).

    Uses the child agent's own OpenAI-compatible client, keeping cloud/local
    routing consistent; a standalone client would duplicate connection setup.
    The prompt reuses the ``summarizer_system`` pattern from ``_PROMPTS``
    (src.live_ai). Any failure returns "" so the caller queues the original.
    """
    summarizer = _SUMMARIZER
    if summarizer is not None:
        return summarizer(text)
    from src.live_ai import _PROMPTS
    try:
        resp = agent.client.chat.completions.create(
            model=agent.model_name,
            messages=[
                {"role": "system", "content": _PROMPTS["summarizer_system"]},
                {"role": "user",
                 "content": f"Summarize this result concisely:\n\n{text}"},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return str(resp.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("Result summarizer failed: %s", exc)
        return ""
