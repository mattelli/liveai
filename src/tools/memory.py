"""Personal-memory and conversation-history tools.

``store_memory`` appends personal notes from the conversation to a history
file, ``recall_memory`` reads them back, ``clear_chat`` resets the live
conversation (re-seeded with the recalled personal data), and
``compact_chat`` shrinks it in place: the oldest messages are summarized
into a single ``[Compacted context ...]`` message while the system prompt,
the recalled personal data and the most recent turns stay verbatim. The
summary comes from a bound LLM summarizer (``bind_summarizer``) when
available, otherwise from a built-in extractive fallback.

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure` (``[brain]`` history path/system
prompt, ``[agent]`` bootstrap prompt). The one piece of non-config state - the
live message list of the running session - is bound via :func:`bind_history`.

Multi-agent isolation (PLAN.md task 2): the tool logic lives in
:class:`MemoryTools`, so every pooled agent can own its own instance
(:meth:`MemoryTools.for_owner`) instead of sharing module-level state. The
module-level functions delegate to a module-default instance whose state is
backed by the module global ``_history_provider``, keeping the historical
monkeypatch-able seam and call API working unchanged for the main agent,
existing callers and tests.
"""

from __future__ import annotations

import pathlib
from datetime import datetime

from src.common import dump_func_call, truncate_text

__all__ = [
    "MemoryTools", "configure", "bind_history", "bind_summarizer",
    "recall_memory", "store_memory", "clear_chat", "compact_chat",
]

_config_cache = None      # [brain] + [agent] sections injected via configure()
_history_provider = None  # seam: callable returning the live messages list


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Keeps the ``[brain]`` section (``memory_path``,
    ``system_prompt``) and the ``[agent]`` section (``bootstrap_prompt``).
    """
    global _config_cache
    _config_cache = {
        "brain": getattr(config, "brain", None),
        "agent": getattr(config, "agent", None),
    }


def _group_aligned_start(messages: list[dict], keep: int) -> int:
    """First index of the kept tail: never splits a tool-call group.

    ``compact_chat`` keeps ``messages[start:]`` verbatim and summarizes
    everything before ``start``. Starting from ``len(messages) - keep`` the
    index is walked *backwards* (keeping more messages) while the tail would
    begin with a ``role == "tool"`` message: such a message answers an
    ``assistant``/``tool_calls`` message that would otherwise land in the
    summarized range, and OpenAI-compatible servers reject the resulting orphan
    tool result ("messages with role 'tool' must be a response to a preceding
    message with 'tool_calls'").

    Tool results are contiguous, so a tail that does *not* start with a
    ``tool`` message also guarantees the summarized range does not end with a
    dangling ``assistant(tool_calls)``. Returns ``0`` when nothing can be
    summarized (the whole history would have to stay verbatim).

    Mirrors ``Agent._next_prunable_group_end`` (``src/live_ai.py``): both keep
    the "never split a tool round" invariant, one walking from the head, one
    from the tail.
    """
    start = max(0, len(messages) - max(1, keep))
    while start > 0 and messages[start].get("role") == "tool":
        start -= 1
    return start


class MemoryTools:
    """Per-agent personal-memory tools (own history binding, no shared state).

    A pooled agent gets a fresh instance via :meth:`for_owner`, so parallel
    agents never fight over one module-level binding.
    """

    def __init__(self) -> None:
        self._history_provider = None  # callable returning the live messages list
        self._summarizer = None        # callable messages -> summary text

    def for_owner(self, agent) -> "MemoryTools":
        """Fresh instance for a specific agent (per-agent tool isolation)."""
        return MemoryTools()

    def bind_history(self, provider) -> None:
        """
        Bind the live conversation history of the running session (called by
        ``Brain``). ``provider`` is a callable returning the current messages
        list (plain OpenAI-style dicts); it is re-read on every call, so
        clearing or reassigning the session history is always reflected.
        """
        self._history_provider = provider

    def bind_summarizer(self, provider) -> None:
        """
        Bind the summarizer used by :meth:`compact_chat` (called by ``Brain``
        when an LLM-backed summary is available). ``provider`` is a callable
        taking the messages to summarize (plain OpenAI-style dicts) and
        returning a plain-text summary. When unbound, the built-in extractive
        fallback (:meth:`_fallback_summary`) is used, keeping compaction
        testable and usable without a model.
        """
        self._summarizer = provider

    def _config(self, key, attr, default=None):
        section = (_config_cache or {}).get(key)
        return getattr(section, attr, default) if section is not None else default

    def _history_file(self) -> pathlib.Path:
        return pathlib.Path(self._config("brain", "memory_path", "")).resolve()

    def _read_memory(self) -> str:
        """Raw personal-memory text (undecorated, no ``[CALL]`` dump).

        Internal counterpart of :meth:`recall_memory`: the decorated tool is
        for the model, while the compaction/clear paths read the file directly
        so an internal read never prints a nested tool-call block.
        """
        history = self._history_file()
        if not history.exists():
            return "No personal data found!"
        return history.read_text(encoding="utf-8")

    def _head_len(self, messages: list[dict]) -> int:
        """Length of the re-seeded head: ``[system, recalled personal data]``.

        ``clear_chat``/``_reseeded_head`` always put the system prompt first
        and the recalled personal data right after it. Compaction must not
        summarize that head (the personal data would be duplicated into the
        summary and the system prompt re-summarized), so the summarized range
        starts after it. Returns 0 for a history with no recognizable head.
        """
        if not messages:
            return 0
        head = 1 if messages[0].get("role") == "system" else 0
        if (head == 1 and len(messages) > 1
                and messages[1].get("role") == "user"
                and messages[1].get("content") == self._read_memory()):
            head = 2
        return head

    def _shaped_user_messages(self) -> list[str]:
        """Selectable user messages of the live history (as plain text).

        Excludes the re-seeded head (``_head_len``) and compaction summaries
        (``_is_summary``), so ``store_memory`` never captures the recalled
        personal data or a ``[Compacted context ...]`` summary as if the user
        had written it.
        """
        messages = self._history_provider() if self._history_provider else []
        head = self._head_len(messages)
        return [
            m["content"] for m in messages[head:]
            if m.get("role") == "user" and isinstance(m.get("content"), str)
            and not self._is_summary(m.get("content"))
        ]

    @classmethod
    def _is_summary(cls, content) -> bool:
        """True when ``content`` is a compaction summary, not user text."""
        return (isinstance(content, str)
                and content.startswith(cls._SUMMARY_PREFIX))

    @dump_func_call
    def recall_memory(self) -> str:
        """
        Read users personal data from file.
        Useful when the users ask to recall their personal data.
        """
        return self._read_memory()

    @dump_func_call
    def store_memory(self, content: str = "") -> str:
        """
        Store the user personal content to a text file for later recall with
        recall_memory. Useful when the users explicitly mention that the
        prompted text is "personal" or ask to save/remember something about them.
        Pass the personal text in ``content``; when omitted, the last user message
        of the conversation is stored, preferring messages that mention
        "personal". The re-seeded head (recalled personal data) and compaction
        summaries are never captured. Returns the file path written.
        """
        try:
            text = (content or "").strip()
            if not text:
                user_msgs = self._shaped_user_messages()
                bootstrap_prompt = self._config("agent", "bootstrap_prompt", None)
                if user_msgs and user_msgs[0] == bootstrap_prompt:
                    user_msgs = user_msgs[1:]
                selected = [m for m in user_msgs if "personal" in m.lower()]
                if not selected and user_msgs:
                    selected = [user_msgs[-1]]  # fall back to the latest user message
                if not selected:
                    return "Error: no user message found to store."
                text = selected[-1]
            history = self._history_file()
            now = datetime.now()  # local time
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            user_prompt = f"At {timestamp} user wrote: {text} \n"
            if not history.exists():
                history.write_text(user_prompt, encoding="utf-8")
            else:
                with open(file=history, mode="a", encoding="utf-8") as f:
                    f.write(user_prompt)
            return f"Conversation written to {history}"
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def clear_chat(self) -> str:
        """
        Useful when user ask for cleaning conversation history.
        """
        try:
            provider = self._history_provider
            if provider is None:
                return "Error: no chat session is bound."
            messages = provider()
            system_prompt = self._config("brain", "system_prompt", "")
            system = next(
                (m["content"] for m in messages if m["role"] == "system"),
                system_prompt,
            )
            user_msgs = [
                m["content"] for m in messages
                if m.get("role") == "user" and isinstance(m.get("content"), str)
                and not self._is_summary(m.get("content"))
            ]
            messages.clear()  # in place: the session keeps the same list object
            messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": self._read_memory()})
            if user_msgs:
                messages.append({"role": "user", "content": user_msgs[-1]})
            return "Chat cleared successfully"
        except Exception as e:
            return f"Error: {e}"

    # ---------------------------------------------------------------- compaction

    _SUMMARY_PREFIX = "[Compacted context"
    _FALLBACK_SUMMARY_LIMIT = 2000  # extractive fallback cap (chars)

    @classmethod
    def _fallback_summary(cls, messages: list[dict]) -> str:
        """Extractive fallback: join message texts, head+tail truncated.

        Used when no summarizer is bound, keeping ``compact_chat`` usable and
        unit-testable without an LLM. Truncation goes through the shared
        :func:`src.common.truncate_text` helper, so the marker reports the
        original length and the format matches the tool-result capping.
        """
        texts = [
            m.get("content") or "" for m in messages
            if isinstance(m.get("content"), str)
        ]
        joined = "\n".join(t.strip() for t in texts if t.strip())
        if not joined:
            return "(no content)"
        return truncate_text(joined, cls._FALLBACK_SUMMARY_LIMIT)

    def _reseeded_head(self, messages: list[dict]) -> list[dict]:
        """``[system, recalled personal data]`` head, like ``clear_chat``.

        Preserves the live system message (falls back to the ``[brain]``
        system_prompt when absent) and the personal-memory seeding. The
        personal data is read undecorated (no nested ``[CALL]`` block) and
        capped with ``[brain] max_tool_result_chars`` (0 = uncapped), so a
        large ``memory.log`` cannot bloat the history on every compaction.
        """
        system_prompt = self._config("brain", "system_prompt", "")
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            system_prompt,
        )
        limit = self._config("brain", "max_tool_result_chars", 0) or 0
        return [
            {"role": "system", "content": system},
            {"role": "user",
             "content": truncate_text(self._read_memory(), int(limit))},
        ]

    @dump_func_call
    def compact_chat(self, keep_last: int = 4) -> str:
        """
        Useful when the conversation history grows long: summarize the oldest
        messages into a compact summary and keep only the most recent turns
        verbatim. The system prompt and the recalled personal data are
        preserved; the history is shrunk in place (the session keeps the same
        list object). Pass ``keep_last`` to control how many recent messages
        are kept (minimum 2, default 4).

        The kept tail is group-aligned (``_group_aligned_start``): a tool round
        is never split, so the resulting history stays valid for
        OpenAI-compatible servers (no orphan ``tool`` result, no dangling
        ``assistant(tool_calls)``). The re-seeded head (system prompt +
        recalled personal data) is excluded from the summarized range
        (``_head_len``), so the personal data is never duplicated into the
        summary. The summary itself is capped with ``[brain]
        max_tool_result_chars`` (0 = uncapped) before it is injected, so a
        summarizer that ignores ``max_tokens`` cannot bloat the history. When
        the boundary would leave nothing to summarize, the history is left
        untouched.
        """
        try:
            provider = self._history_provider
            if provider is None:
                return "Error: no chat session is bound."
            messages = provider()
            try:
                keep = int(keep_last)
            except (TypeError, ValueError):
                keep = 4
            keep = max(2, keep)
            # At least one summaryable message beyond the kept recent turns.
            if len(messages) < keep + 2:
                return "Context is already compact; nothing to do."

            # Group-aligned split: the kept tail never starts with an orphan
            # ``tool`` message (whose ``assistant``/``tool_calls`` parent would
            # be summarized away) and the summarized range never ends with a
            # dangling ``assistant(tool_calls)`` (which would make the
            # summarizer request invalid). See ``_group_aligned_start``.
            start = _group_aligned_start(messages, keep)
            # The re-seeded head (system prompt + recalled personal data) is
            # never summarized: the personal data would be duplicated into the
            # summary and the system prompt re-summarized. See ``_head_len``.
            head = self._head_len(messages)
            old = messages[head:start]
            recent = list(messages[start:])
            if not old:
                return "Context is already compact; nothing to do."
            summarizer = self._summarizer or self._fallback_summary
            summary = (summarizer(old) or "").strip()
            if not summary:
                summary = self._fallback_summary(old) or "(empty summary)"
            # A bound summarizer may ignore max_tokens (custom provider,
            # overrunning local model); cap the text before it enters the
            # history so an oversized summary cannot defeat the compaction.
            # 0/unset = uncapped (the historical behavior).
            summary_limit = self._config("brain", "max_tool_result_chars", 0) or 0
            summary = truncate_text(summary, int(summary_limit))

            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_history = self._reseeded_head(messages)
            new_history.append({
                "role": "user",
                "content": (
                    f"{self._SUMMARY_PREFIX} on {stamp}] Summary of the "
                    f"earlier conversation:\n{summary}"
                ),
            })
            new_history.extend(recent)

            messages[:] = new_history  # in place: same list object for the session
            return (
                f"Chat compacted: {len(old)} messages summarized into 1, "
                f"{len(recent)} kept."
            )
        except Exception as e:
            return f"Error: {e}"


class _ModuleDefaultMemoryTools(MemoryTools):
    """Module-default instance backed by the module global ``_history_provider``.

    Keeps the historical ``tools.memory._history_provider`` seam (monkeypatched
    by tests) working while pooled agents use their own isolated instances.
    """

    @property
    def _history_provider(self):
        return globals()["_history_provider"]

    @_history_provider.setter
    def _history_provider(self, value) -> None:
        globals()["_history_provider"] = value


# Module-default instance: keeps the historical module-level API working for
# the main agent, existing callers and tests. Pooled agents use their own
# instance instead (see Agent._build_tools / MemoryTools.for_owner).
_default = _ModuleDefaultMemoryTools()


def bind_history(provider) -> None:
    """Bind the live history of the running session (module-default instance)."""
    _default.bind_history(provider)


def recall_memory() -> str:
    """Read users personal data from file (module-default instance)."""
    return _default.recall_memory()


def store_memory(content: str = "") -> str:
    """Store the user personal content (module-default instance)."""
    return _default.store_memory(content)


def clear_chat() -> str:
    """Clean the conversation history (module-default instance)."""
    return _default.clear_chat()


def bind_summarizer(provider) -> None:
    """Bind the compaction summarizer (module-default instance)."""
    _default.bind_summarizer(provider)


def compact_chat(keep_last: int = 4) -> str:
    """Compact the conversation history into a summary (module-default)."""
    return _default.compact_chat(keep_last)
