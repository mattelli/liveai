"""Interactive input layer for chat mode, built on prompt_toolkit.

This module is the single input primitive for ``--chat``. It gives line
editing, persistent history, inline history suggestions, multi-line paste and
completion from one cross-platform dependency. prompt_toolkit is a required
installation (see ``setup-lms-cpu.ps1``), but the import guard below keeps
``import src.input_reader`` from crashing in a degraded environment; in that
case ``InputReader`` routes through :func:`src.common.prompt_input`.

``readline`` / ``pyreadline3`` are deliberately not used (Unix-only /
incomplete). See ``PLAN-INPUT.md`` for the design rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.common import prompt_input

try:  # pragma: no cover - env dependent
    from prompt_toolkit import PromptSession, ANSI
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

    _PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - env dependent
    PromptSession = None  # type: ignore[assignment]
    ANSI = None  # type: ignore[assignment]
    FileHistory = None  # type: ignore[assignment]
    InMemoryHistory = None  # type: ignore[assignment]
    AutoSuggestFromHistory = None  # type: ignore[assignment]
    _PROMPT_TOOLKIT = False

# Bold ANSI codes for the prompt label (same palette as src.common.cprint).
_PROMPT_COLORS = {
    "black": "30", "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "grey": "90", "white": "37",
}


class InputReader:
    """Owns one interactive prompt and maps EOF / interrupt correctly.

    Parameters
    ----------
    memory_path:
        When set (and prompt_toolkit is available) a :class:`FileHistory` is
        used so Up/Down recall works across sessions. The parent directory is
        created if needed. When ``None`` an in-memory history is used.
    auto_suggest:
        Enable inline history suggestions.
    completer:
        Optional prompt_toolkit completer (e.g. WordCompleter) forwarded to
        :class:`PromptSession`.
    multiline:
        Advanced: explicit multi-line editing. Enter inserts a newline and
        Esc+Enter accepts. Defaults to ``False`` (single-line).
    session:
        Test seam: a prebuilt :class:`PromptSession`.
    fallback:
        Test seam / no-prompt_toolkit path. When provided, no prompt_toolkit
        session is constructed and ``fallback(label)`` is used directly.
    explicit_paste:
        Enable the ``/paste`` ... ``.`` fallback for terminals that do not send
        bracketed-paste markers. Defaults to ``False`` (bracketed paste is
        handled natively by prompt_toolkit).
    prompt_color:
        Color name (black/red/green/yellow/blue/magenta/cyan/grey/white) used
        to render the prompt label in bold on the prompt_toolkit path. The
        builtin ``input()`` fallback ignores it (keeps its default color).
    continuation:
        Prompt used for continuation lines in explicit-paste / multiline mode.
    """

    def __init__(
        self,
        memory_path: Path | None = None,
        auto_suggest: bool = True,
        completer=None,
        multiline: bool = False,
        session=None,
        fallback: Callable[[str], str | None] | None = None,
        explicit_paste: bool = False,
        prompt_color: str | None = None,
        continuation: str = "... ",
    ):
        self._fallback = fallback
        self._explicit_paste = explicit_paste
        self._continuation = continuation
        self._session = session
        self._closed = False

        # Whether this reader should use prompt_toolkit at all. A supplied
        # session or fallback wins; otherwise prompt_toolkit is used when
        # importable. The PromptSession itself is built lazily on first read()
        # so constructing an InputReader never touches the terminal.
        self._use_pt = session is not None or (
            fallback is None and _PROMPT_TOOLKIT
        )
        self._history_path = memory_path
        if memory_path is not None:
            Path(memory_path).parent.mkdir(parents=True, exist_ok=True)
        self._auto_suggest = auto_suggest
        self._completer = completer
        self._multiline = multiline
        self._prompt_color = prompt_color

    def _ensure_session(self):
        if self._session is None:
            self._session = self._build_session(
                memory_path=self._history_path,
                auto_suggest=self._auto_suggest,
                completer=self._completer,
                multiline=self._multiline,
            )
        return self._session

    @staticmethod
    def _build_session(memory_path, auto_suggest, completer, multiline):
        history = InMemoryHistory()
        if memory_path is not None:
            history = FileHistory(str(memory_path))
        return PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory() if auto_suggest else None,
            completer=completer,
            multiline=multiline,
        )

    def read(self, label: str = "") -> str | None:
        """Return the entered text (may contain newlines), or ``None`` on EOF.

        ``KeyboardInterrupt`` propagates to the caller (the session loop
        re-prompts on Ctrl+C and exits on EOF).
        """
        if not self._use_pt:
            return (self._fallback or prompt_input)(label)
        session = self._ensure_session()
        if self._explicit_paste:
            return self._read_explicit_paste(session, label)
        try:
            return session.prompt(self._styled_label(label))
        except EOFError:
            return None

    def _styled_label(self, label: str):
        """Colorize the prompt label (bold) when a prompt_color is configured."""
        code = _PROMPT_COLORS.get((self._prompt_color or "").lower())
        if code is None:
            return label
        return ANSI(f"\x1b[1;{code}m{label}\x1b[0m")

    def _read_explicit_paste(self, session, label: str) -> str | None:
        """Read a line; if it is exactly ``/paste``, collect lines until ``.``."""
        try:
            first = session.prompt(self._styled_label(label))
        except EOFError:
            return None
        if first.strip() != "/paste":
            return first

        lines: list[str] = []
        while True:
            try:
                line = session.prompt(self._continuation)
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        return "\n".join(lines)

    def close(self) -> None:
        """Flush history / release resources. Safe to call twice.

        ``FileHistory`` appends as it goes, so there is nothing to flush; this
        exists so callers can wrap the session loop in ``try/finally``.
        """
        self._closed = True
