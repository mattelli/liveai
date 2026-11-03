import copy
import contextvars
import inspect
import json
import os
import pathlib
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints
from logging import getLogger
from datetime import datetime, timezone
from urllib.parse import urlsplit
from src.common import SimpleNamespace, dump_func_call, cprint, crule, StreamPrinter, set_subagent_dump
from src.tools import (
    exit_gracefully, system_info, current_date_time, get_working_directory,
    read_text_file, write_text_file, run_command, spawn_process,
    web_search, web_images, web_videos, web_news, web_books, extract_webpage,
    capture_image,
    generate_image, unload_image_generator,
    summarize_doc, query_document, image_search,
    list_corpora, store_corpus, query_corpus,
    recall_memory, store_memory, clear_chat, compact_chat, allow_speech, image_response,
    schedule_task, list_scheduled_tasks, cancel_scheduled_task,
    download_file, extract_archive, zip_files,
    search_pattern_in_files, replace_string_in_file, search_in_file,
    fetch_financial_data, save_financial_chart,
    mcp_list_tools, mcp_call_tool,
)
from src.tools import speech as speech_tools


# Padded into the history when a generation is cancelled before any text was
# produced: an empty assistant turn would be dropped by the model's prompt
# template and break user/assistant role alternation on the next request.
CANCELLED_PLACEHOLDER = "(generation cancelled by user)"


def _validated_style(value: str | None, default: str = "cyan") -> str:
    """`[ui]` assistant color: any rich style name, falling back to `default`."""
    value = value or default
    try:
        from rich.style import Style
        Style.parse(value)
    except Exception:
        return default
    return value


def _validated_code_theme(value: str | None, default: str = "monokai") -> str:
    """`[ui]` code theme: any Pygments style name, falling back to `default`."""
    value = value or default
    try:
        from pygments.styles import get_style_by_name
        get_style_by_name(value)
    except Exception:
        return default
    return value


def __getattr__(name: str):
    """Backward-compatible access to ``SpeechGenerator``.

    The speech engine is owned by ``src.tools.speech`` now; this shim keeps
    older callers (and tests) that import it from this module working without
    pulling in the heavy ``piper``/``sounddevice`` stack at import time.
    """
    if name == "SpeechGenerator":
        from src.audio_generator import SpeechGenerator
        return SpeechGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _response_token_cap(max_response_tokens: float, context_window: int) -> int:
    """Resolve ``[brain] max_response_tokens`` into a per-generation cap.

    The config value is a ratio of the model's context window, so the cap
    self-adapts to whichever model is configured: ``0.1`` on a 4k-window model
    caps a response at ~409 tokens. Valid range is 0 < ratio < 1; unset/0
    keeps the historical default of the full context window (no practical
    cap). The cap bounds any single generation - including one the provider
    cannot stop instantly after Ctrl+C (e.g. cancelled during prompt
    processing) - so a stale prediction can never monopolise the model.
    """
    if not max_response_tokens:
        return context_window
    try:
        ratio = float(max_response_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"brain.max_response_tokens must be a number (got {max_response_tokens!r})"
        ) from exc
    if not 0 < ratio < 1:
        raise ValueError(
            "brain.max_response_tokens must be a ratio with 0 < ratio < 1, "
            f"got {max_response_tokens!r}"
        )
    return max(1, int(context_window * ratio))


def _truncate_tool_result(text: str, limit: int) -> str:
    """Cap a tool result before it is stored in the chat history.

    ``limit <= 0`` (unset) or a short-enough result is returned unchanged.
    Oversized results keep the **head and the tail** — ``head = limit * 3 // 4``
    chars, ``tail = limit // 4`` chars — joined by an explicit truncation
    marker carrying the original length. Head+tail matters for append-ordered
    files read whole by tools (e.g. ``recall_memory`` returns the entire
    ``memory.log``, oldest entries first): head-only truncation would
    silently drop the *most recent* entries, so the tail must survive too.
    The marker tells the model the result was cut so it can re-query with a
    targeted tool (``search_in_file``, ``query_document``, ``summarize_doc``)
    instead of re-reading everything. Truncation is preferred over eager
    summarization: it is free, deterministic and never distorts what it keeps.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit // 4
    marker = f"\n… [truncated, {len(text)} chars total] …\n"
    return text[:head] + marker + text[-tail:]


def _thinking_token_cap(value) -> int:
    """Resolve ``[ui] max_thinking_tokens`` into a visualization cap.

    Bounds the number of chain-of-thought tokens that are streamed to the
    console while the model is thinking (reasoning tokens are never part of
    the answer, so this only limits what is shown, not what is generated).
    Unset/0/negative disables the cap (show everything until the answer
    starts); a non-integer value falls back to no cap instead of failing the
    session - a broken display cap must not stop the agent.
    """
    try:
        cap = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return cap if cap > 0 else 0


def _default_tools(agent: "Brain") -> list[Callable]:
    """Tool set of the agent.

    Only ``model_info`` stays a session-bound method (model/endpoint/usage
    state); every other tool lives in a ``src.tools`` submodule and gets its
    config injected via ``src.tools.configure`` (called by ``main.py``).
    """
    return [
        agent.model_info,
        agent_activity,
        recall_memory, store_memory, clear_chat, compact_chat,
        image_response, allow_speech,
        schedule_task, list_scheduled_tasks, cancel_scheduled_task,
        summarize_doc, query_document, image_search,
        list_corpora, store_corpus, query_corpus,
        capture_image, system_info, current_date_time, get_working_directory,
        run_command, spawn_process, search_in_file, search_pattern_in_files,
        read_text_file, replace_string_in_file, write_text_file,
        web_search, web_images, web_videos, web_news, web_books,
        extract_webpage, download_file, extract_archive, zip_files,
        fetch_financial_data, save_financial_chart,
        generate_image, unload_image_generator,
        mcp_list_tools, mcp_call_tool,
        exit_gracefully
    ]


def _select_tools(tools: list[Callable], groups) -> list[Callable]:
    """Filter a tool list down to the configured ``[agent] tool_groups``.

    ``groups`` accepts ``"all"`` (also empty/None: unfiltered, the historical
    behavior), a comma-separated string or a list of topic names from
    ``_TOOL_TOPICS`` (e.g. ``"files, web"`` or ``["files", "web"]``). Only the
    tools of the listed topics are kept; tools outside every topic fall under
    ``"other"`` and are kept only when ``"other"`` is listed. Unknown topic
    names are logged once and ignored (a typo must not silently disable
    everything). Excluded tools cost zero tokens: no schema, no description.
    """
    if groups is None:
        return tools
    if isinstance(groups, str):
        names = _split_list(groups)
    else:
        names = [str(g).strip() for g in groups if str(g).strip()]
    if not names:
        return tools
    wanted = {name.lower() for name in names}
    if "all" in wanted:
        return tools
    unknown = wanted - set(_TOOL_TOPICS) - {"other"}
    if unknown:
        getLogger("Agent").info(
            "Unknown [agent] tool_groups entries ignored: %s",
            ", ".join(sorted(unknown)))
    name_to_topic = {
        name: topic
        for topic, topic_names in _TOOL_TOPICS.items() for name in topic_names
    }
    return [
        tool for tool in tools
        if name_to_topic.get(
            getattr(tool, "__name__", tool.__class__.__name__), "other")
        in wanted
    ]


# Topic grouping for the /tools launcher (session.py): tool name -> topic.
# Curated order = display order; tools missing from the map (e.g. MCP-injected
# or test doubles) fall under "other", listed last.
_TOOL_TOPICS: dict[str, tuple[str, ...]] = {
    "model": ("model_info",),
    "memory": ("recall_memory", "store_memory", "clear_chat", "compact_chat"),
    "speech": ("allow_speech",),
    "scheduling": ("schedule_task", "list_scheduled_tasks",
                   "cancel_scheduled_task", "agent_activity"),
    "documents": ("summarize_doc", "query_document", "list_corpora",
                  "store_corpus", "query_corpus"),
    "images": ("image_response", "capture_image", "image_search",
               "generate_image", "unload_image_generator"),
    "files": ("run_command", "search_in_file", "search_pattern_in_files",
              "read_text_file", "replace_string_in_file", "write_text_file",
              "download_file", "extract_archive", "zip_files"),
    "web": ("web_search", "web_images", "web_videos", "web_news", "web_books",
            "extract_webpage"),
    "finance": ("fetch_financial_data", "save_financial_chart"),
    "system": ("system_info", "current_date_time", "get_working_directory",
               "spawn_process", "exit_gracefully"),
    "mcp": ("mcp_list_tools", "mcp_call_tool"),
}


def tool_topics(tools: list) -> list[tuple[str, list]]:
    """Group a tool list by topic for display.

    Returns ``[(topic, [tools...]), ...]`` with topics in the curated
    ``_TOOL_TOPICS`` order (only non-empty ones), unknown tools collected
    under ``"other"`` last. Within a topic the input order is preserved, so
    the flattened groups reproduce the input order exactly (selection
    numbering in the /tools launcher stays stable).
    """
    name_to_topic = {
        name: topic
        for topic, names in _TOOL_TOPICS.items() for name in names
    }
    grouped: dict[str, list] = {}
    for tool in tools:
        name = getattr(tool, "__name__", tool.__class__.__name__)
        topic = name_to_topic.get(name, "other")
        grouped.setdefault(topic, []).append(tool)
    ordered = [t for t in _TOOL_TOPICS if t in grouped]
    if "other" in grouped:
        ordered.append("other")
    return [(topic, grouped[topic]) for topic in ordered]


def _split_list(value) -> list[str]:
    """Normalize a pool setting: comma-separated string or list -> list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


# Known [agent] monitor flags; unknown tokens are ignored, "none" clears all.
_MONITOR_FLAGS = frozenset({"calls", "log"})
_DEFAULT_MONITOR_LOG_FILE = "agents.jsonl"
_DEFAULT_MONITOR_BUFFER = 100

# Process-wide cache of server-reported model metadata, keyed by
# (base_url, model). The main agent and the scheduler execution agent usually boot the
# same models.json entry; without this each of them re-pays the metadata
# round-trips (OpenAI /models + native-API probes) at startup. Populated by
# Brain._fetch_model_metadata; /model switches need no invalidation (the key
# includes the model, and metadata is static per model).
_MODEL_META_CACHE: dict[tuple[str, str], dict] = {}


def _monitor_flags(value) -> set:
    """Parse ``[agent] monitor`` (string or list) into a set of known flags.

    ``"calls,log"`` / ``["calls", "log"]`` -> ``{"calls", "log"}``; unknown
    tokens are ignored; the token ``"none"`` (or an empty value) clears all
    flags, keeping the historical fully-quiet sub-agent behavior.
    """
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in (value or []) if str(item).strip()]
    flags: set = set()
    for item in items:
        if item == "none":
            return set()
        if item in _MONITOR_FLAGS:
            flags.add(item)
    return flags


def _monitor_buffer(value) -> int:
    """Resolve ``[agent] monitor_buffer``: int or numeric string, default 100.

    Invalid or non-positive values fall back to the default (a zero/negative
    buffer would make ``agent_activity`` useless, so it is not honored).
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_MONITOR_BUFFER
    return parsed if parsed > 0 else _DEFAULT_MONITOR_BUFFER


def monitor_settings(config: SimpleNamespace) -> dict:
    """Resolve the ``[agent]`` activity-monitor settings (kept for the scheduler).

    Returns ``{"monitor", "monitor_log_file", "monitor_buffer"}``:
    ``monitor`` is the set of monitoring flags (``"calls"``: labeled [CALL]
    blocks for sub-agent tool calls; ``"log"``: JSONL event log),
    ``monitor_log_file`` the JSONL log path and ``monitor_buffer`` the
    in-memory event ring size. Missing keys or a missing ``[agent]`` section
    keep the unmonitored behavior (``monitor = set()``).
    """
    agent = getattr(config, "agent", None)
    # load_config path-resolves keys ending in "_file" (relative to the config
    # dir); keep whatever arrives (str or Path) and default when unset.
    log_file = getattr(agent, "monitor_log_file", None)
    return {
        "monitor": _monitor_flags(getattr(agent, "monitor", None)),
        "monitor_log_file": str(log_file) if log_file else _DEFAULT_MONITOR_LOG_FILE,
        "monitor_buffer": _monitor_buffer(getattr(agent, "monitor_buffer", None)),
    }


def build_monitor(config: SimpleNamespace) -> "AgentMonitor | None":
    """Create and bind the process-wide monitor when monitoring is enabled.

    Called by ``main.py`` at startup (the pool no longer owns this). The JSONL
    sink only exists when ``"log"`` is among the flags; the ring buffer is
    always populated when monitoring is enabled. Returns the bound monitor, or
    ``None`` when monitoring is off (nothing is recorded, no log file written).
    """
    settings = monitor_settings(config)
    flags = settings["monitor"]
    if not flags:
        return None
    monitor = AgentMonitor(
        maxlen=settings["monitor_buffer"] or MONITOR_BUFFER_DEFAULT,
        log_file=settings["monitor_log_file"] if "log" in flags else None,
    )
    bind_monitor(monitor)
    return monitor


## sub-agent activity monitor (task 2) ##

# Thread-safe ring buffer of sub-agent activity events, filled by background
# agents (the scheduler executor) and drained by the UI (session poll /
# console tail / JSONL file). Kept at module level so the session and any
# watcher see the same stream; a module-level lock serializes appends from
# worker threads.
MONITOR_BUFFER_DEFAULT = 100


class AgentMonitor:
    """Thread-safe ring buffer of sub-agent activity events.

    Events are plain dicts (JSON-friendly): ``{"ts", "agent", "event", ...}``
    with ``event`` in ``{"start", "answer", "tool_call", "tool_result"}``.
    ``record()`` appends, trims to ``maxlen`` and mirrors every event to the
    optional JSONL file (``[agent] monitor_log_file``) - a failed write is
    logged once and the file sink is dropped (monitoring must never break the
    agents). ``snapshot()`` returns the most recent events oldest-first for
    the UI; ``clear()`` empties the buffer (the file sink is untouched).
    """

    def __init__(self, maxlen: int = MONITOR_BUFFER_DEFAULT, log_file: str | None = None):
        self.maxlen = max(1, int(maxlen or MONITOR_BUFFER_DEFAULT))
        self.log_file = log_file
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._log_failed = False
        self.logger = getLogger("AgentMonitor")

    def record(self, agent: str, event: str, **fields) -> None:
        """Append one event (thread-safe) and mirror it to the JSONL sink."""
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "agent": agent, "event": event, **fields}
        with self._lock:
            self._events.append(entry)
            if len(self._events) > self.maxlen:
                del self._events[: len(self._events) - self.maxlen]
            # The file write happens under the same lock: parallel worker
            # threads appending concurrently would otherwise interleave
            # partial lines and lose events in the JSONL sink.
            self._write_log(entry)

    def __len__(self) -> int:
        """Current number of buffered events."""
        with self._lock:
            return len(self._events)

    def _write_log(self, entry: dict) -> None:
        """Append one event as a JSON line; drop the sink after a failure."""
        if not self.log_file or self._log_failed:
            return
        try:
            path = pathlib.Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._log_failed = True
            self.logger.warning("Monitor log %s disabled after write error: %s",
                                self.log_file, exc)

    def snapshot(self, limit: int | None = None) -> list[dict]:
        """Most recent events, oldest-first (all when ``limit`` is None)."""
        with self._lock:
            events = list(self._events)
        return events[-limit:] if limit else events

    def clear(self) -> None:
        """Empty the buffer (the JSONL file sink is not touched)."""
        with self._lock:
            self._events.clear()

    _EVENT_ICONS = {"start": "🚀", "answer": "✅", "tool_call": "🔧", "tool_result": "↩"}

    def format_event(self, event: dict) -> str:
        """One human-readable console line for an event."""
        icon = self._EVENT_ICONS.get(event.get("event"), "•")
        agent = event.get("agent", "?")
        kind = event.get("event", "?")
        if kind == "start":
            return f"{icon} [{agent}] started: {str(event.get('prompt', ''))[:80]}"
        if kind == "answer":
            return f"{icon} [{agent}] answered: {str(event.get('answer', ''))[:80]}"
        if kind == "tool_call":
            return f"{icon} [{agent}] tool {event.get('tool', '?')}"
        if kind == "tool_result":
            return f"{icon} [{agent}] -> {str(event.get('result', ''))[:80]}"
        return f"{icon} [{agent}] {kind}"

    def tail(self, limit: int = 20) -> str:
        """The most recent events as formatted console lines (oldest-first)."""
        events = self.snapshot(limit=limit)
        if not events:
            return "No sub-agent activity recorded yet."
        return "\n".join(self.format_event(event) for event in events)


_MONITOR: AgentMonitor | None = None


def get_monitor() -> AgentMonitor | None:
    """The process-wide monitor (None until bind_monitor is called)."""
    return _MONITOR


def bind_monitor(monitor: AgentMonitor | None) -> None:
    """Install the process-wide monitor (called by main.py with the pool)."""
    global _MONITOR
    _MONITOR = monitor


## background results waiting for the main agent (task 1) ##

# Results produced by background workers (scheduled tasks) and waiting to be
# injected into the main agent's history. Entries are (kind, id, text) with
# kind "scheduled"; the session drains the queue at the start of every turn.
# Background threads only append under the lock; the main thread is the sole
# consumer (atomic swap), so history writes never race an in-flight tool-call
# round.
_BACKGROUND_RESULTS: list[tuple[str, str, str]] = []
_BACKGROUND_RESULTS_LOCK = threading.Lock()


def queue_agent_result(kind: str, result_id: str, text: str) -> None:
    """Queue one background result for main-agent delivery (thread-safe)."""
    with _BACKGROUND_RESULTS_LOCK:
        _BACKGROUND_RESULTS.append((kind, str(result_id), text))


def drain_agent_results() -> list[tuple[str, str, str]]:
    """Return and clear every pending background result (main thread only)."""
    with _BACKGROUND_RESULTS_LOCK:
        pending, _BACKGROUND_RESULTS[:] = _BACKGROUND_RESULTS[:], []
    return pending


def drain_agent_results_of_kind(kind: str) -> list[tuple[str, str]]:
    """Return and clear the pending results of one kind as (id, text) pairs."""
    with _BACKGROUND_RESULTS_LOCK:
        matching = [entry for entry in _BACKGROUND_RESULTS if entry[0] == kind]
        _BACKGROUND_RESULTS[:] = [
            entry for entry in _BACKGROUND_RESULTS if entry[0] != kind]
    return [(result_id, text) for _, result_id, text in matching]


# Per-thread name of the agent currently running a tool call ('' = main agent).
# The scheduler sets it to 'scheduler' so its executor agent's tool calls are
# reported to the activity monitor.
_TOOL_CONTEXT = threading.local()


def _current_agent_name() -> str:
    """Name of the agent currently executing a tool call ('' when main)."""
    return getattr(_TOOL_CONTEXT, "agent_name", "") or ""


def agent_activity(last: int = 20) -> str:
    """
    Report the recent sub-agent activity: the last `last` events
    (scheduled-task runs, tool calls, answers) as human-readable one-line
    strings, oldest first. Useful when the user asks what the scheduled tasks
    did or what is currently running.
    """
    monitor = get_monitor()
    if monitor is None:
        return "No sub-agent activity recorded."
    events = monitor.snapshot(limit=max(1, int(last or 20)))
    if not events:
        return "No sub-agent activity recorded."
    return "\n".join(monitor.format_event(event) for event in events)


def _load_env_file(dotenv_path: pathlib.Path | None = None) -> None:
    """Load ``.env`` from the project root (next to ``main.py``).

    Optional convenience: only active when ``python-dotenv`` is installed, so
    secrets can live in a ``.env`` file instead of the shell. Existing
    environment variables always win over file entries (``override=False``),
    and the file is never required.
    """
    if dotenv_path is None:
        dotenv_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


class OpenAICore:
    """
    Core client for an OpenAI-compatible chat API (API-key providers).

    Talks to an OpenAI-compatible endpoint with the ``openai`` client, so
    cloud models (OpenAI, Mistral, Groq, OpenRouter, DeepSeek, Together,
    Gemini, xAI, Anthropic, ...) and local servers (Ollama, LM Studio) share
    one implementation: connection handling, model metadata resolution
    (context window, vision) from the OpenAI models endpoint - plus the
    native APIs of local servers, whose OpenAI /models data is thin - and the
    function-calling helpers (JSON Schema tool definitions, stream chunk
    parsing).
    The connection comes from the models JSON file (``[brain] models_file``):
    its first entry boots the session and ``/model`` switches between the
    entries at runtime; without a models file the legacy ``[brain]``
    model/base_url/api_key_env keys are used. The API key is read from the
    environment variable named by the entry's ``api_key_env`` (also resolved
    from a project ``.env`` file) and must never be stored in config.toml or
    the models file.
    """

    DEFAULT_CONTEXT_WINDOW = 128_000

    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        try:
            from openai import OpenAI  # lazy import: keeps startup light
        except ImportError as exc:
            raise ImportError("pip install openai  # required for the OpenAI-based Brain") from exc

        model, base_url, api_key = self._boot_connection()

        self.logger.info("🚀 Connecting to remote model '%s' @ %s ...", model, base_url)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model
        self._base_url = base_url
        self._context_window = self._resolve_context_window(config)
        self._vision = self._resolve_vision_flag(config)
        self._temperature = getattr(config.brain, "temperature", 0.7)
        self._max_tokens = _response_token_cap(
            getattr(config.brain, "max_response_tokens", 0),
            self._context_window,
        )
        # Chain-of-thought visualization ([ui] show_thinking/max_thinking_tokens).
        self._show_thinking = bool(getattr(config.ui, "show_thinking", False))
        self._max_thinking_tokens = _thinking_token_cap(
            getattr(config.ui, "max_thinking_tokens", 0))
        self._reset_thinking()
        self._messages: list[dict] = [{"role": "system", "content": config.brain.system_prompt}]
        ui = getattr(config, "ui", None)
        if getattr(self, "quiet", False):
            # Quiet agents (e.g. the scheduler execution agent) buffer fragments instead of printing:
            # parallel act() calls must not interleave on the console.
            self._stream = NullPrinter()
        else:
            self._stream = StreamPrinter(
                text_color=_validated_style(getattr(ui, "assistant_color", None)),
                code_theme=_validated_code_theme(getattr(ui, "code_theme", None)),
            )
        self._cancel_requested = False
        self._active_stream = None
        # Session-bound tool modules: their config comes from main.py via
        # src.tools.configure; the live state is bound here (getter first:
        # the messages list is re-read on every access, so clearing or
        # reassigning the session history stays visible to the tools).
        # Isolated agents (e.g. the scheduler execution agent) skip the module-default binding: they get
        # their own tool instances in Agent._build_tools, and binding here
        # would hijack the main agent's module-default tools.
        from src.tools import memory as memory_tools, vision as vision_tools
        if not getattr(self, "isolated_tools", False):
            memory_tools.bind_history(lambda: self._messages)
            memory_tools.bind_summarizer(self._summarize_messages)
            vision_tools.bind_session(self)
        self.logger.info("✅ Remote model '%s' connected.", self._model_name)

    ## models file: boot connection + /model switching ##

    def _models_file(self) -> pathlib.Path | None:
        """Models JSON file from ``[brain] models_file`` (None when unset)."""
        value = getattr(self.config.brain, "models_file", None)
        return pathlib.Path(value) if value else None

    def _load_model_entries(self) -> dict[str, dict]:
        """Read the models JSON file into ``{name: entry}`` (file order kept).

        Accepted shapes: a ``name -> entry`` mapping, a list of entries each
        carrying a ``name``, and both wrapped in a top-level ``"models"`` key.
        A bare string stands for a model id selectable under its own name.
        Every usable entry needs a ``model`` id; anything else is skipped.
        """
        path = self._models_file()
        if path is None:
            raise ValueError(
                "No models file configured: set [brain] models_file in config.toml")
        if not path.is_file():
            raise FileNotFoundError(f"Models file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in models file {path}: {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("models"), (list, dict)):
            data = data["models"]
        if isinstance(data, dict):
            pairs = list(data.items())
        elif isinstance(data, list):
            pairs = []
            for entry in data:
                if isinstance(entry, dict):
                    pairs.append((entry.get("name"), entry))
                else:  # bare model id: selectable under its own name
                    text = str(entry).strip()
                    pairs.append((text, {"model": text}))
        else:
            raise ValueError(
                f"Models file {path} must hold a name->entry mapping or a list of entries")
        entries: dict[str, dict] = {}
        for name, entry in pairs:
            if isinstance(entry, str):  # shorthand mapping value: {"alias": "model-id"}
                entry = {"model": entry.strip()}
            name = str(name or "").strip()
            if name and isinstance(entry, dict) and str(entry.get("model") or "").strip():
                entries[name] = entry
        if not entries:
            raise ValueError(
                f"No usable model entries in models file {path} "
                "(every entry needs a name and a 'model' id)")
        return entries

    def list_models(self) -> dict[str, dict]:
        """Selectable models from the models JSON file (``{name: entry}``)."""
        return self._load_model_entries()

    def _entry_connection(self, display_name: str, entry: dict) -> tuple[str, str, str]:
        """Connection triple ``(model, base_url, api_key)`` for a models entry.

        Entry values win; missing ``base_url``/``api_key_env`` fall back to the
        ``[brain]`` config (legacy setups keep them there). The API key is
        resolved from the environment (``.env`` included) exactly as at
        startup and is never stored in the JSON file.
        """
        brain = self.config.brain
        model = str(entry.get("model") or "").strip()
        base_url = str(entry.get("base_url")
                       or getattr(brain, "base_url", "") or "").strip()
        api_key_env = str(entry.get("api_key_env")
                          or getattr(brain, "api_key_env", "") or "").strip()
        if not model:
            raise ValueError(f"Model entry {display_name!r} has no 'model' id")
        if not base_url:
            raise ValueError(
                f"Model entry {display_name!r} has no 'base_url' and none is configured")
        if not api_key_env:
            raise ValueError(
                f"Model entry {display_name!r} has no 'api_key_env' and none is configured")
        _load_env_file()  # fill unset variables from .env (real env vars win)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found for model {display_name!r}: set the '{api_key_env}' "
                "environment variable, e.g. put "
                f"'{api_key_env}=sk-...' in the project .env file "
                f"or run: setx {api_key_env} 'sk-...'")
        return model, base_url, api_key

    def _boot_connection(self) -> tuple[str, str, str]:
        """Resolve the startup connection: ``(model, base_url, api_key)``.

        With ``[brain] models_file`` configured, the **first entry** of the
        models JSON file boots the session: the startup model is always one of
        the ``/model``-switchable entries and ``[brain]`` carries no duplicate
        ``model``/``base_url``/``api_key_env``. Without a models file the
        legacy ``[brain]`` keys are used directly.
        """
        brain = self.config.brain
        if getattr(brain, "models_file", None):
            entries = self._load_model_entries()
            display_name, entry = next(iter(entries.items()))
            self._model_entry = display_name
            self.logger.info("Boot model: first entry '%s' of the models file.",
                             display_name)
            return self._entry_connection(display_name, entry)
        model = str(getattr(brain, "model", "") or "").strip()
        base_url = str(getattr(brain, "base_url", "") or "").strip()
        api_key_env = str(getattr(brain, "api_key_env", "") or "").strip()
        if not base_url:
            raise ValueError("Missing [brain] base_url")
        if not model:
            raise ValueError("Missing [brain] model")
        _load_env_file()  # fill unset variables from .env (real env vars win)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found: set the '{api_key_env}' environment variable, "
                f"e.g. put '{api_key_env}=sk-...' in the project .env file "
                f"or run: setx {api_key_env} 'sk-...'")
        return model, base_url, api_key

    def select_model(self, name: str) -> str:
        """Switch the live session to a model entry from the models JSON file.

        Every entry carries ``model`` (the model id) plus optional
        ``api_key_env`` and ``base_url``; missing connection fields keep the
        values currently configured under ``[brain]``. The API key is resolved
        from the environment (``.env`` included) exactly as at startup and is
        never stored in the JSON file. On success the client is rebuilt for
        the new endpoint, the cached model metadata is dropped and the context
        window / vision support are re-resolved from the new server (the
        per-generation token cap follows the new window). The conversation
        history is kept. Any failure raises before the current model is
        touched; success returns a human-readable confirmation.
        """
        entries = self._load_model_entries()
        lookup = {key.lower(): (key, entry) for key, entry in entries.items()}
        match = lookup.get(str(name or "").strip().lower())
        if match is None:
            raise KeyError(
                f"Unknown model {name!r} in the models file. Available: "
                + ", ".join(entries))
        display_name, entry = match
        model, base_url, api_key = self._entry_connection(display_name, entry)

        self.logger.info("🔄 Switching to model '%s' @ %s ...", model, base_url)
        from openai import OpenAI  # installed: __init__ already imported it
        client = OpenAI(base_url=base_url, api_key=api_key)

        # Validation passed: swap the connection, drop the cached metadata and
        # re-resolve the model-dependent state from the new server.
        self._client = client
        self._model_name = model
        self._base_url = base_url
        self._model_entry = display_name
        self.__dict__.pop("_model_meta", None)
        self._context_window = self._resolve_context_window(self.config)
        self._vision = self._resolve_vision_flag(self.config)
        self._max_tokens = _response_token_cap(
            getattr(self.config.brain, "max_response_tokens", 0), self._context_window)
        self.logger.info(
            "✅ Model '%s' is now active (context %d, vision %s).",
            model, self._context_window, "yes" if self._vision else "no",
        )
        return (f"Model switched to {display_name}: {model} @ {base_url} "
                f"(context window {self._context_window}, "
                f"vision {'yes' if self._vision else 'no'})")

    def _resolve_vision_flag(self, config) -> bool:
        """Derive vision support from the server-reported model capabilities.

        The OpenAI models endpoint may include a ``capabilities`` list (e.g.
        LM Studio: ["completion", "tools", "vision"]); when the server
        reports it, that value is authoritative. Local servers are covered by
        the native-API fallback in ``_fetch_model_metadata``; servers still
        reporting nothing (most cloud providers) fall back to the optional
        ``[brain] vision`` config flag (default False), so the key stays
        usable as an override but is no longer required.
        """
        configured = bool(getattr(config.brain, "vision", False))
        capabilities = self._fetch_model_metadata().get("capabilities")
        if not isinstance(capabilities, (list, tuple)) or not capabilities:
            self._vision_source = "config"
            self.logger.info(
                "Model '%s' vision: %s (config; server did not report capabilities)",
                self._model_name, configured,
            )
            return configured
        self._vision_source = "server"
        has_vision = "vision" in [str(c).lower() for c in capabilities]
        self.logger.info(
            "Model '%s' capabilities (server): %s -> vision=%s",
            self._model_name, ", ".join(map(str, capabilities)), has_vision,
        )
        return has_vision

    def _resolve_context_window(self, config) -> int:
        """Resolve the context window: explicit config > server report > default.

        ``[brain] context_window`` stays authoritative when set. When unset
        (0 or missing), the window is derived from the server-reported model
        metadata via the OpenAI /models endpoint - e.g. LM Studio's
        ``max_context_length`` or Together's ``context_length`` - preferring
        explicit maximums over the currently loaded allocation
        (``loaded_context_length``). Servers reporting nothing (e.g. Ollama)
        fall back to ``DEFAULT_CONTEXT_WINDOW``.
        """
        try:
            configured = int(getattr(config.brain, "context_window", 0) or 0)
        except (TypeError, ValueError):
            configured = 0
        if configured:
            self._context_window_source = "config"
            return configured
        meta = self._fetch_model_metadata()
        context_fields = {
            k: v for k, v in meta.items()
            if "context" in k.lower() and isinstance(v, int)
        }
        for key in self._CONTEXT_MAX_KEYS:
            if key in context_fields:
                return self._adopt_server_context_window(key, context_fields[key])
        for key in sorted(context_fields):
            if not key.lower().startswith("loaded"):
                return self._adopt_server_context_window(key, context_fields[key])
        self._context_window_source = "default"
        self.logger.info(
            "Model '%s' context window: default %d (server did not report one)",
            self._model_name, self.DEFAULT_CONTEXT_WINDOW,
        )
        return self.DEFAULT_CONTEXT_WINDOW

    def _adopt_server_context_window(self, key: str, value: int) -> int:
        """Adopt a server-reported context length and log where it came from."""
        self._context_window_source = "server"
        self.logger.info(
            "Model '%s' context window from server (%s): %d",
            self._model_name, key, value,
        )
        return value

    _MODEL_RECORD_SKIP = {"id", "object", "root", "parent", "permission"}

    # Server metadata keys describing the model's maximum context, in the
    # order they are preferred when several are reported.
    _CONTEXT_MAX_KEYS = (
        "max_context_length", "context_length", "context_window", "max_model_len",
    )

    @staticmethod
    def _model_record_fields(record) -> dict:
        """Flatten an OpenAI SDK model record into a plain dict.

        Accepts pydantic records (``model_dump()``, used by the real SDK
        client), plain dicts (raw fallback) and SimpleNamespace-based fakes
        (``__dict__``, tests), so server-specific extra fields survive the
        round-trip.
        """
        if isinstance(record, dict):
            return dict(record)
        dump = getattr(record, "model_dump", None)
        data = dump() if callable(dump) else None
        if not isinstance(data, dict) or not data:
            data = dict(getattr(record, "__dict__", {}) or {})
        return data

    def _fetch_model_metadata(self, use_raw_fallback: bool = True) -> dict:
        """Query the server for model metadata via the OpenAI models endpoint.

        Only OpenAI-style requests are used: ``GET /models/{id}`` first, then
        ``GET /models`` + lookup by id as fallback (some servers, e.g. Ollama,
        only implement the list variant). The standard fields are id/object/
        created/owned_by; any extra fields a server returns beyond those (e.g.
        LM Studio: max_context_length, loaded_context_length, capabilities,
        quantization, arch) are passed through untouched.

        When the OpenAI endpoint reports neither capabilities nor context
        lengths (true for every local server), the server's native API is
        probed as well (see ``_fetch_native_model_metadata``) and fills the
        missing keys (existing keys are never overwritten).

        ``use_raw_fallback=False`` skips the tolerant re-request for servers
        returning a bare JSON array. Failures degrade silently. Successful
        results are cached per instance *and* process-wide (keyed by
        ``base_url + model``): the main agent and the scheduler execution agent usually
        boot the same model entry, and without the shared cache each of them
        pays the metadata round-trips again at startup. A total failure stores
        no cache so a later call can retry. Returns a dict of non-empty
        scalar/list-of-scalar values, possibly {}.
        """
        cached = getattr(self, "_model_meta", None)
        if cached is not None:
            return cached
        cache_key = (self._base_url, self._model_name)
        shared = _MODEL_META_CACHE.get(cache_key)
        if shared is not None:
            self._model_meta = shared
            return shared
        record = None
        try:
            record = self._client.models.retrieve(self._model_name)
        except Exception as exc:
            self.logger.debug("GET /models/%s failed: %s", self._model_name, exc)
        if record is None:
            try:
                models = self._client.models.list().data
                record = next((m for m in models if m.id == self._model_name), None)
            except Exception as exc:
                self.logger.debug("GET /models failed: %s", exc)
                if use_raw_fallback:
                    record = self._find_in_raw_models_list()
        meta: dict = {}
        if record is not None:
            for key, value in self._model_record_fields(record).items():
                if key in self._MODEL_RECORD_SKIP or value in (None, ""):
                    continue
                if isinstance(value, (list, tuple)):
                    # Keep lists of scalars (capabilities); drop object lists
                    # (OpenAI's permission entries) - model info stays flat.
                    if not value or not all(
                        isinstance(v, (str, int, float, bool)) for v in value
                    ):
                        continue
                elif not isinstance(value, (str, int, float, bool)):
                    continue  # skip nested structures
                meta[key] = value
        if not meta.get("capabilities") or not any(
                "context" in k.lower() for k in meta):
            # Local servers report neither through the OpenAI endpoint; fill
            # the gaps from their native APIs (no-op for cloud hosts).
            for key, value in self._fetch_native_model_metadata().items():
                meta.setdefault(key, value)
        if meta:
            self._model_meta = meta
            _MODEL_META_CACHE[cache_key] = meta
        return meta

    def _find_in_raw_models_list(self):
        """Lookup fallback for servers the OpenAI SDK cannot parse.

        Some OpenAI-compatible servers (e.g. Together.ai) answer ``GET
        /models`` with a bare JSON array instead of the standard
        ``{"object": "list", "data": [...]}`` envelope, which makes the SDK
        fail while parsing. This retries the same endpoint with the SDK's own
        httpx client and accepts both shapes. Returns the matching record
        (plain dict) or None.
        """
        try:
            import httpx
            resp = httpx.get(
                self._base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {self._client.api_key}"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self.logger.debug("Raw GET /models failed: %s", exc)
            return None
        if isinstance(data, dict):
            data = data.get("data") or []
        return next(
            (m for m in data if isinstance(m, dict) and m.get("id") == self._model_name),
            None,
        )

    ## native-API metadata (local servers) ##

    _LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

    def _server_root(self) -> str:
        """The server root URL: the OpenAI base_url without a trailing '/v1'."""
        base = self._base_url.rstrip("/")
        return base[:-3] if base.endswith("/v1") else base

    def _fetch_native_model_metadata(self) -> dict:
        """Model metadata from a local server's native API.

        The OpenAI ``/models`` endpoints of local servers are thin (no
        capabilities, no context lengths), while their native APIs carry the
        full picture: Ollama (``POST /api/show`` + ``GET /api/ps``) and LM
        Studio (``GET /api/v0/models`` - ``type == "vlm"`` adds a "vision"
        capability, the REST capability list only covers tool use). Tries
        Ollama first, then LM Studio; probes only loopback hosts and returns
        {} when neither server answers, so cloud providers are unaffected and
        callers degrade to the OpenAI-endpoint data.
        """
        if urlsplit(self._base_url).hostname not in self._LOOPBACK_HOSTS:
            return {}  # the native APIs are a local-server thing
        for fetcher in (self._ollama_native_metadata, self._lmstudio_native_metadata):
            meta = fetcher()
            if meta:
                return meta
        return {}

    def _ollama_native_metadata(self) -> dict:
        """Ollama-native metadata for the model ({} when not an Ollama server)."""
        try:
            import httpx
            resp = httpx.post(
                self._server_root() + "/api/show",
                json={"model": self._model_name},
                # local servers need no auth; tolerate clients without api_key
                headers={"Authorization": f"Bearer {getattr(self._client, 'api_key', '')}"},
                timeout=5.0,
            )
            resp.raise_for_status()
            shown = resp.json()
            if not isinstance(shown, dict):
                return {}
        except Exception as exc:
            self.logger.debug("GET /api/show failed: %s", exc)
            return {}
        meta: dict = {}
        capabilities = shown.get("capabilities")
        if isinstance(capabilities, (list, tuple)) and capabilities:
            meta["capabilities"] = [str(c) for c in capabilities]
        modelinfo = shown.get("model_info")
        if isinstance(modelinfo, dict):
            max_ctx = self._ollama_context_length(modelinfo)
            if max_ctx:
                meta["max_context_length"] = max_ctx
        loaded_ctx = self._ollama_loaded_context()
        if loaded_ctx:
            meta["loaded_context_length"] = loaded_ctx
        return meta

    @staticmethod
    def _ollama_context_length(modelinfo: dict) -> int | None:
        """Trained maximum context from Ollama's modelinfo blob.

        Stored as ``<architecture>.context_length``, with the architecture
        also under ``general.architecture`` (mirrors src.local_model_handler).
        """
        arch = modelinfo.get("general.architecture")
        if arch and isinstance(modelinfo.get(f"{arch}.context_length"), int):
            return int(modelinfo[f"{arch}.context_length"])
        for key, value in modelinfo.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return int(value)
        return None

    def _ollama_loaded_context(self) -> int | None:
        """The model's currently allocated context (``GET /api/ps``), if loaded."""
        try:
            import httpx
            resp = httpx.get(self._server_root() + "/api/ps", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models") or []
        except Exception as exc:
            self.logger.debug("GET /api/ps failed: %s", exc)
            return None
        for entry in models:
            if isinstance(entry, dict) and self._model_name in (
                    entry.get("model"), entry.get("name")):
                if isinstance(entry.get("context_length"), int):
                    return entry["context_length"]
        return None

    def _lmstudio_native_metadata(self) -> dict:
        """LM Studio-native metadata for the model ({} when not LM Studio)."""
        try:
            import httpx
            resp = httpx.get(self._server_root() + "/api/v0/models", timeout=5.0)
            resp.raise_for_status()
            records = resp.json().get("data") or []
        except Exception as exc:
            self.logger.debug("GET /api/v0/models failed: %s", exc)
            return {}
        record = next(
            (r for r in records
             if isinstance(r, dict) and r.get("id") == self._model_name),
            None,
        )
        if record is None:
            return {}
        meta: dict = {}
        for key in ("type", "publisher", "arch", "quantization", "state",
                    "max_context_length", "loaded_context_length"):
            value = record.get(key)
            if value not in (None, ""):
                meta[key] = value
        capabilities = [str(c) for c in (record.get("capabilities") or [])]
        if record.get("type") == "vlm" and "vision" not in capabilities:
            capabilities.append("vision")  # LM Studio marks vision models by type
        if capabilities:
            meta["capabilities"] = capabilities
        return meta

    ## OpenAI function-calling helpers (tool definitions, stream chunks) ##

    _JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}

    @staticmethod
    def _json_schema(annotation: Any) -> dict:
        """Best-effort JSON Schema for a Python type annotation.

        ``Annotated[T, "description"]`` maps the string metadata to the schema's
        ``description``, so tools can document parameters for the model.
        """
        metadata = getattr(annotation, "__metadata__", None)  # typing.Annotated
        if metadata:
            schema = OpenAICore._json_schema(get_args(annotation)[0])
            for meta in metadata:
                if isinstance(meta, str):
                    schema["description"] = meta
                    break
            return schema
        origin = get_origin(annotation)
        if origin is Literal:
            return {"type": "string", "enum": list(get_args(annotation))}
        if origin in (list, set, tuple):
            args = get_args(annotation)
            return {"type": "array", "items": OpenAICore._json_schema(args[0]) if args else {}}
        if annotation is dict or origin is dict:
            # A bare ``dict`` annotation has no origin (get_origin(dict) is
            # None); without this check it fell through to {"type": "string"},
            # telling models that e.g. schedule_task's task objects were
            # plain strings.
            return {"type": "object"}
        if origin is Union or origin is types.UnionType:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            return OpenAICore._json_schema(args[0]) if len(args) == 1 else {"type": "string"}
        json_type = OpenAICore._JSON_TYPES.get(annotation)
        return {"type": json_type} if json_type else {"type": "string"}

    @staticmethod
    def _openai_tool_definition(tool: Callable, max_doc_chars: int = 0) -> dict:
        """Describe ``tool`` in OpenAI function-calling (JSON Schema) format.

        Parameter names keep their snake_case spelling: OpenAI-compatible models
        handle it natively, and the arguments come back as a JSON blob that can be
        passed to the implementation directly.

        ``max_doc_chars > 0`` truncates the tool description to that many chars
        (token-reduction: the full docstring is sent with every request).
        ``default`` values are never included in the schemas: they are
        informational only (models never need them) and 74 defaults across the
        default tool set cost ~1.5k chars per request.
        """
        signature = inspect.signature(tool)
        try:
            hints = get_type_hints(tool, include_extras=True)
        except Exception:
            hints = getattr(tool, "__annotations__", {}) or {}

        description = (getattr(tool, "__doc__", "") or "").strip()
        if max_doc_chars and max_doc_chars > 0:
            description = description[:max_doc_chars]

        properties: dict = {}
        required: list[str] = []
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema = OpenAICore._json_schema(hints.get(name, str))
            if param.default is inspect.Parameter.empty:
                required.append(name)
            properties[name] = schema
        return {
            "type": "function",
            "function": {
                "name": getattr(tool, "__name__", "tool"),
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    @staticmethod
    def _stream_delta(chunk: Any):
        """First choice's delta of an OpenAI-compatible stream chunk (or None)."""
        choices = getattr(chunk, "choices", None)
        return choices[0].delta if choices else None


class Brain(OpenAICore):
    """
    Tool-bearing assistant built on top of OpenAICore.

    Owns the conversation history (plumbing and repair across cancelled or
    failed turns), streaming generation with cancellation, and the model
    info tool. All other LLM tools live in the ``src.tools`` submodules:
    their config is injected once by ``main.py`` via ``src.tools.configure``,
    and their session-bound state (chat history, LLM client) is bound to
    this Brain instance in ``__init__`` via the ``bind_*`` functions.
    """

    ## history plumbing (plain OpenAI-style dicts) ##

    def _add_user_prompt(self, prompt: str):
        self._repair_history()
        self._messages.append({"role": "user", "content": prompt})
        self.logger.debug("Prompt:\n%s", prompt)

    def _repair_history(self) -> None:
        """Keep the chat history template-valid across cancelled/failed turns.

        Prompt templates require user/assistant roles to alternate (tool-call
        /result sequences excepted). Two failure modes break that:

        * a Ctrl+C-cancelled generation can end in an *empty* assistant
          message; templates drop empty turns, which leaves a tool result
          directly followed by a user message and fails rendering with
          "conversation roles must alternate ...";
        * a turn that errored or was cancelled before any response leaves the
          user prompt unanswered, so the next prompt would render as two
          consecutive user messages.

        Called before each new prompt: pad the empty assistant turn with a
        minimal placeholder and drop a trailing unanswered user prompt.
        """
        messages = self._messages
        while messages:
            role = messages[-1].get("role")
            if role == "assistant":
                if not str(messages[-1].get("content") or "").strip():
                    messages.pop()
                    messages.append({"role": "assistant", "content": CANCELLED_PLACEHOLDER})
                break
            if role == "user":
                messages.pop()  # unanswered prompt from a failed/aborted turn
                continue
            break

    ## context pruning ([brain] context_trim_ratio) ##

    @staticmethod
    def _next_prunable_group_end(messages: list[dict], start: int,
                                 keep: int) -> int | None:
        """End index (exclusive) of the oldest prunable group from ``start``.

        A group is either a *complete* tool round (one assistant message
        carrying ``tool_calls`` plus every following ``role == "tool"``
        message with a matching ``tool_call_id`` - never split, servers
        reject orphan tool results) or a single plain user/assistant
        message. Nothing at or after ``len(messages) - keep`` (the
        keep-window) may be dropped. A leading orphan ``tool`` message
        (already template-invalid on its own) is dropped alone to unblock
        pruning. Returns None when nothing can be dropped.
        """
        limit = len(messages) - keep  # exclusive droppable boundary
        if start >= limit:
            return None
        first = messages[start]
        role = first.get("role")
        if role == "assistant" and first.get("tool_calls"):
            call_ids = {tc.get("id") for tc in first["tool_calls"]}
            end = start + 1
            while (end < len(messages)
                   and messages[end].get("role") == "tool"
                   and messages[end].get("tool_call_id") in call_ids):
                end += 1
            return end if end <= limit else None
        if role == "tool":
            return start + 1  # orphan tool result: drop it to unblock
        if role in ("user", "assistant", "system"):
            return start + 1
        return None

    def _prune_history(self) -> int:
        """Drop the oldest history when context usage exceeds the ratio.

        Called at the start of every ``act()`` turn. ``[brain]
        context_trim_ratio`` triggers pruning when the approximate usage
        (``_context_usage``) passes it; oldest complete groups are dropped
        (see ``_next_prunable_group_end``) until usage is back under the
        ratio. The system prompt (index 0) and the last ``[brain]
        context_keep_messages`` messages are never touched, so the live
        exchange stays intact. A one-line marker is appended to the system
        prompt (never a mid-history system message) so the model knows
        earlier context was compacted. Returns the dropped message count
        (0 = nothing pruned). 0/unset ratio disables pruning entirely.

        With ``[brain] auto_compaction = true``, the same usage checkpoint
        first runs a summarizing compaction (``src.tools.memory.compact_chat``
        keeping ``[brain] compact_keep_messages`` recent turns) instead of the
        plain drop; legacy pruning still runs afterwards if usage remains
        above the ratio, so the loop cannot spin.
        """
        ratio = getattr(self, "_context_trim_ratio", 0.0)
        if not ratio:
            return 0
        keep = getattr(self, "_context_keep_messages", 40)
        messages = self._messages
        dropped_total = 0
        if getattr(self, "_auto_compaction", False):
            usage_pct, _window = self._context_usage()
            if usage_pct > ratio * 100:
                compacted = self._auto_compact()
                if compacted:
                    self.logger.info(
                        "Context auto-compacted: %s", compacted)
                    usage_pct, _window = self._context_usage()
        while True:
            usage_pct, _window = self._context_usage()
            if usage_pct <= ratio * 100:
                break
            end = self._next_prunable_group_end(messages, 1, keep)
            if end is None:
                break
            del messages[1:end]
            dropped_total += end - 1
        if dropped_total:
            system = next(
                (m for m in messages if m.get("role") == "system"), None)
            if system is not None:
                base = str(system.get("content") or "")
                cut = base.find("\n[history pruned:")
                if cut != -1:  # replace a marker from an earlier prune
                    base = base[:cut]
                system["content"] = (
                    base + f"\n[history pruned: {dropped_total} messages dropped]")
        return dropped_total

    def _auto_compact(self) -> str:
        """One automatic compaction via the memory tool (auto_compaction).

        Calls the module-default ``compact_chat`` with the configured
        ``[brain] compact_keep_messages``. Never raises: any failure returns
        an empty string so the caller falls back to the plain pruning path.
        """
        try:
            from src.tools import memory as memory_tools
            keep = getattr(self, "_compact_keep_messages", 40)
            result = memory_tools.compact_chat(keep)
            return result if not result.startswith("Error:") else ""
        except Exception as e:
            self.logger.warning("Auto-compaction failed: %s", e)
            return ""

    ## session view for the session-bound tool modules (src.tools.vision) ##

    @property
    def client(self):
        """OpenAI-compatible chat client (read-only session view)."""
        return self._client

    @property
    def model_name(self) -> str:
        """Configured model id (read-only session view)."""
        return self._model_name

    @property
    def temperature(self) -> float:
        """Configured sampling temperature (read-only session view)."""
        return self._temperature

    @property
    def vision(self) -> bool:
        """Whether the model supports image input (read-only session view)."""
        return self._vision

    @property
    def vision_source(self) -> str:
        """Where the vision flag came from: 'server' or 'config'."""
        return self._vision_source

    @property
    def messages(self) -> list[dict]:
        """The live conversation history (tools may append/pop entries)."""
        return self._messages

    ## generation ##

    @property
    def cancel_requested(self) -> bool:
        """True while a user cancel (Ctrl+C) is pending for the current turn."""
        return self._cancel_requested

    def _stream_completion(self, tools: list[dict] | None = None):
        kwargs: dict = {
            "model": self._model_name,
            "messages": self._messages,
            "stream": True,
            "temperature": self._temperature,
            # Legacy name, still accepted by all OpenAI-compatible endpoints.
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        self._stream_options_ok = True
        try:
            return self._client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True})
        except TypeError:
            # Very old SDK without stream_options support.
            self._stream_options_ok = False
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._stream_options_ok and "stream_options" in str(exc):
                # Server rejected the option (older OpenAI-compatible
                # endpoints): fall back once and remember the refusal.
                self._stream_options_ok = False
                self.logger.debug(
                    "stream_options rejected by the server; retrying "
                    "without it (%s)", exc)
                return self._client.chat.completions.create(**kwargs)
            raise

    ## LLM-backed summarization for history compaction (src.tools.memory) ##

    _SUMMARIZER_SYSTEM_PROMPT = (
        "Summarize this conversation concisely: key facts, decisions, user "
        "preferences, open tasks, and any file paths or resources mentioned. "
        "Write only the summary."
    )

    def _summarize_messages(self, messages: list[dict]) -> str:
        """Summarize messages via the chat model (for history compaction).

        Non-interactive one-shot call with tool use disabled. Returns an empty
        string on any failure so the caller falls back to the extractive
        summary instead of failing the turn.
        """
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system",
                     "content": self._SUMMARIZER_SYSTEM_PROMPT},
                    *messages,
                ],
                temperature=0.2,
                max_tokens=500,
            )
            return str(resp.choices[0].message.content or "")
        except Exception as e:
            self.logger.warning("Compaction summarizer failed: %s", e)
            return ""

    ## chain-of-thought visualization ([ui] show_thinking) ##

    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"

    def _reset_thinking(self) -> None:
        """Reset the per-turn chain-of-thought display state.

        Called once per turn (``model_fragments``/``act``): the char counter
        and truncation flag restart, and the ``ine``-tag machine begins
        outside a thinking block with no held-back answer text.
        """
        self._thinking_chars = 0
        self._thinking_truncated = False
        self._thinking_header_shown = False
        self._in_think_tag = False
        self._pending_answer = ""

    @staticmethod
    def _reasoning_delta(delta: Any) -> str:
        """Reasoning text of a stream chunk across the common delta fields.

        OpenAI-compatible endpoints disagree on the field name: DeepSeek-style
        servers (LM Studio, Together) use ``reasoning_content`` and
        OpenRouter uses ``reasoning``. Empty string when the chunk carries no
        reasoning.
        """
        for field in ("reasoning_content", "reasoning"):
            text = getattr(delta, field, None)
            if text:
                return text
        return ""

    def _split_thinking(self, content: str) -> tuple[str, str]:
        """Split a content chunk into (thinking, answer) around `istine` tags.

        Some endpoints stream the chain of thought inline inside
        `istine`...`ine` (raw Qwen3, LM Studio with reasoning parsing off)
        instead of a dedicated reasoning delta field. Tags may split across
        chunks: a trailing partial tag is held back in ``_pending_answer``
        (prefixed to the next chunk) and :meth:`_flush_pending_answer` emits
        it when the stream ends.
        """
        text = self._pending_answer + content
        self._pending_answer = ""
        thinking: list[str] = []
        answer: list[str] = []
        while text:
            # Inside a think block we scan for the closing tag, outside for
            # the opening one.
            tag = self._THINK_CLOSE if self._in_think_tag else self._THINK_OPEN
            idx = text.find(tag)
            if idx == -1:
                # Hold back a trailing partial tag (e.g. "<thi" outside or
                # "</thi" inside a block) so a tag split across chunk
                # boundaries cannot leak into the wrong stream.
                held = ""
                for k in range(len(tag) - 1, 0, -1):
                    if text.endswith(tag[:k]):
                        held = text[-k:]
                        break
                body = text[: len(text) - len(held)]
                (thinking if self._in_think_tag else answer).append(body)
                self._pending_answer = held
                break
            (thinking if self._in_think_tag else answer).append(text[:idx])
            text = text[idx + len(tag):]
            self._in_think_tag = not self._in_think_tag
        return "".join(thinking), "".join(answer)

    def _flush_pending_answer(self) -> str:
        """Emit a held-back partial-tag tail when the stream ends."""
        pending, self._pending_answer = self._pending_answer, ""
        return pending

    def _feed_thinking(self, text: str) -> None:
        """Display one chain-of-thought chunk, honoring the token cap.

        ``[ui] max_thinking_tokens`` bounds what is *visualized* per
        generation round (the ~4 chars/token estimate matches
        ``_context_usage``): once the cap is hit a truncation marker is shown
        once and further thinking chunks of the round are dropped silently -
        the model itself keeps thinking. The display state resets per round
        (see ``_stream_round``), so every round gets a fresh budget.
        """
        if not text:
            return
        if not self._thinking_header_shown:
            self._thinking_header_shown = True
            self._stream.feed_thinking(chr(10) + "💭 ")
        if self._thinking_truncated:
            return  # cap already reached this round: drop further chunks
        self._thinking_chars += len(text)
        if self._max_thinking_tokens and self._thinking_chars > self._max_thinking_tokens * 4:
            overflow = self._thinking_chars - self._max_thinking_tokens * 4
            shown = text[: len(text) - overflow] if overflow < len(text) else ""
            self._thinking_truncated = True
            self._stream.feed_thinking(shown + " … (thinking truncated)")
            return
        self._stream.feed_thinking(text)

    def _consume_delta(self, delta: Any) -> str:
        """Route one stream delta to the thinking display / answer stream.

        Reasoning (``reasoning_content``/``reasoning`` fields, or inline
        ``ine`` tags inside ``content``) goes to the thinking display;
        the return value is the answer text of the delta (may be empty).
        With ``[ui] show_thinking`` off the answer passes through
        unchanged - including any literal ``ine`` tags - exactly as before
        the feature existed.
        """
        if not self._show_thinking:
            return getattr(delta, "content", None) or ""
        reasoning = self._reasoning_delta(delta)
        if reasoning:
            self._feed_thinking(reasoning)
        content = getattr(delta, "content", None) or ""
        if not content:
            return ""
        thinking, answer = self._split_thinking(content)
        if thinking:
            self._feed_thinking(thinking)
        return answer

    def model_fragments(self, prompt: str):
        """
        Get response fragments as stream from the model for a given prompt.
        Tools are not supported in this method.
        """
        self.logger.info("Sending to Model [🤖: %s] ...", self._model_name)
        self._add_user_prompt(prompt)
        self._cancel_requested = False  # fresh turn
        self._reset_thinking()
        self._stream.start_spinner("thinking")
        stream = self._stream_completion()
        self._active_stream = stream
        parts: list[str] = []
        try:
            for chunk in stream:
                if self._cancel_requested:
                    break
                delta = self._stream_delta(chunk)
                if delta is None:
                    continue
                answer = self._consume_delta(delta)
                if answer:
                    parts.append(answer)
                    yield SimpleNamespace(content=answer)
            pending = self._flush_pending_answer()
            if pending:  # trailing partial istine tag held back by _split_thinking
                parts.append(pending)
                yield SimpleNamespace(content=pending)
        finally:
            self._active_stream = None
            # Keep partial (or cancelled) output in the history.
            text = "".join(parts)
            if text:
                self._messages.append({"role": "assistant", "content": text})
            elif self._cancel_requested:
                self._messages.append({"role": "assistant", "content": CANCELLED_PLACEHOLDER})

    def print_fragment(self, fragment):
        """
        Print a response fragment to the console.
        Fenced code blocks are syntax-highlighted; plain text is cyan.
        This can be used as a callback for streaming responses.
        """
        self._stream.feed(fragment.content)

    def flush_stream(self) -> None:
        """Flush buffered text and stop the activity spinner (turn boundary)."""
        self._stream.flush()
        self._stream.stop_spinner()

    def cancel_generation(self) -> str:
        """
        Stop the in-flight model response as soon as possible (best effort).
        Not exposed to the LLM; invoked by the session's Ctrl+C handling.
        Closes the active HTTP stream for both streamed responses and ``act()``
        rounds; the pending flag covers a cancel raised while a tool is
        running (between rounds). Partial output is kept in the chat history.
        """
        self._cancel_requested = True
        stream, self._active_stream = self._active_stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                self.logger.debug("Closing prediction stream failed: %s", exc)
        return "Cancel requested"

    def interrupt(self) -> str:
        """
        Request a graceful stop of the current turn (public API).

        Unlike :meth:`cancel_generation` this never closes the active HTTP
        stream: the running completion round finishes naturally, the loop
        stops before the next round, and queued tools are not executed
        (see ``_execute_tool``). Not exposed to the LLM.
        """
        self._cancel_requested = True
        return "Interrupt requested"

    def _context_usage(self) -> tuple[float, int]:
        """Approximate context usage (no server-side tokenizer available)."""
        chars = sum(len(str(m.get("content") or "")) for m in self._messages)
        tokens = chars // 4  # rough ~4 chars per token estimate
        return float(tokens / self._context_window) * 100, self._context_window


    ## TOOLS ##

    @dump_func_call
    def model_info(self) -> str:
        """
        Returns detailed info about the loaded model: endpoint, provider and
        everything the server reports through the OpenAI models endpoint (or
        through its native API for local servers: ownership, context lengths,
        capabilities, quantization, ...), plus the current session settings.
        Useful when user ask for information about the loaded model.
        """
        meta = self._fetch_model_metadata()
        usage_pct, context_window = self._context_usage()
        lines = [
            f"model={self._model_name}",
            f"base_url={self._base_url} (OpenAI-compatible API)",
        ]
        if meta.get("owned_by"):
            lines.append(f"provider={meta['owned_by']}")
        created = meta.get("created")
        if created:
            if isinstance(created, (int, float)):
                created = datetime.fromtimestamp(
                    created, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            lines.append(f"created={created}")
        window_labels = {
            "config": "configured",
            "server": "server-reported",
            "default": "default (server did not report)",
        }
        window_source = getattr(self, "_context_window_source", "config")
        lines.append(
            f"context_window={context_window} tokens"
            f" ({window_labels.get(window_source, 'configured')})"
        )
        context_fields = {
            k: v for k, v in meta.items()
            if "context" in k.lower() and isinstance(v, int)
        }
        if context_fields:
            lines.append("server_context=" + ", ".join(
                f"{k}={v}" for k, v in sorted(context_fields.items())))
            max_fields = [v for k, v in context_fields.items()
                          if not k.lower().startswith("loaded")]
            if any(v < context_window for v in max_fields):
                lines.append(
                    "warning: configured context_window exceeds the "
                    "server-reported maximum"
                )
        if meta.get("capabilities"):
            lines.append(f"capabilities={', '.join(map(str, meta['capabilities']))}")
        rendered = {"owned_by", "created", "capabilities"} | set(context_fields)
        for key in sorted(k for k in meta if k not in rendered):
            value = meta[key]
            if isinstance(value, (list, tuple)):
                value = ", ".join(map(str, value))
            lines.append(f"{key}={value}")
        lines += [
            f"temperature={self._temperature}",
            f"max_response_tokens={self._max_tokens} (per-generation cap)",
            f"vision={'yes' if self._vision else 'no'}",
            f"vision_source={self._vision_source}",
            f"session_messages={len(self._messages)} (context ~{usage_pct:.1f}% used)",
        ]
        return "\n".join(lines)


class NullPrinter:
    """StreamPrinter stand-in for quiet agents (scheduler executor): buffer only, no console.

    Keeps the StreamPrinter interface (feed/flush/note/spinner/mark_line_clean)
    so Agent code paths stay unchanged; nothing is written to stdout.
    """

    def __init__(self) -> None:
        self.buffer: list[str] = []

    def feed(self, content: str) -> None:
        self.buffer.append(content)

    def feed_thinking(self, content: str) -> None:
        """Chain-of-thought is display-only: quiet agents drop it."""
        pass

    def flush(self) -> None:
        self.buffer.clear()

    def note(self, text: str) -> None:
        pass

    def start_spinner(self, note: str = "working") -> None:
        pass

    def stop_spinner(self) -> None:
        pass

    def mark_line_clean(self) -> None:
        pass


class Agent(Brain):
    """
    Agent that extends the Brain with OpenAI function-calling tool usage.

    Implements the round-based ``act()`` loop: stream a round, run the
    requested tools, feed the results back, and repeat until the model
    answers without tool calls.
    """

    MAX_ACT_ROUNDS = 100  # fallback when [agent] max_act_rounds is unset/0

    def _build_tools(self) -> list[Callable]:
        """Tool set of this agent.

        Isolated agents (``isolated_tools``) get fresh per-agent instances of
        the session-bound tools (memory, vision) so background agents never
        share module-level state. The main agent keeps the module-default
        instances bound by ``Brain.__init__`` — existing behavior unchanged.
        """
        tools = _select_tools(
            _default_tools(self),
            getattr(getattr(getattr(self, "config", None), "agent", None),
                    "tool_groups", "all"),
        )
        if not getattr(self, "isolated_tools", False):
            return tools
        from src.tools import memory as memory_mod, vision as vision_mod
        memory = memory_mod.MemoryTools()
        vision = vision_mod.VisionTools()
        memory.bind_history(lambda: self._messages)
        vision.bind_session(self)
        per_agent = {
            memory_mod.recall_memory: memory.recall_memory,
            memory_mod.store_memory: memory.store_memory,
            memory_mod.clear_chat: memory.clear_chat,
            memory_mod.compact_chat: memory.compact_chat,
            vision_mod.image_response: vision.image_response,
        }
        return [per_agent.get(tool, tool) for tool in tools]

    def __init__(self, config: SimpleNamespace, isolated_tools: bool = False, quiet: bool = False):
        # Set before super().__init__: OpenAICore.__init__ skips the
        # module-default tool binding for isolated agents, and
        # OpenAICore.__init__ builds the stream printer (quiet -> NullPrinter).
        self.isolated_tools = isolated_tools
        self.quiet = quiet
        super().__init__(config)
        # Per-agent round limit from [agent] max_act_rounds (0/unset -> class default).
        rounds = int(getattr(getattr(config, "agent", None), "max_act_rounds", 0) or 0)
        self.MAX_ACT_ROUNDS = rounds if rounds > 0 else type(self).MAX_ACT_ROUNDS
        # [brain] max_tool_result_chars: cap tool results stored in the history
        # (0/unset/invalid = uncapped, the historical behavior).
        try:
            result_chars = int(getattr(config.brain, "max_tool_result_chars", 0) or 0)
        except (TypeError, ValueError):
            result_chars = 0
        self._max_tool_result_chars = max(0, result_chars)

        # [agent] max_parallel_tools: run multi-tool rounds concurrently
        # (0/1 = sequential, the default; capped at 8 workers).
        try:
            parallel = int(getattr(getattr(config, "agent", None),
                                   "max_parallel_tools", 0) or 0)
        except (TypeError, ValueError):
            parallel = 0
        self._max_parallel_tools = min(max(0, parallel), 8)
        # [brain] context_trim_ratio / context_keep_messages: prune the oldest
        # history when the approximate context usage exceeds the ratio
        # (0/unset/invalid = never prune, the historical behavior).
        try:
            trim_ratio = float(getattr(config.brain, "context_trim_ratio", 0) or 0)
        except (TypeError, ValueError):
            trim_ratio = 0.0
        self._context_trim_ratio = trim_ratio if 0 < trim_ratio < 1 else 0.0
        try:
            keep_messages = int(
                getattr(config.brain, "context_keep_messages", 40) or 40)
        except (TypeError, ValueError):
            keep_messages = 40
        self._context_keep_messages = keep_messages if keep_messages >= 4 else 40
        # [brain] compact_keep_messages: recent turns kept by auto-compaction
        # (Task 5); missing/invalid falls back to context_keep_messages.
        try:
            compact_keep = int(
                getattr(config.brain, "compact_keep_messages", 0) or 0)
        except (TypeError, ValueError):
            compact_keep = 0
        self._compact_keep_messages = (
            compact_keep if compact_keep >= 2 else self._context_keep_messages)
        # [brain] auto_compaction: when true, trigger compact_chat at the
        # context-usage checkpoint instead of the plain oldest-round pruning
        # (missing/anything falsy = the historical behavior).
        self._auto_compaction = str(
            getattr(config.brain, "auto_compaction", False)).strip().lower() \
            in ("true", "1", "yes", "on")
        self._fragments_buffer: list[str] = []
        # Runtime tool toggling (/tools on|off): names here are excluded from
        # the schemas sent with every request (session-only, main agent).
        self._disabled_tools: set[str] = set()
        self.logger.info(
            "✅ Agent with Model '%s' initialized. Setting up tools ...",
            self._model_name
        )
        self.tools: list[Callable] = self._build_tools()
        # [agent] tool_doc_chars: cap tool descriptions in the JSON schemas
        # (0/unset/invalid = full docstring, the historical behavior).
        try:
            doc_chars = int(getattr(
                getattr(config, "agent", None), "tool_doc_chars", 0) or 0)
        except (TypeError, ValueError):
            doc_chars = 0
        self.tool_definitions: list[dict] = [
            self._openai_tool_definition(tool, max_doc_chars=max(0, doc_chars))
            for tool in self.tools
        ]
        self._tool_impls = {tool.__name__: tool for tool in self.tools}
        # MCP server tools (PLAN.md task 5): injected after the built-in tool
        # setup, for the main agent and isolated agents alike (the manager is
        # process-wide; isolated memory/vision tools are unaffected).
        from src.tools import mcp as mcp_tools
        extra_defs, extra_impls = mcp_tools.mcp_tool_bindings(
            existing_names=set(self._tool_impls))
        self._mcp_tool_names: list[str] = []
        if extra_defs:
            self.tools.extend(extra_impls.values())
            self.tool_definitions.extend(extra_defs)
            self._tool_impls.update(extra_impls)
            self._mcp_tool_names = list(extra_impls)
            self.logger.info(
                "✅ Injected %d MCP tool(s): %s",
                len(extra_impls), ", ".join(extra_impls))
        self.logger.info(
            "✅ Agent with Model '%s' is ready with [%d] tools.",
            self._model_name, len(self.tools)
        )

    def refresh_mcp_tools(self) -> str:
        """Reconnect the MCP servers and rebuild the injected tools in place.

        Drops every previously injected MCP tool from ``tools`` /
        ``tool_definitions`` / ``_tool_impls``, asks the manager to refresh
        its connections and re-runs the injection. No-op with a log line when
        no MCP manager is configured. Returns a short human-readable summary
        (used by ``/mcp refresh``).
        """
        from src.tools import mcp as mcp_tools
        manager = mcp_tools.get_manager()
        if manager is None:
            self.logger.debug("MCP refresh requested without a manager; ignored")
            return "MCP is not configured (no [mcp] servers)"
        for name in self._mcp_tool_names:
            self._tool_impls.pop(name, None)
        self.tools = [
            tool for tool in self.tools
            if tool.__name__ not in self._mcp_tool_names
        ]
        self.tool_definitions = [
            definition for definition in self.tool_definitions
            if definition.get("function", {}).get("name") not in self._mcp_tool_names
        ]
        results = manager.refresh()
        extra_defs, extra_impls = mcp_tools.mcp_tool_bindings(
            existing_names=set(self._tool_impls))
        self._mcp_tool_names = list(extra_impls)
        if extra_defs:
            self.tools.extend(extra_impls.values())
            self.tool_definitions.extend(extra_defs)
            self._tool_impls.update(extra_impls)
        # Runtime toggles survive a refresh: re-disable MCP tools the user
        # turned off via /tools off (they were just re-injected above).
        disabled = getattr(self, "_disabled_tools", None) or set()
        for name in [n for n in self._mcp_tool_names if n in disabled]:
            self._tool_impls.pop(name, None)
            self.tools = [
                tool for tool in self.tools
                if getattr(tool, "__name__", "") != name
            ]
            self.tool_definitions = [
                definition for definition in self.tool_definitions
                if definition.get("function", {}).get("name") != name
            ]
        connected = ", ".join(results) if results else "none"
        tools = ", ".join(extra_impls) if extra_impls else "none"
        return (f"MCP refreshed ({connected}): "
                f"{len(extra_impls)} tool(s) injected: {tools}")

    def bootstrap(self) -> None:
        """Run the configured bootstrap prompt (called by the session, not on init)."""
        if self.config.agent.bootstrap_prompt:
            self.act(self.config.agent.bootstrap_prompt)

    ## runtime tool toggling (/tools on|off) ##

    def enabled_tool_definitions(self) -> list[dict]:
        """Tool schemas of the currently enabled tools (toggle-aware).

        ``_stream_round`` sends these instead of the raw ``tool_definitions``,
        so a disabled tool stops costing tokens on the very next request.
        """
        disabled = getattr(self, "_disabled_tools", None) or set()
        if not disabled:
            return self.tool_definitions
        return [
            definition for definition in self.tool_definitions
            if definition.get("function", {}).get("name") not in disabled
        ]

    def set_tool_enabled(self, name: str, enabled: bool) -> str:
        """Enable/disable one tool at runtime; returns a confirmation line.

        A disabled tool is dropped from every subsequent request (token
        saving) and its execution is refused with a clear error, so a stale
        model reference fails loudly instead of silently running. Unknown
        names return an error string (never raise).
        """
        name = str(name or "").strip()
        known = {getattr(tool, "__name__", "") for tool in self.tools}
        if name not in known:
            return (f"Error: unknown tool '{name}'. Available: "
                    + ", ".join(sorted(n for n in known if n)))
        disabled = getattr(self, "_disabled_tools", None)
        if disabled is None:
            disabled = self._disabled_tools = set()
        if enabled:
            disabled.discard(name)
            return f"Tool '{name}' enabled ({len(self.tools) - len(disabled)} of {len(self.tools)} tools active)."
        disabled.add(name)
        return f"Tool '{name}' disabled ({len(self.tools) - len(disabled)} of {len(self.tools)} tools active)."

    def toggle_tools(self, names: list[str], enabled: bool) -> str:
        """Enable/disable several tools (or whole topics) at once.

        A name matching a ``_TOOL_TOPICS`` topic toggles every tool of that
        topic (e.g. ``finance`` covers ``fetch_financial_data`` and
        ``save_financial_chart``); otherwise it must be a tool name. Returns
        a multi-line confirmation.
        """
        name_to_topic = {
            name: topic
            for topic, topic_names in _TOOL_TOPICS.items() for name in topic_names
        }
        tool_names = [getattr(tool, "__name__", "") for tool in self.tools]
        topics = {name_to_topic[n] for n in tool_names if n in name_to_topic}
        lines: list[str] = []
        for raw in names:
            name = str(raw or "").strip()
            if not name:
                continue
            if name.lower() in {t.lower() for t in topics}:
                topic = next(t for t in topics if t.lower() == name.lower())
                members = [n for n in tool_names if name_to_topic.get(n) == topic]
                for member in members:
                    lines.append(self.set_tool_enabled(member, enabled))
                continue
            lines.append(self.set_tool_enabled(name, enabled))
        return "\n".join(lines) if lines else "Error: nothing to toggle."

    def _handle_fragment(self, content: str) -> None:
        self._fragments_buffer.append(content)
        self.print_fragment(SimpleNamespace(content=content))

    def _handle_round_end(self, round_index: int = 0) -> None:
        context_usage_percentage, context_window = self._context_usage()
        self._stream.flush()  # keep the spinner alive across rounds
        if not getattr(self, "quiet", False):
            cprint("")
            stats = f"context {context_usage_percentage:.1f}% [{int(context_window / 1000)}k]"
            turn_usage = getattr(self, "_turn_usage", None)
            if turn_usage and (turn_usage[0] or turn_usage[1]):
                stats += f" · in {turn_usage[0]} / out {turn_usage[1]} tok"
            started = getattr(self, "_turn_started", None)
            if started is not None:
                stats += f" · {time.monotonic() - started:.1f}s"
            crule(stats)
        self._stream.mark_line_clean()  # spinner may resume for the next round
        self._stream.note("working")
        response = " ".join(self._fragments_buffer).strip()
        if not getattr(self, "quiet", False) and not self._cancel_requested:
            speech_tools.speak_if_enabled(response)
        self._fragments_buffer.clear()

    def _stream_round(self, tools: list[dict]) -> tuple[str, list[dict]]:
        """Stream one completion round; return (text, accumulated tool calls).

        The final stream chunk's ``usage`` (servers that honor
        ``stream_options.include_usage``) is stored on
        ``_last_round_usage`` as ``(prompt_tokens, completion_tokens)``
        or ``None``; ``act()`` accumulates it per turn.
        """
        # Fresh thinking-display budget per round: a multi-round act() turn
        # would otherwise exhaust the cap in round 1 and show nothing later.
        self._reset_thinking()
        self._last_round_usage: tuple[int, int] | None = None
        stream = self._stream_completion(self.enabled_tool_definitions())
        self._active_stream = stream
        parts: list[str] = []
        calls: dict[int, dict] = {}
        try:
            for chunk in stream:
                if self._cancel_requested:
                    break
                usage = getattr(chunk, "usage", None)
                if usage is not None and getattr(usage, "total_tokens", None):
                    self._last_round_usage = (
                        int(getattr(usage, "prompt_tokens", 0) or 0),
                        int(getattr(usage, "completion_tokens", 0) or 0))
                delta = self._stream_delta(chunk)
                if delta is None:
                    continue
                answer = self._consume_delta(delta)
                if answer:
                    parts.append(answer)
                    self._handle_fragment(answer)
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
            pending = self._flush_pending_answer()
            if pending:  # trailing partial think tag held back by _split_thinking
                parts.append(pending)
                self._handle_fragment(pending)
        finally:
            self._active_stream = None
        return "".join(parts), [calls[i] for i in sorted(calls)]

    @staticmethod
    def _json_type_ok(value: Any, schema: dict) -> bool:
        """Best-effort check of a Python value against one JSON schema entry.

        Used by ``_validate_tool_args`` to catch obviously wrong argument
        types before the tool runs. ``None`` is always accepted (nullable
        parameters exist across the tool set and the schema fallbacks do not
        model them); unknown/absent types pass.
        """
        if value is None:
            return True
        json_type = schema.get("type")
        checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float))
                                and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int)
                                  and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }
        check = checks.get(json_type)
        return True if check is None else check(value)

    def _validate_tool_args(self, name: str, args: dict) -> str | None:
        """Check decoded tool arguments against the tool's JSON schema.

        smolagents' ``validate_tool_arguments`` pattern: required keys,
        unknown keys, primitive types and ``enum`` values are checked before
        the tool runs, and a *corrective* message is returned so the model
        can fix the call in the next round instead of guessing. Best-effort:
        tools without an object schema (e.g. some MCP tools) are not
        validated. Returns ``None`` when the arguments are acceptable.
        """
        definition = next(
            (d for d in self.tool_definitions
             if d.get("function", {}).get("name") == name), None)
        if definition is None:
            return None
        params = definition.get("function", {}).get("parameters") or {}
        properties = params.get("properties")
        if not isinstance(properties, dict):
            return None
        required = params.get("required") or []
        problems: list[str] = []
        missing = [key for key in required if key not in args]
        if missing:
            problems.append(
                f"missing required parameter(s): {', '.join(missing)}")
        unknown = [key for key in args if key not in properties]
        if unknown:
            problems.append(
                f"unknown parameter(s): {', '.join(unknown)}"
                f" (expected: {', '.join(properties) or 'none'})")
        type_errors: list[str] = []
        for key, value in args.items():
            schema = properties.get(key)
            if not isinstance(schema, dict):
                continue
            enum = schema.get("enum")
            if isinstance(enum, list) and value is not None and value not in enum:
                type_errors.append(
                    f"{key} must be one of {enum!r} (got {value!r})")
                continue
            if not self._json_type_ok(value, schema):
                type_errors.append(
                    f"{key} must be {schema.get('type')}"
                    f" (got {type(value).__name__})")
        if type_errors:
            problems.append("; ".join(type_errors))
        if not problems:
            return None
        return (f"Error: invalid arguments for '{name}': "
                + "; ".join(problems)
                + f". Valid parameters: {', '.join(sorted(properties)) or 'none'}.")
    def _execute_tool(self, call: dict) -> str:
        tool = self._tool_impls.get(call["name"])
        if tool is None:
            return f"Error: unknown tool '{call['name']}'"
        if call["name"] in (getattr(self, "_disabled_tools", None) or set()):
            return (f"Error: tool '{call['name']}' is disabled "
                    "(re-enable it with /tools on)")
        if getattr(self, "_cancel_requested", False):
            # interrupt()/cancel_generation() while tools were queued: do not
            # start new work, let the act() loop unwind instead.
            return "Error: execution cancelled by user (tool not run)"
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: malformed arguments for '{call['name']}': {exc}"
        if not isinstance(args, dict):
            return ("Error: invalid arguments for %r: expected a JSON object" % call["name"])
        validation_error = self._validate_tool_args(call["name"], args)
        if validation_error is not None:
            return validation_error
        try:
            result = tool(**args)
        except Exception as exc:
            result = f"Error: {exc}"
        result = result if isinstance(result, str) else str(result)
        # Sub-agent tool activity: isolated agents report their tool
        # calls/results to the process-wide monitor (agent name comes from
        # the thread-local set by the scheduler executor).
        monitor = get_monitor()
        if monitor is not None and _current_agent_name():
            monitor.record(_current_agent_name(), "tool_call",
                           tool=call["name"], args=call["arguments"] or "{}")
            monitor.record(_current_agent_name(), "tool_result",
                           tool=call["name"], result=result[:200])
        return result

    def _force_final_answer(self) -> str:
        """
        One tool-free completion asking the model to conclude the turn.

        Called by ``act()`` when the round budget is exhausted (smolagents'
        ``_handle_max_steps_reached`` pattern): the model gets the same
        history but no tool schemas, so its only way forward is a final
        answer. Returns the answer text ("" on failure or cancellation).
        """
        if self._cancel_requested:
            return ""
        self.logger.warning(
            "Max act rounds reached (%s); forcing a final answer.",
            self.MAX_ACT_ROUNDS)
        try:
            stream = self._stream_completion(None)
            self._active_stream = stream
            parts: list[str] = []
            try:
                for chunk in stream:
                    if self._cancel_requested:
                        break
                    delta = self._stream_delta(chunk)
                    if delta is None:
                        continue
                    answer = self._consume_delta(delta)
                    if answer:
                        parts.append(answer)
                        self._handle_fragment(answer)
                pending = self._flush_pending_answer()
                if pending:
                    parts.append(pending)
                    self._handle_fragment(pending)
            finally:
                self._active_stream = None
            return "".join(parts).strip()
        except Exception as exc:
            self.logger.error("Forced final answer failed: %s", exc)
            return ""

    def _run_tool_calls(self, tool_calls: list[dict]) -> None:
        """
        Execute one round's tool calls and append the tool messages.

        Sequential by default (the historical behavior, safest for
        tools that share session state). With ``[agent]
        max_parallel_tools > 1`` and more than one call in the round,
        the calls run in a bounded thread pool: each worker gets a
        ``contextvars.copy_context()`` so thread-locals (agent name,
        monitor wiring) keep working, and results are appended to the
        history in the original call order regardless of completion
        order (smolagents ``process_tool_calls`` pattern).
        """
        results = self._execute_tool_calls(tool_calls)
        for index, (call, result) in enumerate(zip(tool_calls, results)):
            self.logger.debug("Tool %s -> %s", call["name"],
                              str(result)[:200])
            self._messages.append({
                "role": "tool",
                "tool_call_id": call["id"] or f"call_{index}",
                "content": _truncate_tool_result(
                    result, getattr(self, "_max_tool_result_chars", 0)),
            })

    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[str]:
        """Results of one round's tool calls, in call order.

        Parallel execution when ``[agent] max_parallel_tools`` allows
        it and the round carries more than one call; sequential
        otherwise.
        """
        if len(tool_calls) < 2 or self._max_parallel_tools < 2:
            return [self._execute_tool(call) for call in tool_calls]
        workers = min(len(tool_calls), self._max_parallel_tools)
        contexts = [contextvars.copy_context() for _ in tool_calls]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(context.run, self._execute_tool, call)
                for context, call in zip(contexts, tool_calls)
            ]
            return [future.result() for future in futures]

    def act(self, prompt: str):
        """
        Perform an action based on the input prompt.
        The response may involve using tools or just generating a text.
        """
        if not getattr(self, "quiet", False):
            cprint(f"\n🤖 [{self._model_name}]", "grey")
        self.logger.info("Sending to Agent [🤖: %s] ...", self._model_name)
        self._add_user_prompt(prompt)
        self._prune_history()
        self._cancel_requested = False  # fresh turn
        self._reset_thinking()
        self._turn_started = time.monotonic()
        self._turn_usage = [0, 0]  # [prompt_tokens, completion_tokens]
        self._stream.start_spinner("thinking")
        try:
            for round_index in range(self.MAX_ACT_ROUNDS):
                text, tool_calls = self._stream_round(self.tool_definitions)
                usage = getattr(self, "_last_round_usage", None)
                if usage:
                    self._turn_usage[0] += usage[0]
                    self._turn_usage[1] += usage[1]
                if tool_calls and not self._cancel_requested:
                    assistant_call = {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {
                                "id": call["id"] or f"call_{index}",
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"] or "{}",
                                },
                            }
                            for index, call in enumerate(tool_calls)
                        ],
                    }
                    self._messages.append(assistant_call)
                    self._run_tool_calls(tool_calls)
                    self._handle_round_end(round_index)
                    continue
                if not text and self._cancel_requested:
                    text = CANCELLED_PLACEHOLDER
                self._messages.append({"role": "assistant", "content": text})
                self._handle_round_end(round_index)
                break
            else:
                # Round budget exhausted without a final text round: one
                # tool-free completion so the turn ends with an answer
                # (smolagents _handle_max_steps_reached pattern).
                final_text = self._force_final_answer()
                if not final_text and self._cancel_requested:
                    final_text = CANCELLED_PLACEHOLDER
                self._messages.append({"role": "assistant",
                                       "content": final_text})
        finally:
            self._stream.stop_spinner()
            self._active_stream = None


