from __future__ import annotations

import contextlib
import inspect
import json
import pathlib
import signal
from datetime import datetime
from typing import Callable

from src.common import cprint, safe_filename
from src.tools import allow_speech, clear_chat


# Startup logo shown inside the banner panel ([ui] banner_logo = false turns
# it off). Raw ASCII art: keep it a raw string so the backslashes survive.
_LOGO = r"""
 _   _ _____ ____      _    ____  _   _    _    ____  _____ ____
| | | | ____|  _ \    / \  |  _ \| \ | |  / \  / ___| ____|  _ \
| | | |  _| | |_) |  / _ \ | | | |  \| | / _ \| |   |  _| | |_) |
| |_| | |___|  _ <  / ___ \| |_| | |\  |/ ___ \| |___| |___|  _ <
 \___/|_____|_| \_\/_/   \_\____/|_| \_/_/   \_\____|_____|_| \_\
"""


def _gradient_logo(art: str = _LOGO, start: tuple = (0, 200, 255),
                   end: tuple = (180, 0, 255)):
    """The logo as rich ``Text`` with a vertical truecolor gradient.

    Each line interpolates ``start`` -> ``end`` RGB; rich degrades the
    styles automatically on terminals without truecolor support.
    """
    from rich.text import Text

    text = Text()
    rows = art.strip("\n").splitlines()
    for index, row in enumerate(rows):
        t = index / max(1, len(rows) - 1)
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        text.append(row + "\n", style=f"rgb({r},{g},{b})")
    return text


@contextlib.contextmanager
def _generation_sigint_guard(agent):
    """Route SIGINT to ``agent.cancel_generation`` while a generation runs.

    A plain KeyboardInterrupt unwinds the streaming loop before the prediction
    stream can be cancelled, leaving LM Studio generating an abandoned
    response. The temporary handler instead asks the agent to cancel first
    (server-side stop); the agent's own callbacks then raise to unwind the
    generation promptly. Agents without ``cancel_generation`` keep the default
    Ctrl+C behaviour.
    """
    cancel = getattr(agent, "cancel_generation", None)
    if not callable(cancel):
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):
        try:
            cancel()
        except Exception:  # never let the handler explode
            pass

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):  # e.g. not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


class ConversationSession:
    """Shared interaction loop for text and voice modes.

    Owns prompt acquisition, slash commands, per-turn error recovery and
    shutdown. All collaborators are injected, so no heavy module is imported.
    """

    COMMANDS = ("/help", "/quit", "/clear", "/voice", "/tools", "/info",
                "/model", "/session", "/mcp", "/monitor")

    def __init__(self, agent, read_prompt: Callable[[], str | None], use_tools: bool = True,
                 quit_on_prompt_interrupt: bool = False):
        """
        agent: object exposing act/model_fragments/print_fragment/flush_stream/
               model_info (FakeAgent in tests). Conversation-level tools
               (clear_chat, allow_speech) live in src.tools and are called
               directly, independent of the agent object.
        read_prompt: Callable[[], str | None]; None means EOF
        use_tools: bool
        quit_on_prompt_interrupt: Ctrl+C while waiting for input exits instead of
              re-prompting (voice / non-interactive prompt sources; in chat mode
              Ctrl+C clears the line and Ctrl+D exits)
        """
        self.agent = agent
        self.read_prompt = read_prompt
        self.use_tools = use_tools
        self._quit_on_prompt_interrupt = quit_on_prompt_interrupt
        self.should_quit = False
        self._iteration = 0
        self._speech_enabled = False
        self._shutdown_done = False

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        self.banner()
        bootstrap = getattr(self.agent, "bootstrap", None)
        if callable(bootstrap):
            self._run_generation_step(bootstrap)

        while not self.should_quit:
            try:
                prompt = self.read_prompt()
            except KeyboardInterrupt:
                # Ctrl+C while waiting for input: quit for voice / non-interactive
                # prompt sources, otherwise clear the line and prompt again.
                if self._quit_on_prompt_interrupt:
                    break
                continue
            if prompt is None:  # EOF
                break
            text = prompt.strip()
            if text == "":
                continue
            # A multi-line paste goes to the agent even if its first line looks
            # like a command; only a single-line string may be a command.
            if "\n" not in text and self.handle_command(text):
                continue
            self._run_generation_step(lambda: self._run_turn(text))
            cprint("")  # blank separator line after each turn
        return self.shutdown()

    def _run_generation_step(self, step: Callable[[], None]) -> None:
        """Run one agent generation (bootstrap or user turn) with Ctrl+C support.

        Ctrl+C cancels the current generation; the session keeps running. The
        ``SIGINT`` swap gives agents with ``cancel_generation`` a best-effort
        server-side stop before the streaming loop unwinds.
        """
        interrupted = False
        try:
            with _generation_sigint_guard(self.agent):
                step()
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            cprint(f"❌ {exc}", "red")
        finally:
            self._flush()
            if interrupted or getattr(self.agent, "cancel_requested", False):
                cprint(" ⏹  generation cancelled", "yellow")

    def _run_turn(self, text: str) -> None:
        # Inject any scheduled-task results that fired since the last turn so
        # the main agent becomes aware of them (same visibility as pooled
        # subagent answers, which are returned into the history directly).
        self._inject_scheduled_results()
        if self.use_tools:
            self.agent.act(text)
        else:
            for f in self.agent.model_fragments(text):
                self.agent.print_fragment(f)

    def _inject_scheduled_results(self) -> None:
        """Append pending scheduled-task results to the agent's history.

        The scheduler thread queues results instead of writing the history
        directly (an append from another thread could corrupt an in-flight
        tool-call round); here the main thread owns the list, so appending is
        safe. Results arrive as assistant messages the agent can mention or act
        on during the turn.
        """
        try:
            from src.tools.scheduler import drain_scheduled_results
        except ImportError:  # pragma: no cover - scheduler module always exists
            return
        for task_id, answer in drain_scheduled_results():
            self.agent.messages.append({
                "role": "assistant",
                "content": f"Scheduled task {task_id} completed. Result:\n{answer}",
            })

    def _flush(self) -> None:
        if hasattr(self.agent, "flush_stream"):
            self.agent.flush_stream()

    # -------------------------------------------------------------- commands

    def handle_command(self, text: str) -> bool:
        """True if the text was consumed as a command."""
        if not text.startswith("/"):
            lowered = text.lower()
            if lowered == "exit":
                self.should_quit = True
                return True
            return False

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit":
            self.should_quit = True
            return True
        if cmd == "/help":
            self._cmd_help()
            return True
        if cmd == "/clear":
            self._cmd_clear()
            return True
        if cmd == "/voice":
            self._cmd_voice(arg)
            return True
        if cmd == "/tools":
            self._cmd_tools(arg)
            return True
        if cmd == "/info":
            self._cmd_info()
            return True
        if cmd == "/model":
            self._cmd_model(arg)
            return True
        if cmd == "/session":
            self._cmd_session(arg)
            return True
        if cmd == "/mcp":
            self._cmd_mcp(arg)
            return True
        if cmd == "/monitor":
            self._cmd_monitor(arg)
            return True

        cprint(f"Unknown command: {text}. Type /help.", "red")
        return True

    def _cmd_help(self) -> None:
        rows = [
            ("/help", "show this help"),
            ("/quit", "exit the session (alias: exit)"),
            ("/clear", "clear the conversation history"),
            ("/voice on|off", "toggle speech output"),
            ("/tools", "list and run an available tool"),
            ("/info", "show model/context info"),
            ("/model [name]", "switch to a model from the models JSON file"),
            ("/session store|list|load", "save/list/restore chat sessions (JSON)"),
            ("/mcp [refresh]", "list the MCP servers/tools; refresh reconnects"),
            ("/monitor [N|clear]", "show the last sub-agent activities; clear empties"),
            ("Ctrl+C", "stop the response being generated"),
            ("Ctrl+D", "exit the session"),
        ]
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 2, 0, 0), pad_edge=False)
            table.add_column(style="cyan", no_wrap=True)
            table.add_column(style="cyan")
            for command, description in rows:
                table.add_row(command, description)
            Console().print(table)
        except ImportError:  # pragma: no cover - rich is a hard dependency
            cprint("Available commands:", "cyan")
            for command, description in rows:
                cprint(f"  {command:<15} - {description}", "cyan")

    def _cmd_clear(self) -> None:
        clear_chat()
        cprint("Chat cleared", "green")

    def _cmd_voice(self, arg: str) -> None:
        arg = arg.strip().lower()
        if arg == "on":
            self._speech_enabled = True
            allow_speech(True)
            cprint("Voice enabled", "green")
        elif arg == "off":
            self._speech_enabled = False
            allow_speech(False)
            cprint("Voice disabled", "green")
        else:
            cprint("Usage: /voice on|off", "yellow")

    def _cmd_tools(self, arg: str) -> None:
        if arg.strip():
            cprint("Usage: /tools", "yellow")
            return
        self._launch_tool()

    @staticmethod
    def _tool_name(tool) -> str:
        return getattr(tool, "__name__", tool.__class__.__name__)

    def _read_tool_input(self, hint: str) -> str | None:
        """Prompt for a tool interaction, handling EOF and Ctrl-C."""
        cprint(hint, "yellow")
        try:
            value = self.read_prompt()
        except KeyboardInterrupt:
            self.should_quit = True
            return None
        if value is None:
            self.should_quit = True
        return value

    def _launch_tool(self) -> None:
        """List available tools, let the user pick one and run it.

        Required parameters (those without a default value) are prompted for;
        a tool that takes no parameters runs immediately after selection.
        """
        tools = list(getattr(self.agent, "tools", []) or [])
        if not tools:
            cprint("No tools available.", "grey")
            return

        cprint("Available tools:", "cyan")
        import src.live_ai as live_ai
        # Grouped display order is also the selection order: flatten the
        # groups so number N maps to the Nth listed tool.
        tools = [tool for _, group in live_ai.tool_topics(tools) for tool in group]
        index = 0
        for topic, topic_tools in live_ai.tool_topics(tools):
            cprint(f"{topic}:", "yellow", bold=True)
            for tool in topic_tools:
                index += 1
                doc = (getattr(tool, "__doc__", "") or "").strip().splitlines()
                summary = doc[0].strip() if doc else ""
                line = f"  {index:>2}. {self._tool_name(tool)}"
                if summary:
                    line += f" - {summary}"
                cprint(line, "cyan")

        selection = self._read_tool_input(
            "Select a tool (number or name, blank to cancel): "
        )
        if selection is None:
            return
        selection = selection.strip()
        if not selection:
            cprint("Cancelled.", "grey")
            return

        tool = None
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(tools):
                tool = tools[index - 1]
        else:
            for candidate in tools:
                if self._tool_name(candidate).lower() == selection.lower():
                    tool = candidate
                    break
        if tool is None:
            cprint(f"Unknown tool: {selection}", "red")
            return

        arguments = []
        for name, param in inspect.signature(tool).parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            type_hint = self._annotation_name(param.annotation)
            hint = f"{name} ({type_hint}): " if type_hint else f"{name}: "
            raw = self._read_tool_input(hint)
            if raw is None:
                return
            arguments.append(self._convert_argument(raw, param.annotation))

        cprint(f"\u25b6 {self._tool_name(tool)}", "green")
        try:
            result = tool(*arguments)
        except Exception as exc:
            cprint(f"\u274c {exc}", "red")
            return
        cprint(str(result), "green")

    @staticmethod
    def _annotation_name(annotation) -> str:
        if annotation is inspect.Parameter.empty:
            return ""
        if isinstance(annotation, str):
            return annotation
        return getattr(annotation, "__name__", str(annotation))

    @staticmethod
    def _convert_argument(value: str, annotation) -> object:
        if annotation is int or annotation == "int":
            try:
                return int(value)
            except ValueError:
                return value
        if annotation is float or annotation == "float":
            try:
                return float(value)
            except ValueError:
                return value
        if annotation is bool or annotation == "bool":
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return value

    def _cmd_info(self) -> None:
        if not hasattr(self.agent, "model_info"):
            cprint("Info unavailable", "grey")
            return
        info = str(self.agent.model_info())
        try:
            from rich.console import Console
            from rich.markup import escape
            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 2, 0, 0), pad_edge=False)
            table.add_column(style="cyan", no_wrap=True)
            table.add_column()
            for line in info.splitlines():
                line = line.strip()
                if not line:
                    continue
                key, sep, value = line.partition("=")
                if sep:
                    table.add_row(escape(key), escape(value))
                else:  # not key=value shaped: render as a plain row
                    table.add_row("", escape(line))
            Console().print(table)
        except ImportError:  # pragma: no cover - rich is a hard dependency
            cprint(info, "cyan")

    # ------------------------------------------------- model switching (/model)

    def _cmd_model(self, arg: str) -> None:
        """Switch the live session to a model entry from the models JSON file.

        ``/model`` lists the selectable models (current one marked);
        ``/model <name>`` rebuilds the client for that entry's
        ``api_key_env``/``base_url``/``model`` (see ``Brain.select_model``).
        """
        arg = arg.strip()
        if not hasattr(self.agent, "select_model"):
            cprint("Model switching unavailable for this agent.", "grey")
            return
        if not arg:
            self._model_list()
            cprint("Usage: /model <name>", "yellow")
            return
        try:
            message = self.agent.select_model(arg)
        except Exception as exc:
            cprint(f"Error switching model: {exc}", "red")
            return
        cprint(message, "green")

    def _model_list(self) -> None:
        """Render the models JSON file entries as a table (current one marked)."""
        try:
            entries = self.agent.list_models()
        except Exception as exc:
            cprint(f"Error reading models file: {exc}", "red")
            return
        current = getattr(self.agent, "model_name", None)
        rows = []
        for name, entry in entries.items():
            model = str(entry.get("model") or "")
            base_url = str(entry.get("base_url") or "")
            rows.append((name, model, base_url))
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(show_header=True, box=None, padding=(0, 2, 0, 0), pad_edge=False)
            table.add_column("name", style="cyan", no_wrap=True)
            table.add_column("model", style="cyan")
            table.add_column("base_url", style="cyan")
            for name, model, base_url in rows:
                marker = "→ " if name.lower() == (current or "").lower() else ""
                table.add_row(f"{marker}{name}", model, base_url)
            Console().print(table)
        except ImportError:  # pragma: no cover - rich is a hard dependency
            for name, model, base_url in rows:
                marker = " ->" if name.lower() == (current or "").lower() else "   "
                cprint(f"  {marker} {name:<20} {model:<40} {base_url}", "cyan")

    # ------------------------------------------------- session files (/session)

    def _cmd_session(self, arg: str) -> None:
        """Dispatch the ``/session`` sub-commands (store/list/load)."""
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        if sub == "store":
            self._session_store(rest or None)
        elif sub == "list":
            self._session_list()
        elif sub == "load":
            self._session_load(rest)
        else:
            cprint("Usage: /session store [name] | /session list | "
                   "/session load <name|index>", "yellow")

    def _sessions_dir(self) -> pathlib.Path | None:
        """Session folder from ``[brain] sessions_dir`` (None when unset)."""
        brain = getattr(getattr(self.agent, "config", None), "brain", None)
        value = getattr(brain, "sessions_dir", None) if brain is not None else None
        return pathlib.Path(value) if value else None

    def _chat_history(self) -> list[dict] | None:
        """Live conversation history of the agent (None when unavailable)."""
        messages = getattr(self.agent, "messages", None)
        return messages if isinstance(messages, list) else None

    def _session_store(self, name: str | None) -> None:
        """Store the full chat history as a JSON file in the session folder."""
        messages = self._chat_history()
        if messages is None:
            cprint("No chat history available to store.", "red")
            return
        directory = self._sessions_dir()
        if directory is None:
            cprint("No session folder configured: "
                   "set [brain] sessions_dir in config.toml", "red")
            return
        if name:
            stem = safe_filename(name)
            filename = stem if stem.endswith(".json") else f"{stem}.json"
        else:
            filename = datetime.now().strftime("session-%Y%m%d-%H%M%S.json")
        path = directory / filename
        if path.exists():
            cprint(f"Session file already exists: {path}", "red")
            return
        payload = {
            "stored_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": getattr(self.agent, "_model_name", None) or "unknown",
            "messages": messages,
        }
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            cprint(f"Error storing session: {exc}", "red")
            return
        cprint(f"Session stored: {path} ({len(messages)} messages)", "green")

    def _session_list(self) -> None:
        """List the session files in the configured folder."""
        directory = self._sessions_dir()
        files = sorted(directory.glob("*.json")) if directory and directory.is_dir() else []
        if not files:
            cprint("No session files found.", "grey")
            return
        rows = []
        for path in files:
            stat = path.stat()
            count = "-"
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data.get("messages"), list):
                    count = str(len(data["messages"]))
            except Exception:
                pass  # unreadable file: keep the row with a '-' count
            rows.append((
                path.name,
                count,
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                f"{stat.st_size / 1024:.1f} KB",
            ))
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(show_header=True, box=None, padding=(0, 2, 0, 0), pad_edge=False)
            table.add_column("session file", style="cyan", no_wrap=True)
            table.add_column("messages", style="cyan", justify="right")
            table.add_column("stored", style="cyan")
            table.add_column("size", style="cyan", justify="right")
            for row in rows:
                table.add_row(*row)
            Console().print(table)
        except ImportError:  # pragma: no cover - rich is a hard dependency
            for name, count, stored, size in rows:
                cprint(f"  {name:<40} {count:>8}  {stored}  {size:>9}", "cyan")

    def _resolve_session_file(self, target: str) -> pathlib.Path | None:
        """Resolve a ``/session load`` target: 1-based list index or file name."""
        directory = self._sessions_dir()
        if directory is None or not directory.is_dir():
            return None
        files = sorted(directory.glob("*.json"))
        if target.isdigit():
            return files[int(target) - 1] if 1 <= int(target) <= len(files) else None
        name = target if target.endswith(".json") else f"{target}.json"
        path = directory / name
        return path if path.exists() else None

    def _session_load(self, target: str) -> None:
        """Replace the live chat history with a stored session file."""
        messages = self._chat_history()
        if messages is None:
            cprint("No chat history available to load into.", "red")
            return
        if not target:
            cprint("Usage: /session load <name|index>  (see /session list)", "yellow")
            return
        path = self._resolve_session_file(target)
        if path is None:
            cprint(f"No session file: {target}", "red")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = data["messages"]
            if not (isinstance(loaded, list) and
                    all(isinstance(m, dict) and m.get("role") for m in loaded)):
                raise ValueError("invalid 'messages' list")
        except Exception as exc:
            cprint(f"Error loading session {path.name}: {exc}", "red")
            return
        messages.clear()  # in place: the session keeps the same list object
        messages.extend(loaded)
        model = data.get("model") or "unknown model"
        cprint(f"Session loaded: {path.name} "
               f"({len(loaded)} messages, model {model})", "green")

    # ------------------------------------------------- MCP servers (/mcp)

    def _cmd_mcp(self, arg: str) -> None:
        """Inspect the configured MCP servers; ``refresh`` reconnects them.

        ``/mcp`` renders a table of servers (name, transport, status, tool
        count); ``/mcp refresh`` asks the agent to reconnect and rebuild the
        injected tools (falling back to the manager when the agent has no
        ``refresh_mcp_tools``).
        """
        import src.tools.mcp as mcp_tools

        arg = arg.strip().lower()
        if arg == "refresh":
            refresh = getattr(self.agent, "refresh_mcp_tools", None)
            if callable(refresh):
                cprint(str(refresh()), "green")
                return
            manager = mcp_tools.get_manager()
            if manager is None:
                cprint("MCP is not configured (no [mcp] servers)", "yellow")
                return
            try:
                results = manager.refresh()
            except Exception as exc:
                cprint(f"Error refreshing MCP: {exc}", "red")
                return
            cprint(f"MCP refreshed: {', '.join(results) or 'no server connected'}",
                   "green")
            return
        if arg:
            cprint("Usage: /mcp [refresh]", "yellow")
            return
        manager = mcp_tools.get_manager()
        if manager is None:
            cprint("MCP is not configured (no [mcp] servers)", "grey")
            cprint("Usage: /mcp [refresh]", "yellow")
            return
        try:
            status = manager.status()
        except Exception as exc:
            cprint(f"Error reading MCP status: {exc}", "red")
            return
        rows = [(name, entry.get("transport", ""),
                 entry.get("status", ""), str(len(entry.get("tools", []))))
                for name, entry in status.items()]
        try:
            from rich.console import Console
            from rich.markup import escape
            from rich.table import Table

            table = Table(show_header=True, box=None, padding=(0, 2, 0, 0), pad_edge=False)
            table.add_column("server", style="cyan", no_wrap=True)
            table.add_column("transport", style="cyan")
            table.add_column("status", style="cyan")
            table.add_column("tools", style="cyan", justify="right")
            for row in rows:
                table.add_row(*(escape(cell) for cell in row))
            Console().print(table)
        except ImportError:  # pragma: no cover - rich is a hard dependency
            for name, transport, status_text, count in rows:
                cprint(f"  {name:<20} {transport:<7} {status_text:<30} {count}", "cyan")
        cprint("Usage: /mcp refresh", "yellow")

    # -------------------------------------------------------------- shutdown

    def shutdown(self) -> int:
        self._flush()
        if self._speech_enabled:
            allow_speech(False)
            self._speech_enabled = False
        if not self._shutdown_done:
            cprint("👋 Bye!", "green")
            self._shutdown_done = True
        return 0

    # ---------------------------------------------------------------- banner

    def _cmd_monitor(self, arg: str) -> None:
        """Show the recent sub-agent activity (``/monitor [N|clear]``).

        ``/monitor`` prints the most recent events (default 20), ``/monitor N``
        the last N events and ``/monitor clear`` empties the in-memory buffer
        (the JSONL file sink is not touched). When monitoring is disabled in
        the config (``[agent] monitor`` without ``calls``) a hint is shown.
        """
        import src.live_ai as live_ai

        arg = arg.strip().lower()
        monitor = live_ai.get_monitor()
        if arg == "clear":
            if monitor is None:
                cprint("Monitoring is not active (no agent pool).", "yellow")
                return
            monitor.clear()
            cprint("Monitor buffer cleared.", "green")
            return
        limit = 20
        if arg:
            try:
                limit = max(1, int(arg))
            except ValueError:
                cprint("Usage: /monitor [N|clear]", "yellow")
                return
        if monitor is None:
            cprint("Monitoring is not active (no agent pool).", "yellow")
            return
        flags = live_ai.agent_pool_settings(
            getattr(self.agent, "config", None)).get("monitor") or set()
        if not monitor.snapshot(limit=1):
            cprint("No sub-agent activity recorded.", "yellow")
            return
        text = monitor.tail(limit=limit)
        cprint(text, "cyan")

    def banner(self) -> None:
        ui = getattr(getattr(self.agent, "config", None), "ui", None)
        tools = "enabled" if self.use_tools else "disabled"
        hint = (
            ""
            if self._quit_on_prompt_interrupt
            else " Ctrl+C stops a response; Ctrl+D quits."
        )
        multiline = self._multiline_enabled()
        input_hint = (
            "Esc+Enter accepts (multiline on)"
            if multiline
            else "Enter accepts (multiline off)"
        )
        plain = f"Agent ready. Tools are {tools}. Input: {input_hint}.{hint}"
        if ui is not None and not getattr(ui, "banner", True):
            cprint(plain, "green")
            return
        show_logo = bool(getattr(ui, "banner_logo", True))
        try:
            from rich.console import Console
            from rich.markup import escape
            from rich.panel import Panel
            from rich.text import Text

            body = Text()
            if show_logo:
                body.append_text(_gradient_logo())
                body.append("\n")
            model = getattr(self.agent, "_model_name", None)
            if model:
                body.append(f"model ", style="bold")
                body.append(f"{escape(str(model))}\n")
            body.append(
                f"tools {tools} · /help commands · Ctrl+C stop"
                + ("" if self._quit_on_prompt_interrupt else " · Ctrl+D quit")
                + "\n"
            )
            body.append(
                f"input multiline {'on' if multiline else 'off'} · "
                + ("Esc+Enter accepts" if multiline else "Enter accepts")
            )
            Console().print(
                Panel(
                    body,
                    title="Live Assistant",
                    border_style="cyan",
                    expand=True,
                    padding=(0, 2),
                )
            )
        except ImportError:  # pragma: no cover - rich is a hard dependency
            cprint(plain, "green")

    def _multiline_enabled(self) -> bool:
        """``[input] multiline`` from config.toml (False when unset)."""
        input_cfg = getattr(getattr(self.agent, "config", None), "input", None)
        return bool(getattr(input_cfg, "multiline", False))
