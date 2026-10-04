from __future__ import annotations

import contextlib
import inspect
import signal
from typing import Callable

from src.common import cprint
from src.tools import allow_speech, clear_chat


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

    COMMANDS = ("/help", "/quit", "/clear", "/voice", "/tools", "/info")

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
        if self.use_tools:
            self.agent.act(text)
        else:
            for f in self.agent.model_fragments(text):
                self.agent.print_fragment(f)

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

        cprint(f"Unknown command: {text}. Type /help.", "red")
        return True

    def _cmd_help(self) -> None:
        cprint("Available commands:", "cyan")
        cprint("  /help              - show this help", "cyan")
        cprint("  /quit              - exit the session (alias: exit)", "cyan")
        cprint("  /clear             - clear the conversation history", "cyan")
        cprint("  /voice on|off      - toggle speech output", "cyan")
        cprint("  /tools             - list and run an available tool", "cyan")
        cprint("  /info              - show model/context info", "cyan")
        cprint("  Ctrl+C             - stop the response being generated", "cyan")
        cprint("  Ctrl+D             - exit the session", "cyan")

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
        for index, tool in enumerate(tools, 1):
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
        if hasattr(self.agent, "model_info"):
            cprint(self.agent.model_info(), "cyan")
        else:
            cprint("Info unavailable", "grey")

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

    def banner(self) -> None:
        hint = (
            ""
            if self._quit_on_prompt_interrupt
            else " Ctrl+C stops a response; Ctrl+D quits."
        )
        cprint(
            f"Agent ready. Tools are {'enabled' if self.use_tools else 'disabled'}."
            f"{hint}",
            "green",
        )
