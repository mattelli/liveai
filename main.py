import argparse
import contextlib
import logging
from pathlib import Path

from src.common import cprint, load_config, prompt_input

WORK_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = WORK_DIR / "config.toml"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live assistant (voice or chat mode)")
    parser.add_argument("--chat", action="store_true", help="Run in text chat mode")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool usage")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to config.toml")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt and exit")
    parser.add_argument("--tools-list", action="store_true",
                        help="List all available tools and exit "
                             "([off] = disabled by [agent] tool_groups)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    from src.common import ensure_data_dirs  # create the [paths] folders
    ensure_data_dirs(config)

    from src.common import configure_paths  # inject [cwd] tmp dir for scratch output
    configure_paths(config)

    from src.tools import configure as configure_tools  # inject config into tool modules
    configure_tools(config)
    from src.common import configure_ui  # inject [ui] display settings
    configure_ui(config)
    from src.live_ai import configure_notify  # inject [delegation.notify] hints
    configure_notify(config)

    logging.basicConfig(
        level=config.logging.level if hasattr(config, "logging") and hasattr(config.logging, "level") else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.tools_list:
        return _print_tools_list(config)
    from src.session import ConversationSession
    from src.live_ai import Agent  # lazy import: keeps `import main` side-effect free

    agent = Agent(config=config)
    _setup_monitor(config)    # bind the activity monitor when [agent] monitor is set
    _setup_scheduler(config)   # no-op when [delegation.scheduler] enabled is false/missing
    reader = None

    try:
        if args.prompt is not None:
            # Non-interactive: exactly one turn, no stdin reads.
            read_prompt = iter([args.prompt, None]).__next__
        elif args.chat:
            reader = _build_chat_reader(config, ConversationSession)
            read_prompt = lambda: reader.read(_chat_label(config))  # noqa: E731
        else:
            try:
                from src.audio_wrapper import AudioDeviceError, SpeechDetector  # lazy import
            except ImportError as exc:  # e.g. faster-whisper not installed
                cprint(f"❌ {exc} — install faster-whisper (see README / "
                       "setup-liveai-cpu.ps1), or use --chat.", "red")
                return 1
            except OSError as exc:  # e.g. sounddevice: PortAudio library not found
                cprint(f"❌ {exc} — install the PortAudio library "
                       "(e.g. sudo apt-get install libportaudio2), or use --chat.", "red")
                return 1
            try:
                detector = SpeechDetector(
                    model_path=config.speech_det.model_path,
                    language=config.speech_det.language,
                    beam_size=getattr(config.speech_det, "beam_size", 5),
                    show_levels=getattr(config.speech_det, "show_levels", True),
                    model_backend=getattr(config.speech_det, "device", "auto"),
                    compute_type=getattr(config.speech_det, "compute_type", "int8"),
                    calibration_chunks=getattr(config.speech_det, "calibration_chunks", 20),
                    guard_factor=getattr(config.speech_det, "guard_factor", 1.0),
                )
            except AudioDeviceError as exc:
                cprint(f"❌ {exc}", "red")
                return 1
            except FileNotFoundError as exc:
                cprint(f"❌ {exc} — install the Whisper model (see README), fix "
                       "[speech_det].model_path in config.toml, or use --chat.", "red")
                return 1
            except ImportError as exc:  # e.g. faster-whisper not installed
                cprint(f"❌ {exc} — install faster-whisper (see README / "
                       "setup-liveai-cpu.ps1), or use --chat.", "red")
                return 1
            read_prompt = lambda: detector.listen_and_transcribe(  # noqa: E731
                silence_timeout_s=config.speech_det.silence_timeout_s
            )

        conversation_session = ConversationSession(
            agent,
            read_prompt,
            use_tools=not args.no_tools,
            quit_on_prompt_interrupt=not args.chat,  # voice: Ctrl+C quits
        )
        prompt_toolkit = getattr(getattr(config, "input", None), "prompt_toolkit", True)
        return _run_session(conversation_session, args.chat, prompt_toolkit)
    finally:
        if reader is not None:
            reader.close()
        _stop_scheduler()  # no-op when the scheduler never started


def _run_session(session, chat: bool = False, prompt_toolkit: bool = False) -> int:
    """Run the session, making background-thread prints prompt-safe.

    Chat mode uses prompt_toolkit, which owns the input line: a raw print
    from a background thread (the [delegation.notify] completion hints) would
    garble
    the prompt. ``patch_stdout`` reroutes such prints above the prompt line
    and redraws the prompt below. Voice mode (plain input()) and the
    builtin-input fallback need nothing: raw printing is already safe there.
    When no interactive console is available (piped stdio, ``pythonw.exe``),
    entering ``patch_stdout`` fails and the session runs with plain prints.
    """
    if not (chat and prompt_toolkit):
        return session.run()
    try:
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError:  # pragma: no cover - guarded by the reader flag
        return session.run()
    with contextlib.ExitStack() as stack:
        try:
            # raw=True: keep ANSI escape codes (colors) in background prints;
            stack.enter_context(patch_stdout(raw=True))
        except Exception:  # no console behind sys.stdout: raw prints only
            pass
        return session.run()


def _setup_scheduler(config) -> None:
    """Start the task scheduler from the ``[delegation.scheduler]`` config section.

    No-op when the section is missing or ``enabled`` is false, so single-agent
    setups are untouched. Failures are logged, never raised: a broken
    scheduler must not prevent the session from starting.
    """
    from src.tools import scheduler as scheduler_module

    if not scheduler_module._enabled():
        return
    try:
        scheduler_module.bind_scheduler(scheduler_module._default_scheduler_factory())
        scheduler_module._SCHEDULER.start()
        cprint("⏰ Task scheduler started.", "grey")
    except Exception as exc:
        logging.getLogger(__name__).warning("Scheduler startup failed: %s", exc)


def _stop_scheduler() -> None:
    """Stop the scheduler thread if it was started (main.py finally block)."""
    from src.tools import scheduler as scheduler_module

    scheduler = scheduler_module._SCHEDULER
    if scheduler is not None:
        try:
            scheduler.stop()
        except Exception as exc:
            logging.getLogger(__name__).warning("Scheduler stop failed: %s", exc)


def _setup_monitor(config) -> None:
    """Create and bind the process-wide activity monitor from ``[agent]``.

    No-op when ``[agent] monitor`` is unset/empty: nothing is recorded and no
    log file is written. The monitor tracks scheduled-task runs.
    """
    from src.live_ai import build_monitor

    build_monitor(config)


def _print_tools_list(config) -> int:
    """Print every available tool grouped by topic and exit.

    All tools known to the agent (the full default set, before any
    ``[agent] tool_groups`` filtering) are listed; tools excluded by
    ``tool_groups`` carry an ``[off]`` marker, so the listing shows both what
    exists and what the running session would actually enable.
    """
    import types as _types

    from src.live_ai import Agent, _default_tools, _select_tools, tool_topics

    # Offline stand-in for the session-bound model_info tool: the listing only
    # needs the tool's name/docstring, never a live model connection. The
    # stand-in is renamed to model_info so the listing shows the real name.
    def _offline_model_info() -> str:
        """Returns detailed info about the loaded model: endpoint, provider and"""
        return ""

    _offline_model_info.__name__ = "model_info"
    offline_agent = _types.SimpleNamespace(
        model_info=_offline_model_info, _messages=[], _context_window=1,
        _temperature=0.0, _max_tokens=0, _vision=False,
        _vision_source="", config=config)
    all_tools = _default_tools(offline_agent)  # unfiltered: the full known set
    groups = getattr(getattr(config, "agent", None), "tool_groups", "all")
    enabled_names = {getattr(tool, "__name__", "")
                     for tool in _select_tools(all_tools, groups)}
    # [mcp] mcp_enabled = false: the two bridge tools are not available at all,
    # so mark them [off] in the listing (the agent drops them from its tools).
    from src.tools import mcp as mcp_tools
    if not mcp_tools.enabled():
        enabled_names -= {"mcp_list_tools", "mcp_call_tool"}

    cprint("Available tools:", "cyan", bold=True)
    for topic, topic_tools in tool_topics(all_tools):
        cprint(f"{topic}:", "yellow", bold=True)
        for tool in topic_tools:
            name = getattr(tool, "__name__", tool.__class__.__name__)
            doc = (getattr(tool, "__doc__", "") or "").strip().splitlines()
            summary = doc[0].strip() if doc else ""
            marker = "" if name in enabled_names else " [off]"
            line = f"  {name}{marker}"
            if summary:
                line += f" - {summary}"
            cprint(line, "cyan")
    cprint(f"{len(enabled_names)} of {len(all_tools)} tools enabled "
           "([off] = excluded by [agent] tool_groups).", "grey")
    return 0


def _chat_label(config) -> str:
    """Chat-mode input label from the optional `[ui]` block (space-terminated)."""
    text = getattr(getattr(config, "ui", None), "prompt", None) or "You ›"
    return text if text.endswith(" ") else f"{text} "


def _build_chat_reader(config, session_cls):
    """Build the chat InputReader from the optional `[input]` config block."""
    from src.input_reader import InputReader  # lazy import

    in_cfg = getattr(config, "input", None)
    if not getattr(in_cfg, "prompt_toolkit", True):
        # Degraded path: route through prompt_input, never prompt_toolkit.
        return InputReader(session=None, fallback=prompt_input)

    completer = None
    if getattr(in_cfg, "completion", True):
        try:
            from prompt_toolkit.completion import WordCompleter

            completer = WordCompleter(
                session_cls.COMMANDS, ignore_case=True, sentence=True
            )
        except ImportError:  # pragma: no cover - env dependent
            completer = None

    return InputReader(
        memory_path=getattr(in_cfg, "history_file_cwd", None),
        auto_suggest=getattr(in_cfg, "auto_suggest", True),
        completer=completer,
        multiline=getattr(in_cfg, "multiline", False),
        explicit_paste=getattr(in_cfg, "explicit_paste", True),
        prompt_color=getattr(getattr(config, "ui", None), "prompt_color", "cyan"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
