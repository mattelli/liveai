import argparse
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
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    from src.tools import configure as configure_tools  # inject config into tool modules
    configure_tools(config)
    from src.common import configure_ui  # inject [ui] display settings
    configure_ui(config)

    logging.basicConfig(
        level=config.logging.level if hasattr(config, "logging") and hasattr(config.logging, "level") else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    from src.session import ConversationSession
    from src.live_ai import Agent  # lazy import: keeps `import main` side-effect free

    agent = Agent(config=config)
    _setup_agent_pool(config)  # no-op when [agent] parallel_agents = 0
    _setup_scheduler(config)   # no-op when [scheduler] enabled is false/missing
    reader = None

    try:
        if args.prompt is not None:
            # Non-interactive: exactly one turn, no stdin reads.
            read_prompt = iter([args.prompt, None]).__next__
        elif args.chat:
            reader = _build_chat_reader(config, ConversationSession)
            read_prompt = lambda: reader.read(_chat_label(config))  # noqa: E731
        else:
            from src.audio_wrapper import AudioDeviceError, SpeechDetector  # lazy import

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
            read_prompt = lambda: detector.listen_and_transcribe(  # noqa: E731
                silence_timeout_s=config.speech_det.silence_timeout_s
            )

        return ConversationSession(
            agent,
            read_prompt,
            use_tools=not args.no_tools,
            quit_on_prompt_interrupt=not args.chat,  # voice: Ctrl+C quits
        ).run()
    finally:
        if reader is not None:
            reader.close()
        _stop_scheduler()  # no-op when the scheduler never started


def _setup_scheduler(config) -> None:
    """Start the task scheduler from the ``[scheduler]`` config section.

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


def _setup_agent_pool(config) -> None:
    """Build the multi-agent pool from ``[agent]`` and bind it to the tools.

    No-op when ``parallel_agents`` is 0/missing, so single-agent setups are
    untouched. Failures raise: a misconfigured pool must not start silently.
    """
    from src.live_ai import AgentPool, agent_pool_settings, bind_agent_pool

    agent_cfg = getattr(config, "agent", None)
    n = getattr(agent_cfg, "parallel_agents", 0) or 0
    if not n:
        return
    settings = agent_pool_settings(config)
    if len(settings["models"]) < n:
        raise ValueError(
            f"[agent] parallel_agents={n} but agent_models has only "
            f"{len(settings['models'])} entries: add one models.json entry "
            "name per worker agent")
    bind_agent_pool(AgentPool(config, settings))


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
        memory_path=getattr(in_cfg, "history_file", None),
        auto_suggest=getattr(in_cfg, "auto_suggest", True),
        completer=completer,
        multiline=getattr(in_cfg, "multiline", False),
        explicit_paste=getattr(in_cfg, "explicit_paste", True),
        prompt_color=getattr(getattr(config, "ui", None), "prompt_color", "cyan"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
