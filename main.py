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
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(args.config)
    from src.session import ConversationSession
    from src.live_ai import Agent  # lazy import: keeps `import main` side-effect free

    agent = Agent(config=config)
    reader = None

    try:
        if args.prompt is not None:
            # Non-interactive: exactly one turn, no stdin reads.
            read_prompt = iter([args.prompt, None]).__next__
        elif args.chat:
            reader = _build_chat_reader(config, ConversationSession)
            read_prompt = lambda: reader.read("[You] (type / for hints): ")  # noqa: E731
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

        return ConversationSession(agent, read_prompt, use_tools=not args.no_tools).run()
    finally:
        if reader is not None:
            reader.close()


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
        history_path=getattr(in_cfg, "history_file", None),
        auto_suggest=getattr(in_cfg, "auto_suggest", True),
        completer=completer,
        multiline=getattr(in_cfg, "multiline", False),
        explicit_paste=getattr(in_cfg, "explicit_paste", True),
    )


if __name__ == "__main__":
    raise SystemExit(main())
