import argparse
import logging
from pathlib import Path

from src.common import load_config, prompt_input

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

    from src.live_ai import Agent  # lazy import: keeps `import main` side-effect free

    agent = Agent(config=config)

    if args.prompt is not None:
        from src.session import ConversationSession

        # Non-interactive: exactly one turn, no stdin reads.
        read_prompt = iter([args.prompt, None]).__next__
        return ConversationSession(agent, read_prompt, use_tools=not args.no_tools).run()

    if args.chat:
        from src.session import ConversationSession

        read_prompt = lambda: prompt_input("[You] (type /quit to exit): ")  # noqa: E731
    else:
        from src.audio_wrapper import SpeechDetector  # lazy import

        detector = SpeechDetector(
            model_path=config.speech_det.model_path,
            language=config.speech_det.language,
            show_levels=True,
        )
        read_prompt = lambda: detector.listen_and_transcribe(  # noqa: E731
            silence_timeout_s=config.speech_det.silence_timeout_s
        )

    from src.session import ConversationSession

    return ConversationSession(agent, read_prompt, use_tools=not args.no_tools).run()


if __name__ == "__main__":
    raise SystemExit(main())
