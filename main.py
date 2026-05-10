import argparse
import logging
from pathlib import Path
from src.common import load_config
from src.live_ai import LiveAi

WORK_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = WORK_DIR / "config.toml"


def main():
    parser = argparse.ArgumentParser(
        description="Local live AI assistant."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    config = load_config(CONFIG_FILE)
    assistant = LiveAi(config)
    assistant.run()

if __name__ == "__main__":
    main()
