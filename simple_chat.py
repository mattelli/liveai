import logging
from pathlib import Path
from src.common import load_config
from src.live_ai import Brain

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

WORK_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = WORK_DIR / "config.toml"
config = load_config(CONFIG_FILE)
brain = Brain(config=config)

while True:
    try:
        user_input = input("You (leave blank to exit): ")
    except EOFError:
        print()
        break
    if not user_input:
        print("Bye!")
        break
    response = brain.stream_response(user_input)
    print(f"🤖 [{config.brain.model}]: ", end="", flush=True)
    for fragment in response:
        print(fragment.content, end="", flush=True)
    print()