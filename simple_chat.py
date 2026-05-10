import logging
from pathlib import Path
from src.common import load_config
from src.live_ai import Agent

USE_TOOLS:bool = True

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

WORK_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = WORK_DIR / "config.toml"
config = load_config(CONFIG_FILE)
agent = Agent(config=config)

print(f"\n🤖 [{config.brain.model}]: ", end="", flush=True)
agent.act("Say hello to the user and introduce briefly yourself.")

while True:
    try:
        user_input = input("You (leave blank to exit): ")
    except EOFError:
        print()
        break
    if not user_input:
        print("Bye! 👋")
        break
    
    print(f"🤖 [{config.brain.model}]: ", end="", flush=True)
    if USE_TOOLS:
        agent.act(user_input)
    else:
        fragments = agent.stream_fragments(user_input)
        for f in fragments:
            agent.print_fragment(f)
    print()
