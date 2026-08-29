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
print(f"Agent Loaded.", "Tools are enabled" if USE_TOOLS else "")
agent = Agent(config=config)

iteration = 0
while True:
    try:
        iteration += 1
        user_input = input(f"[{iteration}] You (leave blank to exit): ")
    except EOFError:
        print()
        break
    if not user_input:
        print("Bye! 👋")
        break

    if USE_TOOLS:
        agent.act(user_input)
    else:
        fragments = agent.model_fragments(user_input)
        for f in fragments:
            agent.print_fragment(f)
    print()
