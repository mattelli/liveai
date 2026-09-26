import logging
from pathlib import Path
from src.common import load_config, cprint
from src.live_ai import Agent

USE_TOOLS: bool = True

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

WORK_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = WORK_DIR / "config.toml"
config = load_config(CONFIG_FILE)

cprint(f"Agent Loaded. {'Tools are enabled' if USE_TOOLS else 'Tools are disabled'}", "green")
agent = Agent(config=config)

iteration = 0
while True:
    try:
        iteration += 1
        cprint(f"[{iteration}] You (type '\\x' to exit): ", "yellow", end="")
        user_input = input()
    except EOFError:
        cprint("\n❌ EOFError", "red")
        break

    if user_input.strip() == "\\x":
        cprint("Bye! 👋", "green")
        break

    if USE_TOOLS:
        agent.act(user_input)
    else:
        fragments = agent.model_fragments(user_input)
        for f in fragments:
            agent.print_fragment(f)
        agent.flush_stream()

    print()
