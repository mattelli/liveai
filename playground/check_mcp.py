"""Quick check of the MCP integration against real servers (playground).

Usage:
    python playground/check_mcp.py                 # deepwiki (HTTP) + echo (stdio)
    python playground/check_mcp.py <name> ...      # only the named servers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.mcp import Manager  # noqa: E402

SERVERS = {
    "deepwiki": {"url": "https://mcp.deepwiki.com/mcp"},
    "echo": {"command": sys.executable,
             "args": [str(Path(__file__).resolve().parents[1]
                          / "tests" / "mcp_echo_server.py")]},
}


def main(argv: list[str]) -> None:
    names = argv or list(SERVERS)
    servers = {name: SERVERS[name] for name in names if name in SERVERS}
    manager = Manager(servers, connect_timeout_s=30.0, call_timeout_s=60.0)
    try:
        result = manager.connect_all()
        print("== connect_all ==")
        for name, tools in result.items():
            print(f"  {name}: {len(tools)} tool(s)")
        for name, error in manager._errors.items():
            print(f"  {name}: ERROR -> {error}")

        print("\n== status ==")
        for name, entry in manager.status().items():
            print(f"  {name} [{entry['transport']}] {entry['status']}")
            for tool in entry["tools"][:10]:
                print(f"    - {tool}")
            if len(entry["tools"]) > 10:
                print(f"    ... and {len(entry['tools']) - 10} more")

        if "echo" in manager._clients:
            print("\n== call echo tool ==")
            print("  ->", manager.call("echo", "echo", {"text": "hello from live assistant"}))

        if "deepwiki" in manager._clients:
            print("\n== call deepwiki read_wiki_structure ==")
            try:
                out = manager.call("deepwiki", "read_wiki_structure", {
                    "repoName": "modelcontextprotocol/python-sdk"})
                print("  ->", out[:400].replace("\n", " | "))
            except Exception as exc:
                print(f"  call failed: {exc}")
    finally:
        manager.shutdown()
        print("\n== shutdown OK ==")


if __name__ == "__main__":
    main(sys.argv[1:])
