"""Minimal MCP stdio echo server for ``tests/test_mcp_integration.py``.

Run directly (``python tests/mcp_echo_server.py``): speaks the MCP protocol
over stdio through the official SDK's server API and exposes a single
``echo`` tool returning ``echo: <text>``. Only used when the ``mcp``
package is installed; every other test runs offline.

Works with both SDK generations: 2.x (``MCPServer``) and 1.x (``FastMCP``).
"""

try:  # SDK >= 2.x: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("echo-server")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back to the caller."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run()  # stdio transport (the default)
