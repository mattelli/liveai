"""MCP (Model Context Protocol) integration: external servers as agent tools.

Configured MCP servers (``[mcp]`` section of config.toml) expose their tools
to the agent in two ways:

- **dynamic injection**: every server tool becomes a native OpenAI
  function-calling tool (``mcp_<server>_<tool>``), built from the MCP
  ``inputSchema`` and callable in the normal ``act()`` loop
  (:func:`mcp_tool_bindings`, used by ``src.live_ai.Agent``);
- **bridge fallback**: :func:`mcp_list_tools` and :func:`mcp_call_tool`
  discover and invoke MCP tools at runtime, covering servers that are
  offline at startup or (re)connected later via ``/mcp refresh``.

The optional ``mcp`` SDK (``pip install mcp``) is imported lazily inside the
connector, so this module imports cleanly without it. All MCP I/O runs on a
single dedicated event-loop thread owned by :class:`Manager`; the public API
is synchronous and safe to call from any agent thread.

``[mcp]`` config section (injected by ``main.py`` via :func:`configure`):

.. code-block:: toml

    [mcp]
    connect_timeout_s = 10.0   # per-server connect + tools/list budget
    call_timeout_s    = 120.0  # per-tool-call budget

    [mcp.servers.filesystem]      # stdio server: local process
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/Max/data"]

    [mcp.servers.deepwiki]        # HTTP server: streamable HTTP endpoint
    url = "https://mcp.deepwiki.com/mcp"

A server entry needs ``command`` (+ optional ``args`` list) **or** ``url``;
anything else is skipped with a warning. Windows note for stdio servers: if
spawning ``npx`` directly fails, use ``command = "cmd"``,
``args = ["/c", "npx", "-y", ...]``. Without the section (or with no usable
servers) the module is a no-op: no manager, no injected tools, no I/O.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.common import dump_func_call

__all__ = [
    "configure", "configured_servers", "get_manager",
    "mcp_call_tool", "mcp_list_tools",
]

logger = logging.getLogger(__name__)

_DEFAULT_CONNECT_TIMEOUT_S = 10.0
_DEFAULT_CALL_TIMEOUT_S = 120.0

# Module state injected by configure() (see the module docstring for the
# [mcp] section). No usable server -> the whole module stays a no-op.
_SERVERS: dict[str, dict] = {}
_CONNECT_TIMEOUT_S: float = _DEFAULT_CONNECT_TIMEOUT_S
_CALL_TIMEOUT_S: float = _DEFAULT_CALL_TIMEOUT_S

_NOT_CONFIGURED = "Error: MCP is not configured (no [mcp] servers)"


def _spec_items(servers) -> list[tuple[str, dict]]:
    """``(name, spec-dict)`` pairs of the ``[mcp.servers]`` table.

    ``load_config`` turns TOML sub-tables into SimpleNamespaces, so both a
    plain dict and a namespace container are accepted; specs are normalized
    to plain dicts either way. Non-mapping entries are dropped.
    """
    if isinstance(servers, dict):
        items = list(servers.items())
    else:
        data = getattr(servers, "__dict__", None)
        items = list(data.items()) if isinstance(data, dict) else []
    normalized: list[tuple[str, dict]] = []
    for name, spec in items:
        spec_dict = spec if isinstance(spec, dict) else getattr(spec, "__dict__", None)
        if isinstance(spec_dict, dict):
            normalized.append((str(name), dict(spec_dict)))
    return normalized


def _timeout(section, attr: str, default: float) -> float:
    """Positive float ``section.attr`` (``default`` when unset/invalid)."""
    raw = getattr(section, attr, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("[mcp] %s=%r is not a number; using %s", attr, raw, default)
        return default
    if value <= 0:
        logger.warning("[mcp] %s=%r must be positive; using %s", attr, raw, default)
        return default
    return value


def configure(config) -> None:
    """Inject the ``[mcp]`` section (called by ``main.py`` via
    ``src.tools.configure``).

    Only parses and stores state: no server is contacted here (connections
    happen lazily on first use). A missing/empty section clears the state,
    so the module stays a no-op.
    """
    global _SERVERS, _CONNECT_TIMEOUT_S, _CALL_TIMEOUT_S
    _SERVERS = {}
    _CONNECT_TIMEOUT_S = _DEFAULT_CONNECT_TIMEOUT_S
    _CALL_TIMEOUT_S = _DEFAULT_CALL_TIMEOUT_S
    section = getattr(config, "mcp", None)
    if section is None:
        return
    parsed: dict[str, dict] = {}
    for name, spec in _spec_items(getattr(section, "servers", None)):
        if spec.get("command") or spec.get("url"):
            parsed[name] = spec
        else:
            logger.warning(
                "[mcp] server %r skipped: needs a 'command' (stdio) "
                "or 'url' (http) key", name)
    _SERVERS = parsed
    _CONNECT_TIMEOUT_S = _timeout(
        section, "connect_timeout_s", _DEFAULT_CONNECT_TIMEOUT_S)
    _CALL_TIMEOUT_S = _timeout(section, "call_timeout_s", _DEFAULT_CALL_TIMEOUT_S)
    if parsed:
        logger.info("[mcp] %d server(s) configured: %s",
                    len(parsed), ", ".join(parsed))
    else:
        logger.debug("[mcp] no usable server configured; MCP integration off")


def configured_servers() -> dict[str, dict]:
    """Configured server specs (``{name: {command|url, args?}}``), by copy."""
    return dict(_SERVERS)


def get_manager():
    """Process-wide :class:`Manager` singleton (``None`` when unconfigured)."""
    return None  # the real singleton arrives with the Manager implementation


@dump_func_call
def mcp_list_tools(server: str = "") -> str:
    """
    List the tools of the configured MCP servers: name, status and one-line
    description each. Without ``server`` all servers are listed. Use this
    first to discover tool names and argument schemas before mcp_call_tool.
    """
    return _NOT_CONFIGURED


@dump_func_call
def mcp_call_tool(server: str, tool: str, arguments: dict) -> str:
    """
    Invoke a tool on a configured MCP server by name (discover names and
    argument schemas with mcp_list_tools first). ``arguments`` is the tool's
    input object; a JSON string is accepted as well.
    """
    return _NOT_CONFIGURED
