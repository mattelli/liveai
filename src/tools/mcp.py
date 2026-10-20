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

import asyncio
import contextlib
import inspect
import json
import logging
import re
import threading
from contextlib import AsyncExitStack
from typing import Any, Callable, Protocol

from src.common import dump_func_call

__all__ = [
    "Manager", "McpError", "configure", "configured_servers", "get_manager",
    "mcp_call_tool", "mcp_list_tools", "mcp_tool_bindings",
]

logger = logging.getLogger(__name__)

_DEFAULT_CONNECT_TIMEOUT_S = 10.0
_DEFAULT_CALL_TIMEOUT_S = 120.0

# Module state injected by configure() (see the module docstring for the
# [mcp] section). No usable server -> the whole module stays a no-op.
_SERVERS: dict[str, dict] = {}
_CONNECT_TIMEOUT_S: float = _DEFAULT_CONNECT_TIMEOUT_S
_CALL_TIMEOUT_S: float = _DEFAULT_CALL_TIMEOUT_S
_MANAGER: "Manager | None" = None

_NOT_CONFIGURED = "Error: MCP is not configured (no [mcp] servers)"


class McpError(Exception):
    """Raised for MCP manager failures (unknown server/tool, timeout, ...)."""


class Client(Protocol):
    """Transport-level client for one connected MCP server (see Connector)."""

    def list_tools(self) -> list[dict]:
        """``[{"name", "description", "inputSchema"}]`` of the server tools."""
        ...

    def call(self, tool: str, arguments: dict) -> str:
        """Run one tool and return its result as a string."""
        ...

    def close(self) -> None:
        """Release the connection (idempotent, best effort)."""
        ...


class Connector(Protocol):
    """Seam between the Manager and the transport layer (Task 3 SDK connector).

    ``connect`` may be a plain function or a coroutine function: the Manager
    awaits awaitables and uses plain results directly (uniform sync/async
    invocation), so unit tests inject sync fakes while the real connector is
    async.
    """

    def connect(self, name: str, spec: dict) -> Client: ...


# Test seam (module-level like ``_VIDEO_CAPTURE_FACTORY`` in src/tools/camera.py):
# patched by tests to hand out fake connectors; the default builds the real
# SDK connector (Task 3).
_CONNECTOR_FACTORY: Callable[[], Connector] | None = None


def _default_connector() -> Connector:
    """Factory used when no connector is injected (the real SDK connector)."""
    return McpSdkConnector()


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
    global _SERVERS, _CONNECT_TIMEOUT_S, _CALL_TIMEOUT_S, _MANAGER
    _SERVERS = {}
    _MANAGER = None  # a config change invalidates the manager singleton
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


def get_manager() -> "Manager | None":
    """Process-wide :class:`Manager` singleton (``None`` when unconfigured).

    Built lazily on the first call, **only when servers are configured**;
    ``configure()`` itself never connects or spawns threads.
    """
    global _MANAGER
    if not _SERVERS:
        return None
    if _MANAGER is None:
        connector = _CONNECTOR_FACTORY() if _CONNECTOR_FACTORY else _default_connector()
        _MANAGER = Manager(
            dict(_SERVERS),
            connect_timeout_s=_CONNECT_TIMEOUT_S,
            call_timeout_s=_CALL_TIMEOUT_S,
            connector=connector,
        )
    return _MANAGER


class Manager:
    """Owns the MCP connections and a dedicated event-loop thread.

    All MCP I/O happens on that loop; the public API is synchronous and
    callable from any agent thread (``ask_parallel`` fans out on threads).
    Every call runs to completion on the loop before the next one starts, so
    per-server operations are naturally serialized - callers blocking on
    ``fut.result()`` from several threads are safe without extra locking.

    Transport work is delegated to a :class:`Connector` (injected via the
    module-level ``_CONNECTOR_FACTORY`` seam): ``connect`` may be sync or
    async, and client methods may be sync or async - :meth:`_invoke` awaits
    awaitables and returns plain results either way.
    """

    _THREAD_JOIN_TIMEOUT_S = 5.0

    def __init__(self, servers: dict[str, dict],
                 connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
                 call_timeout_s: float = _DEFAULT_CALL_TIMEOUT_S,
                 connector: Connector | None = None):
        self.logger = logger
        self._servers = dict(servers)
        self._connect_timeout_s = connect_timeout_s
        self._call_timeout_s = call_timeout_s
        self._connector = connector or (
            _CONNECTOR_FACTORY() if _CONNECTOR_FACTORY else _default_connector())
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: dict[str, Client] = {}
        self._tool_lists: dict[str, list[dict]] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()  # guards the state dicts only

    ## event-loop thread ##

    def start(self) -> None:
        """Spawn the daemon thread running the event loop (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, name="mcp-manager-loop", daemon=True)
            self._thread.start()

    def _run_loop(self) -> None:
        loop = self._loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_forever()
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _submit(self, fn: Callable, *args, timeout: float | None = None) -> Any:
        """Run ``fn`` on the loop thread and wait for its result.

        Converts timeouts and any failure into a raised :class:`McpError`
        (the caller decides whether to raise or degrade to an error string).
        """
        self.start()
        try:
            coro = self._invoke(fn, *args)
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(
                timeout if timeout is not None else self._connect_timeout_s)
        except McpError:
            raise
        except Exception as exc:
            if isinstance(exc, asyncio.TimeoutError) or "timeout" in str(exc).lower():
                raise McpError(f"timed out after {timeout}s: {exc}") from exc
            raise McpError(str(exc)) from exc

    async def _invoke(self, fn: Callable, *args) -> Any:
        """Await ``fn(*args)`` when it returns an awaitable (sync/async seam)."""
        result = fn(*args)
        if inspect.isawaitable(result):
            result = await result
        return result

    ## connections ##

    def connect_all(self) -> dict[str, list[str]]:
        """Connect every configured server; return ``{server: [tool names]}``.

        A failing server is captured as an error string (never raised) and
        simply misses from the result; it can be retried via :meth:`refresh`.
        Already-connected servers are reused.
        """
        results: dict[str, list[str]] = {}
        for name, spec in self._servers.items():
            if name in self._clients:
                results[name] = [tool.get("name", "")
                                 for tool in self._tool_lists.get(name, [])]
                continue
            try:
                client = self._submit(self._connector.connect, name, spec)
                tools = self._submit(client.list_tools)
            except Exception as exc:
                self._errors[name] = str(exc)
                self.logger.warning("[mcp] server %r failed to connect: %s",
                                    name, exc)
                continue
            self._clients[name] = client
            self._tool_lists[name] = list(tools or [])
            self._errors.pop(name, None)
            results[name] = [tool.get("name", "") for tool in self._tool_lists[name]]
            self.logger.info("[mcp] server %r connected with %d tool(s)",
                             name, len(results[name]))
        return results

    def _require_client(self, server: str) -> Client:
        """Connected client of ``server`` (:class:`McpError` otherwise)."""
        if server not in self._servers:
            raise McpError(f"unknown server {server!r}. Configured: "
                           + (", ".join(self._servers) or "(none)"))
        client = self._clients.get(server)
        if client is None:
            error = self._errors.get(server)
            raise McpError(
                f"server {server!r} is not connected"
                + (f" ({error})" if error else ""))
        return client

    def list_tools(self, server: str) -> list[dict]:
        """Tool records (``{"name", "description", "inputSchema"}``) of a server."""
        client = self._require_client(server)
        return self._submit(client.list_tools)

    def call(self, server: str, tool: str, arguments: dict) -> str:
        """Run one tool on ``server``; returns the result string.

        Raises :class:`McpError` on unknown server/tool, unconnected server
        or timeout (``call_timeout_s`` budget).
        """
        client = self._require_client(server)
        tools = {t.get("name") for t in self._tool_lists.get(server, [])}
        if tool not in tools:
            raise McpError(f"server {server!r} has no tool {tool!r}")
        return self._submit(client.call, tool, arguments,
                            timeout=self._call_timeout_s)

    def status(self) -> dict[str, dict]:
        """Per-server view for ``/mcp``: transport, tools, error state."""
        view: dict[str, dict] = {}
        for name, spec in self._servers.items():
            entry: dict[str, Any] = {
                "transport": "http" if spec.get("url") else "stdio",
            }
            if name in self._clients:
                entry["status"] = "connected"
                entry["tools"] = [t.get("name", "")
                                  for t in self._tool_lists.get(name, [])]
            elif name in self._errors:
                entry["status"] = f"error: {self._errors[name]}"
                entry["tools"] = []
            else:
                entry["status"] = "not connected"
                entry["tools"] = []
            view[name] = entry
        return view

    def refresh(self) -> dict[str, list[str]]:
        """Close every client and reconnect from scratch (same shape as
        :meth:`connect_all`)."""
        self._close_clients()
        return self.connect_all()

    def _close_clients(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                self._submit(client.close)
            except Exception as exc:
                self.logger.debug("[mcp] closing %r failed: %s", name, exc)
        self._clients.clear()
        self._tool_lists.clear()

    def shutdown(self) -> None:
        """Close clients, stop the loop thread and drop the singleton state."""
        with self._lock:
            if self._loop is None:
                return
        self._close_clients()
        loop, self._loop = self._loop, None
        thread, self._thread = self._thread, None
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass  # loop already closed
        if thread is not None:
            thread.join(timeout=self._THREAD_JOIN_TIMEOUT_S)


## real connector (official ``mcp`` SDK, imported lazily) ##


def _flatten_result(result) -> str:
    """Flatten an MCP ``CallToolResult`` into the string returned to the model.

    Joins the ``text`` parts of ``result.content``; when there is none and
    ``result.structuredContent`` exists, that object is JSON-encoded. An
    ``isError`` result is prefixed with ``Error: `` (the agent tool-error
    contract). Pure function: takes/returns plain values only, so tests can
    use SimpleNamespace stand-ins without the SDK.
    """
    content = getattr(result, "content", None) or []
    parts = [getattr(part, "text", None) for part in content
             if getattr(part, "type", None) == "text" or hasattr(part, "text")]
    text = "\n".join(str(part) for part in parts if part is not None)
    if not text:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            text = json.dumps(structured)
    if not text:
        text = "(empty result)"
    if getattr(result, "isError", False):
        text = f"Error: {text}"
    return text


class _SdkClient:
    """Client backed by an initialized MCP SDK ``ClientSession``.

    The transports/session live on the connector's **serve task** (anyio
    transports must be entered and exited by the same task), so ``close()``
    only sets the close event and waits for that task to unwind its own
    ``AsyncExitStack``. All methods are async and awaited by the Manager's
    ``_invoke``.
    """

    _CLOSE_JOIN_TIMEOUT_S = 5.0

    def __init__(self, session, serve_task: asyncio.Task,
                 close_event: asyncio.Event, call_timeout_s: float):
        self._session = session
        self._serve_task = serve_task
        self._close_event = close_event
        self._call_timeout_s = call_timeout_s

    async def list_tools(self) -> list[dict]:
        """Normalize the ``tools/list`` result into plain dicts."""
        result = await self._session.list_tools()
        tools = []
        for tool in getattr(result, "tools", []) or []:
            tools.append({
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", "") or "",
                "inputSchema": getattr(tool, "inputSchema", None),
            })
        return tools

    async def call(self, tool: str, arguments: dict) -> str:
        """Run one tool; flatten the result (``Error: ...`` when ``isError``)."""
        result = await asyncio.wait_for(
            self._session.call_tool(tool, arguments or {}), self._call_timeout_s)
        return _flatten_result(result)

    async def close(self) -> None:
        """Signal the serve task to exit its stack (best effort, bounded wait)."""
        self._close_event.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(self._serve_task), self._CLOSE_JOIN_TIMEOUT_S)
        except Exception:
            pass  # best effort: the daemon loop thread dies with the process


class McpSdkConnector:
    """Connector speaking the real MCP protocol through the official SDK.

    For each server a **serve task** is spawned on the Manager's loop: it
    enters the transport (stdio subprocess or streamable HTTP) plus a
    ``ClientSession`` in its own ``AsyncExitStack``, runs ``initialize()``
    and then waits on a close event. ``connect()`` resolves as soon as the
    session is ready; ``close()`` sets the event so the serve task exits the
    stack itself. An inner connect timeout (the configured
    ``connect_timeout_s``) unblocks ``connect`` early and tears the partial
    session down; the Manager's outer ``_submit`` timeout stays as the hard
    net.
    """

    def connect(self, name: str, spec: dict):
        """Coroutine: connect to one server and return an :class:`_SdkClient`."""
        return self._connect(name, spec)

    async def _connect(self, name: str, spec: dict) -> _SdkClient:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise McpError(
                "mcp package not installed (pip install mcp)") from exc

        stack = AsyncExitStack()
        ready = asyncio.Event()
        close_event = asyncio.Event()
        holder: dict[str, Any] = {}

        async def serve() -> None:
            try:
                if spec.get("url"):
                    transport = await stack.enter_async_context(
                        streamablehttp_client(str(spec["url"])))
                else:
                    params = StdioServerParameters(
                        command=str(spec["command"]),
                        args=[str(arg) for arg in spec.get("args", []) or []],
                    )
                    transport = await stack.enter_async_context(
                        stdio_client(params))
                read, write = transport[0], transport[1]
                session = await stack.enter_async_context(
                    ClientSession(read, write))
                await session.initialize()
                holder["client"] = _SdkClient(
                    session, serve_task, close_event, _CALL_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 - reported via holder
                holder["error"] = exc
            finally:
                ready.set()
            if "client" not in holder:
                with contextlib.suppress(Exception):
                    await stack.aclose()  # connect failed: unwind right here
                return
            await close_event.wait()  # session lifetime (until close())
            with contextlib.suppress(Exception):
                await stack.aclose()

        serve_task = asyncio.create_task(serve(), name=f"mcp-serve-{name}")
        try:
            await asyncio.wait_for(ready.wait(), _CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            close_event.set()  # unblock/tear down the serve task
            with contextlib.suppress(Exception):
                await serve_task
            raise McpError(f"connecting {name!r} timed out") from exc
        except asyncio.CancelledError:
            close_event.set()
            with contextlib.suppress(Exception):
                await serve_task
            raise
        if "error" in holder:
            await serve_task  # serve already unwound its stack
            raise McpError(str(holder["error"]))
        return holder["client"]


@dump_func_call
def mcp_list_tools(server: str = "") -> str:
    """
    List the tools of the configured MCP servers: name, status and one-line
    description each. Without ``server`` all servers are listed. Use this
    first to discover tool names and argument schemas before mcp_call_tool.
    """
    manager = get_manager()
    if manager is None:
        return _NOT_CONFIGURED
    try:
        manager.connect_all()  # cheap when already connected
    except Exception as exc:  # pragma: no cover - connect_all never raises
        return f"Error: {exc}"
    status = manager.status()
    names = [server] if server else list(status)
    if server and server not in status:
        return (f"Error: unknown MCP server {server!r}. Configured: "
                + (", ".join(status) or "(none)"))
    lines = []
    for name in names:
        entry = status[name]
        lines.append(f"{name} [{entry['transport']}] {entry['status']}")
        if not entry["tools"]:
            lines.append("  (no tools)")
            continue
        descriptions: dict[str, str] = {}
        try:
            for tool in manager.list_tools(name):
                descriptions[tool.get("name", "")] = str(
                    tool.get("description") or "").splitlines()[0] if (
                    tool.get("description") or "").strip() else ""
        except Exception as exc:
            descriptions = {}
            lines[-1] += f" (tool details unavailable: {exc})"
        for tool_name in entry["tools"]:
            description = descriptions.get(tool_name, "")
            lines.append(f"  - {tool_name}" + (f": {description}" if description else ""))
    return "\n".join(lines) if lines else "No MCP servers configured."


@dump_func_call
def mcp_call_tool(server: str, tool: str, arguments: dict) -> str:
    """
    Invoke a tool on a configured MCP server by name (discover names and
    argument schemas with mcp_list_tools first). ``arguments`` is the tool's
    input object; a JSON string is accepted as well.
    """
    manager = get_manager()
    if manager is None:
        return _NOT_CONFIGURED
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: arguments is not valid JSON: {exc}"
    if not isinstance(arguments, dict):
        return f"Error: arguments must be a JSON object, got {type(arguments).__name__}"
    try:
        return str(manager.call(server, tool, arguments))
    except McpError as exc:
        return f"Error: {exc}"


## dynamic tool injection (native OpenAI function-calling tools) ##


def _sanitize_tool_name(server: str, tool: str) -> str:
    """``mcp_<server>_<tool>`` shaped for OpenAI: ``^[a-zA-Z0-9_-]{1,64}$``."""
    raw = f"mcp_{server}_{tool}"
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64]
    return sanitized if sanitized else "mcp_tool"


def _resolve_name_collision(tool_name: str, taken: set[str]) -> str:
    """First free variant of ``tool_name`` (``_2``, ``_3`` ... on collision).

    Never silently shadows a built-in tool such as ``web_search``: the first
    collision gets ``<name>_2``, the second ``<name>_3``, and so on.
    """
    if tool_name not in taken:
        return tool_name
    suffix = 2
    while f"{tool_name}_{suffix}" in taken:
        suffix += 1
    return f"{tool_name}_{suffix}"


def mcp_tool_bindings(existing_names: "set[str] | None" = None,
                      manager=None) -> tuple[list[dict], dict[str, Callable]]:
    """Build OpenAI tool definitions + wrapper callables for every MCP tool.

    Returns ``([], {})`` when no manager exists (cheap: no connection
    attempt). Otherwise ``connect_all()`` runs first (already-connected
    servers are reused), then every server tool becomes:

    - a definition ``{"type": "function", "function": {...}}`` whose
      ``parameters`` is the MCP ``inputSchema`` **verbatim** (the schema
      comes from the server, so ``_openai_tool_definition`` is bypassed);
      tools with a missing/invalid ``inputSchema`` are skipped with a warning,
    - a wrapper callable ``manager.call(server, tool, kwargs)`` named
      ``mcp_<server>_<tool>`` (sanitized, truncated to 64 chars; collisions
      with ``existing_names`` or other MCP tools get a ``_2``, ``_3`` ...
      suffix).

    ``existing_names`` are the tool names already registered by the agent,
    so MCP tools never silently shadow a built-in.
    """
    if manager is None:
        manager = get_manager()
    if manager is None:
        return [], {}
    try:
        manager.connect_all()  # cheap when already connected
    except Exception as exc:  # pragma: no cover - connect_all never raises
        logger.warning("[mcp] connect_all failed during bindings: %s", exc)
    definitions: list[dict] = []
    impls: dict[str, Callable] = {}
    taken: set[str] = set(existing_names or ())
    for server, entry in manager.status().items():
        if entry.get("status") != "connected":
            continue  # failed servers contribute no tools (G4)
        try:
            tools = manager.list_tools(server)
        except Exception as exc:
            logger.warning("[mcp] listing tools of %r failed: %s", server, exc)
            continue
        for tool in tools:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            schema = tool.get("inputSchema")
            if not isinstance(schema, dict):
                logger.warning(
                    "[mcp] tool %r of server %r skipped: no usable inputSchema",
                    name, server)
                continue
            tool_name = _resolve_name_collision(
                _sanitize_tool_name(server, name), taken)
            taken.add(tool_name)
            description = str(tool.get("description") or "").strip()
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": schema,
                },
            })

            def make_wrapper(server: str = server, tool: str = name,
                             tool_name: str = tool_name) -> Callable:
                def wrapper(**kwargs) -> str:
                    try:
                        return str(manager.call(server, tool, kwargs))
                    except McpError as exc:
                        return f"Error: {exc}"
                wrapper.__name__ = tool_name
                wrapper.__doc__ = (
                    f"MCP tool '{name}' of server '{server}'."
                    + (f" {description}" if description else ""))
                return wrapper

            impls[tool_name] = make_wrapper()
    return definitions, impls
