"""Unit tests for the MCP tool module (offline: no real MCP server, no SDK)."""

import threading
import time
from types import SimpleNamespace

import pytest

import src.tools.mcp as mcp_mod
from src.tools.mcp import McpError, Manager, configure, configured_servers


def _mcp_config(**overrides):
    """A config object carrying a ``[mcp]`` section (as ``load_config`` builds it)."""
    defaults = dict(
        connect_timeout_s=10.0,
        call_timeout_s=120.0,
        servers=SimpleNamespace(
            filesystem=SimpleNamespace(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "C:/data"],
            ),
            deepwiki=SimpleNamespace(url="https://mcp.deepwiki.com/mcp"),
        ),
    )
    defaults.update(overrides)
    return SimpleNamespace(mcp=SimpleNamespace(**defaults))


def _reset_state():
    configure(SimpleNamespace())  # no [mcp] section: clears all module state


def setup_function(_):
    _reset_state()


def teardown_function(_):
    _reset_state()


# ------------------------------------------------------------- configure()


def test_configure_without_section_is_noop():
    configure(SimpleNamespace())
    assert configured_servers() == {}
    assert mcp_mod._CONNECT_TIMEOUT_S == 10.0
    assert mcp_mod._CALL_TIMEOUT_S == 120.0


def test_configure_empty_servers_is_noop():
    configure(SimpleNamespace(mcp=SimpleNamespace()))
    assert configured_servers() == {}


def test_configure_parses_stdio_and_url_specs():
    configure(_mcp_config())
    servers = configured_servers()
    assert set(servers) == {"filesystem", "deepwiki"}
    # Specs are stored verbatim (normalized to plain dicts).
    assert servers["filesystem"]["command"] == "npx"
    assert servers["filesystem"]["args"] == [
        "-y", "@modelcontextprotocol/server-filesystem", "C:/data"]
    assert servers["deepwiki"]["url"] == "https://mcp.deepwiki.com/mcp"


def test_configure_skips_invalid_server_spec():
    configure(_mcp_config(servers=SimpleNamespace(
        broken=SimpleNamespace(description="neither command nor url"),
        good=SimpleNamespace(url="https://mcp.example/mcp"),
    )))
    assert configured_servers() == {"good": {"url": "https://mcp.example/mcp"}}


def test_configure_timeout_defaults_and_overrides():
    configure(SimpleNamespace(mcp=SimpleNamespace(
        servers=SimpleNamespace(s=SimpleNamespace(url="u")))))
    assert mcp_mod._CONNECT_TIMEOUT_S == 10.0
    assert mcp_mod._CALL_TIMEOUT_S == 120.0

    configure(SimpleNamespace(mcp=SimpleNamespace(
        connect_timeout_s=2.5, call_timeout_s=30,
        servers=SimpleNamespace(s=SimpleNamespace(url="u")))))
    assert mcp_mod._CONNECT_TIMEOUT_S == 2.5
    assert mcp_mod._CALL_TIMEOUT_S == 30.0


# ------------------------------------------------- bridge tools (Task 1 stubs)


def test_mcp_bridge_tools_exist_in_default_tools():
    import src.live_ai as live_ai_mod

    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "mcp_list_tools" in names
    assert "mcp_call_tool" in names


def test_bridge_tools_return_error_without_config():
    assert mcp_mod.mcp_list_tools() == mcp_mod._NOT_CONFIGURED
    assert mcp_mod.mcp_call_tool("srv", "tool", {}) == mcp_mod._NOT_CONFIGURED


# ------------------------------------------------------------------- Manager


class FakeClient:
    """Sync Client double: records calls, returns scripted results."""

    def __init__(self, tools, results=None, call_delay_s=0.0):
        self.tools = tools
        self.results = results or {}
        self.call_delay_s = call_delay_s
        self.calls: list[tuple[str, dict]] = []
        self.closed = 0

    def list_tools(self):
        return list(self.tools)

    def call(self, tool, arguments):
        if self.call_delay_s:
            time.sleep(self.call_delay_s)
        self.calls.append((tool, arguments))
        return self.results.get(tool, f"ran {tool} with {arguments}")

    def close(self):
        self.closed += 1


class FakeConnector:
    """Sync Connector double: builds FakeClients and records the specs."""

    def __init__(self, tools=None, fail_for=(), call_delay_s=0.0):
        self.tools = tools or {"srv": [{"name": "echo", "description": "Echo it",
                                        "inputSchema": {"type": "object"}}]}
        self.fail_for = {fail_for} if isinstance(fail_for, str) else set(fail_for)
        self.call_delay_s = call_delay_s
        self.connected: list[tuple[str, dict]] = []
        self.clients: dict[str, FakeClient] = {}

    def connect(self, name, spec):
        self.connected.append((name, dict(spec)))
        if name in self.fail_for:
            raise ConnectionError(f"cannot reach {name}")
        client = FakeClient(self.tools.get(name, []), call_delay_s=self.call_delay_s)
        self.clients[name] = client
        return client


def _make_manager(connector=None, servers=None, **kw):
    manager = Manager(
        servers or {"srv": {"url": "https://mcp.example/mcp"}},
        connect_timeout_s=kw.pop("connect_timeout_s", 5.0),
        call_timeout_s=kw.pop("call_timeout_s", 5.0),
        connector=connector or FakeConnector(),
        **kw,
    )
    return manager


@pytest.fixture
def shutdown_managers():
    """Tear down every Manager created by a test (stops its loop thread)."""
    managers: list[Manager] = []
    original_init = Manager.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        managers.append(self)

    import unittest.mock as mock
    with mock.patch.object(Manager, "__init__", tracking_init):
        yield managers
    for manager in managers:
        manager.shutdown()


@pytest.fixture
def reset_singleton():
    """Keep the module-level manager singleton clean around each test."""
    mcp_mod._MANAGER = None
    yield
    if mcp_mod._MANAGER is not None:
        mcp_mod._MANAGER.shutdown()
    mcp_mod._MANAGER = None


def test_manager_connect_all_returns_tool_lists(shutdown_managers):
    connector = FakeConnector(tools={
        "srv": [{"name": "echo", "description": "Echo it", "inputSchema": {}}],
        "other": [{"name": "ping", "description": "", "inputSchema": {}}],
    })
    manager = _make_manager(connector, servers={"srv": {"url": "u"},
                                                "other": {"command": "c"}})
    assert manager.connect_all() == {"srv": ["echo"], "other": ["ping"]}
    assert [name for name, _ in connector.connected] == ["srv", "other"]


def test_manager_connect_all_captures_per_server_errors(shutdown_managers):
    connector = FakeConnector(tools={
        "working": [{"name": "echo", "description": "Echo it", "inputSchema": {}}],
    }, fail_for="broken")
    manager = _make_manager(connector, servers={
        "broken": {"url": "u"}, "working": {"url": "u"}})
    result = manager.connect_all()
    assert result == {"working": ["echo"]}  # the healthy server still connects
    assert "broken" not in manager._clients
    assert "cannot reach broken" in manager._errors["broken"]


def test_manager_connect_all_reuses_connected_servers(shutdown_managers):
    connector = FakeConnector()
    manager = _make_manager(connector)
    manager.connect_all()
    assert manager.connect_all() == {"srv": ["echo"]}
    assert len(connector.connected) == 1  # no second connect attempt


def test_manager_call_returns_client_result(shutdown_managers):
    manager = _make_manager()
    manager.connect_all()
    assert manager.call("srv", "echo", {"text": "hi"}) == "ran echo with {'text': 'hi'}"


def test_manager_call_unknown_server_raises(shutdown_managers):
    manager = _make_manager()
    with pytest.raises(McpError, match="unknown server 'nope'"):
        manager.call("nope", "echo", {})


def test_manager_call_unknown_tool_raises(shutdown_managers):
    manager = _make_manager()
    manager.connect_all()
    with pytest.raises(McpError, match="no tool 'missing'"):
        manager.call("srv", "missing", {})


def test_manager_call_unconnected_server_raises(shutdown_managers):
    connector = FakeConnector(fail_for="srv")
    manager = _make_manager(connector)
    manager.connect_all()  # fails, error captured
    with pytest.raises(McpError, match="not connected"):
        manager.call("srv", "echo", {})


def test_manager_call_timeout_raises(shutdown_managers):
    connector = FakeConnector(call_delay_s=0.5)
    manager = _make_manager(connector, call_timeout_s=0.2)
    manager.connect_all()
    with pytest.raises(McpError, match="timed out"):
        manager.call("srv", "echo", {})


def test_manager_list_tools_unconnected_raises(shutdown_managers):
    manager = _make_manager()
    with pytest.raises(McpError, match="not connected"):
        manager.list_tools("srv")


def test_manager_list_tools_returns_records(shutdown_managers):
    manager = _make_manager()
    manager.connect_all()
    assert manager.list_tools("srv") == [
        {"name": "echo", "description": "Echo it", "inputSchema": {"type": "object"}}]


def test_manager_refresh_reconnects_and_replaces_clients(shutdown_managers):
    connector = FakeConnector()
    manager = _make_manager(connector)
    manager.connect_all()
    first_client = manager._clients["srv"]
    assert manager.refresh() == {"srv": ["echo"]}
    assert manager._clients["srv"] is not first_client
    assert first_client.closed == 1
    assert len(connector.connected) == 2


def test_manager_shutdown_closes_clients_and_stops_thread(shutdown_managers):
    connector = FakeConnector()
    manager = _make_manager(connector)
    manager.connect_all()
    thread = manager._thread
    manager.shutdown()
    assert connector.clients["srv"].closed == 1
    assert not thread.is_alive()
    assert manager._loop is None


def test_manager_invokes_async_connector_methods(shutdown_managers):
    """The _invoke coroutine contract: async connector/client methods work."""

    class AsyncConnector:
        def __init__(self):
            self.client = FakeClient(
                [{"name": "echo", "description": "", "inputSchema": {}}])

        async def connect(self, name, spec):
            return self.client

    async_client = AsyncConnector().client
    original_call = async_client.call

    async def async_call(tool, arguments):
        return await original_call(tool, arguments)

    async_client.call = async_call

    manager = _make_manager(AsyncConnector())
    assert manager.connect_all() == {"srv": ["echo"]}
    assert manager.call("srv", "echo", {"a": 1}) == "ran echo with {'a': 1}"


def test_manager_calls_safe_from_multiple_threads(shutdown_managers):
    """G5: concurrent callers on one manager all get their own result."""
    manager = _make_manager()
    manager.connect_all()
    results: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(index):
        try:
            result = manager.call("srv", "echo", {"i": index})
            with lock:
                results.append((index, result))
        except Exception as exc:  # pragma: no cover - must not happen
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 8
    assert all(f"'i': {index}" in result for index, result in results)


# ------------------------------------------------------------- get_manager()


def test_get_manager_none_without_servers(reset_singleton):
    configure(SimpleNamespace())
    assert mcp_mod.get_manager() is None


def test_get_manager_builds_singleton_with_servers(reset_singleton, monkeypatch):
    connector = FakeConnector()
    monkeypatch.setattr(mcp_mod, "_CONNECTOR_FACTORY", lambda: connector)
    configure(_mcp_config())
    manager = mcp_mod.get_manager()
    assert manager is mcp_mod.get_manager()  # same singleton on repeat calls
    assert isinstance(manager, Manager)
    assert manager._servers == configured_servers()
    manager.shutdown()
    mcp_mod._MANAGER = None


def test_configure_resets_manager_singleton(reset_singleton, monkeypatch):
    connector = FakeConnector()
    monkeypatch.setattr(mcp_mod, "_CONNECTOR_FACTORY", lambda: connector)
    configure(_mcp_config())
    manager = mcp_mod.get_manager()
    configure(SimpleNamespace())  # config change invalidates the singleton
    assert mcp_mod._MANAGER is None
    assert mcp_mod.get_manager() is None  # no servers -> stays None
    manager.shutdown()


# ------------------------------------------------- _flatten_result (pure fn)


def test_flatten_result_joins_text_content():
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello"),
                 SimpleNamespace(type="text", text="world")],
        structuredContent=None, isError=False)
    assert mcp_mod._flatten_result(result) == "hello\nworld"


def test_flatten_result_uses_structured_content():
    result = SimpleNamespace(
        content=[], structuredContent={"answer": 42}, isError=False)
    assert mcp_mod._flatten_result(result) == '{"answer": 42}'


def test_flatten_result_marks_is_error():
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="boom")],
        structuredContent=None, isError=True)
    assert mcp_mod._flatten_result(result) == "Error: boom"


def test_flatten_result_empty_returns_placeholder():
    result = SimpleNamespace(content=[], structuredContent=None, isError=False)
    assert mcp_mod._flatten_result(result) == "(empty result)"


# ------------------------------------------------- real connector (fake SDK)


class _FakeEnterable:
    """Async context manager returning a fixed value (transport stand-in)."""

    def __init__(self, value):
        self.value = value
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self.value

    async def __aexit__(self, *exc):
        self.exited += 1
        return False


class _FakeSession:
    """ClientSession stand-in: records initialize/list/call."""

    instances: list["_FakeSession"] = []

    def __init__(self, read, write):
        self.read, self.write = read, write
        self.initialized = 0
        self.calls: list[tuple[str, dict]] = []
        _FakeSession.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        self.initialized += 1

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(name="echo", description="Echo it",
                            inputSchema={"type": "object"}),
        ])

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"echo:{tool}")],
            structuredContent=None, isError=False)


def _install_fake_sdk(monkeypatch):
    """Patch sys.modules with a fake ``mcp`` SDK (works with or without it)."""
    import sys
    import types

    _FakeSession.instances = []
    fake_mcp = types.ModuleType("mcp")
    fake_mcp.ClientSession = _FakeSession

    params_seen: list[tuple[str, list]] = []

    class FakeStdioServerParameters:
        def __init__(self, command=None, args=None):
            self.command = command
            self.args = args
            params_seen.append((command, list(args or [])))

    fake_mcp.StdioServerParameters = FakeStdioServerParameters

    def fake_stdio_client(params):
        return _FakeEnterable((SimpleNamespace(), SimpleNamespace()))

    def fake_http_client(url):
        return _FakeEnterable((SimpleNamespace(), SimpleNamespace(), {}))

    stdio_mod = types.ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = fake_stdio_client
    http_mod = types.ModuleType("mcp.client.streamable_http")
    http_mod.streamablehttp_client = fake_http_client
    client_pkg = types.ModuleType("mcp.client")
    client_pkg.stdio = stdio_mod
    client_pkg.streamable_http = http_mod
    fake_mcp.client = client_pkg

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client_pkg)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", http_mod)
    return params_seen


def test_connector_missing_sdk_raises_mcperror(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mcp", None)  # import fails even if installed
    connector = mcp_mod.McpSdkConnector()
    with pytest.raises(McpError, match="pip install mcp"):
        import asyncio
        asyncio.run(connector._connect("srv", {"url": "https://x/mcp"}))


def test_connector_builds_stdio_parameters(monkeypatch):
    import asyncio
    params_seen = _install_fake_sdk(monkeypatch)
    connector = mcp_mod.McpSdkConnector()
    client = asyncio.run(connector._connect("files", {
        "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem"]}))
    assert params_seen == [("npx", ["-y", "@modelcontextprotocol/server-filesystem"])]
    assert isinstance(client, mcp_mod._SdkClient)
    session = _FakeSession.instances[-1]
    assert session.initialized == 1


def test_connector_sdk_client_end_to_end(monkeypatch, shutdown_managers):
    """Fake SDK through the real Manager: connect, list, call, close."""
    import asyncio
    _install_fake_sdk(monkeypatch)
    manager = Manager({"srv": {"url": "https://x/mcp"}},
                      connect_timeout_s=5.0, call_timeout_s=5.0)
    result = manager.connect_all()
    assert result == {"srv": ["echo"]}
    assert manager.call("srv", "echo", {"text": "hi"}) == "echo:echo"
    session = _FakeSession.instances[-1]
    assert session.calls == [("echo", {"text": "hi"})]
    manager.shutdown()
    assert _FakeSession.instances[-1].initialized == 1


def test_connector_http_and_stdio_transports_selected(monkeypatch):
    """url spec -> streamable HTTP context; command spec -> stdio context."""
    import asyncio
    entered = []
    params_seen = _install_fake_sdk(monkeypatch)

    import sys
    http_ctx = sys.modules["mcp.client.streamable_http"]
    original = http_ctx.streamablehttp_client

    def spy_http(url):
        entered.append(("http", url))
        return original(url)

    http_ctx.streamablehttp_client = spy_http
    connector = mcp_mod.McpSdkConnector()
    asyncio.run(connector._connect("web", {"url": "https://mcp.example/mcp"}))
    assert entered == [("http", "https://mcp.example/mcp")]
    assert params_seen == []  # stdio parameters never built for a url spec


# ------------------------------------------------------------- bridge tools


class FakeManager:
    """Manager double for the bridge tools (records calls, fixed status)."""

    def __init__(self, servers=None, results=None):
        self.status_value = servers or {
            "srv": {"transport": "stdio", "status": "connected",
                    "tools": ["echo"]},
            "down": {"transport": "http", "status": "error: cannot reach down",
                     "tools": []},
        }
        self.results = results or {}
        self.calls: list[tuple[str, str, dict]] = []
        self.connect_calls = 0

    def connect_all(self):
        self.connect_calls += 1
        return {}

    def status(self):
        return self.status_value

    def list_tools(self, server):
        return [{"name": "echo", "description": "Echo it\nlong line",
                 "inputSchema": {"type": "object"}}]

    def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return self.results.get((server, tool), f"ran {tool}")


def test_mcp_list_tools_without_manager_returns_error(monkeypatch):
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    assert mcp_mod.mcp_list_tools() == mcp_mod._NOT_CONFIGURED


def test_mcp_call_tool_without_manager_returns_error(monkeypatch):
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    assert mcp_mod.mcp_call_tool("srv", "echo", {}) == mcp_mod._NOT_CONFIGURED


def test_mcp_call_tool_delegates_to_manager(monkeypatch):
    fake = FakeManager(results={("srv", "echo"): "echoed!"})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: fake)
    assert mcp_mod.mcp_call_tool("srv", "echo", {"text": "hi"}) == "echoed!"
    assert fake.calls == [("srv", "echo", {"text": "hi"})]


def test_mcp_call_tool_accepts_json_string_arguments(monkeypatch):
    fake = FakeManager()
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: fake)
    mcp_mod.mcp_call_tool("srv", "echo", '{"text": "hi"}')
    assert fake.calls == [("srv", "echo", {"text": "hi"})]


def test_mcp_call_tool_rejects_invalid_json_arguments(monkeypatch):
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: FakeManager())
    result = mcp_mod.mcp_call_tool("srv", "echo", "{not json")
    assert result.startswith("Error: arguments is not valid JSON")


def test_mcp_call_tool_maps_mcperror_to_error_string(monkeypatch):
    class Failing(FakeManager):
        def call(self, server, tool, arguments):
            raise McpError("server 'srv' has no tool 'missing'")

    monkeypatch.setattr(mcp_mod, "get_manager", lambda: Failing())
    result = mcp_mod.mcp_call_tool("srv", "missing", {})
    assert result == "Error: server 'srv' has no tool 'missing'"


def test_mcp_list_tools_renders_all_servers(monkeypatch):
    fake = FakeManager()
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: fake)
    out = mcp_mod.mcp_list_tools()
    assert "srv [stdio] connected" in out
    assert "down [http] error: cannot reach down" in out
    assert "- echo: Echo it" in out  # one-line description only
    assert "long line" not in out
    assert fake.connect_calls == 1  # connect_all called for discovery


def test_mcp_list_tools_single_server(monkeypatch):
    fake = FakeManager()
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: fake)
    out = mcp_mod.mcp_list_tools("srv")
    assert "srv [stdio] connected" in out
    assert "down" not in out


def test_mcp_list_tools_unknown_server(monkeypatch):
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: FakeManager())
    result = mcp_mod.mcp_list_tools("nope")
    assert result.startswith("Error: unknown MCP server 'nope'")
    assert "srv" in result  # lists the configured names


# --------------------------------------------------- mcp_tool_bindings (G1)


class BindingManager:
    """Manager double for the injection tests (status + list_tools)."""

    def __init__(self, status, tools_by_server=None, results=None):
        self.status_value = status
        self.tools_by_server = tools_by_server or {}
        self.results = results or {}
        self.calls: list[tuple[str, str, dict]] = []

    def connect_all(self):
        return {name: [] for name, entry in self.status_value.items()
                if entry.get("status") == "connected"}

    def refresh(self):
        return self.connect_all()

    def status(self):
        return self.status_value

    def list_tools(self, server):
        return list(self.tools_by_server.get(server, []))

    def call(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return self.results.get((server, tool), f"ran {tool}")


def test_bindings_empty_without_manager(monkeypatch):
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: None)
    assert mcp_mod.mcp_tool_bindings() == ([], {})


def test_bindings_build_definitions_from_input_schema(monkeypatch):
    schema = {"type": "object", "properties": {"path": {"type": "string"}},
              "required": ["path"]}
    manager = BindingManager(
        {"srv": {"transport": "http", "status": "connected", "tools": ["read"]}},
        tools_by_server={"srv": [{"name": "read", "description": "Read a file",
                                  "inputSchema": schema}]})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    definitions, impls = mcp_mod.mcp_tool_bindings()
    assert len(definitions) == 1
    function = definitions[0]["function"]
    assert definitions[0]["type"] == "function"
    assert function["name"] == "mcp_srv_read"
    assert function["description"] == "Read a file"
    assert function["parameters"] == schema  # inputSchema passes through verbatim
    assert set(impls) == {"mcp_srv_read"}


def test_bindings_prefix_and_sanitize_names(monkeypatch):
    manager = BindingManager(
        {"web.data": {"transport": "http", "status": "connected", "tools": []}},
        tools_by_server={"web.data": [
            {"name": "read file!", "description": "", "inputSchema": {}}]})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    _, impls = mcp_mod.mcp_tool_bindings()
    assert list(impls) == ["mcp_web_data_read_file_"]  # '!' -> '_', ' ' -> '_'


def test_bindings_truncate_long_names_to_64_chars(monkeypatch):
    manager = BindingManager(
        {"srv": {"transport": "http", "status": "connected", "tools": []}},
        tools_by_server={"srv": [
            {"name": "t" * 100, "description": "", "inputSchema": {}}]})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    _, impls = mcp_mod.mcp_tool_bindings()
    name = next(iter(impls))
    assert len(name) == 64 and name.startswith("mcp_srv_tttt")


def test_bindings_resolve_name_collisions(monkeypatch):
    """Same tool name on two servers, plus a collision with a built-in."""
    tool = {"name": "search", "description": "", "inputSchema": {}}
    manager = BindingManager(
        {"a": {"transport": "http", "status": "connected", "tools": ["search"]},
         "b": {"transport": "http", "status": "connected", "tools": ["search"]}},
        tools_by_server={"a": [tool], "b": [dict(tool)]})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    _, impls = mcp_mod.mcp_tool_bindings(existing_names={"web_search", "mcp_a_search"})
    # 'mcp_a_search' already taken by a built-in -> suffixed; then 'mcp_b_search'
    assert set(impls) == {"mcp_a_search_2", "mcp_b_search"}


def test_bindings_skip_tools_with_invalid_schema(monkeypatch):
    manager = BindingManager(
        {"srv": {"transport": "http", "status": "connected", "tools": []}},
        tools_by_server={"srv": [
            {"name": "no_schema", "description": "", "inputSchema": None},
            {"name": "ok", "description": "", "inputSchema": {"type": "object"}},
        ]})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    definitions, impls = mcp_mod.mcp_tool_bindings()
    assert list(impls) == ["mcp_srv_ok"]
    assert [d["function"]["name"] for d in definitions] == ["mcp_srv_ok"]


def test_bindings_skip_failed_servers(monkeypatch):
    manager = BindingManager(
        {"down": {"transport": "http", "status": "error: unreachable",
                  "tools": []}})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    assert mcp_mod.mcp_tool_bindings() == ([], {})


def test_bindings_wrapper_calls_manager(monkeypatch):
    manager = BindingManager(
        {"srv": {"transport": "http", "status": "connected", "tools": ["echo"]}},
        tools_by_server={"srv": [{"name": "echo", "description": "Echo it",
                                  "inputSchema": {"type": "object"}}]},
        results={("srv", "echo"): "echoed!"})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    _, impls = mcp_mod.mcp_tool_bindings()
    wrapper = impls["mcp_srv_echo"]
    assert wrapper.__name__ == "mcp_srv_echo"
    assert "Echo it" in wrapper.__doc__
    assert wrapper(text="hi") == "echoed!"
    assert manager.calls == [("srv", "echo", {"text": "hi"})]


def test_bindings_wrapper_maps_mcperror_to_error_string(monkeypatch):
    class Failing(BindingManager):
        def call(self, server, tool, arguments):
            raise McpError("server 'srv' is not connected")

    manager = Failing(
        {"srv": {"transport": "http", "status": "connected", "tools": ["echo"]}},
        tools_by_server={"srv": [{"name": "echo", "description": "",
                                  "inputSchema": {}}]})
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: manager)
    _, impls = mcp_mod.mcp_tool_bindings()
    assert impls["mcp_srv_echo"]() == "Error: server 'srv' is not connected"
