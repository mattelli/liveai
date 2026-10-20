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
