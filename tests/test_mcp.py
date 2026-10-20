"""Unit tests for the MCP tool module (offline: no real MCP server, no SDK)."""

from types import SimpleNamespace

import src.tools.mcp as mcp_mod
from src.tools.mcp import configure, configured_servers


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
