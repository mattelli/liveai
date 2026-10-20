"""Integration smoke test for the MCP connector (optional, needs the SDK).

Starts a tiny in-repo MCP echo server over stdio
(``python tests/mcp_echo_server.py``, using the ``mcp`` SDK's server API) and
proves the real ``McpSdkConnector`` end-to-end: ``connect_all`` lists the
echo tool and ``call`` echoes a payload. Skipped silently when the ``mcp``
package is missing (offline/CI runs without the SDK).
"""

import sys
import pathlib

import pytest

import src.tools.mcp as mcp_mod
from src.tools.mcp import Manager

pytest.importorskip("mcp", reason="requires the mcp package (pip install mcp)")

ECHO_SERVER = pathlib.Path(__file__).parent / "mcp_echo_server.py"


@pytest.fixture
def echo_manager():
    """Manager wired to the in-repo echo server (stdio, real SDK)."""
    manager = Manager(
        {"echo": {"command": sys.executable, "args": [str(ECHO_SERVER)]}},
        connect_timeout_s=30.0,
        call_timeout_s=30.0,
    )
    yield manager
    manager.shutdown()


def test_connect_all_lists_echo_tool(echo_manager):
    result = echo_manager.connect_all()
    assert "echo" in result, f"connect failed: {echo_manager._errors}"
    assert "echo" in result["echo"]


def test_call_echoes_payload(echo_manager):
    echo_manager.connect_all()
    assert echo_manager.call("echo", "echo", {"text": "ping"}) == "echo: ping"


def test_refresh_still_works(echo_manager):
    echo_manager.connect_all()
    result = echo_manager.refresh()
    assert "echo" in result.get("echo", [])
    assert echo_manager.call("echo", "echo", {"text": "again"}) == "echo: again"


def test_unknown_tool_is_an_error_string_via_bridge(echo_manager, monkeypatch):
    """The bridge tool maps McpError to an 'Error: ...' string (G4)."""
    monkeypatch.setattr(mcp_mod, "get_manager", lambda: echo_manager)
    echo_manager.connect_all()
    result = mcp_mod.mcp_call_tool("echo", "missing", {})
    assert result.startswith("Error:")
    assert "no tool 'missing'" in result
