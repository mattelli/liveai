"""Tests for the sandboxed python_eval tool (smolagents PythonInterpreterTool port)."""
import sys
import time

import pytest

import src.tools.pyeval as pyeval_mod
import src.tools as tools_mod


def test_python_eval_returns_stdout():
    result = pyeval_mod.python_eval("print(21 * 2)")
    assert "42" in result
    assert "[exit 0]" in result


def test_python_eval_captures_stderr_and_status():
    result = pyeval_mod.python_eval(
        "import sys; sys.stderr.write('oops'); raise SystemExit(3)")
    assert "oops" in result
    assert "[exit 3]" in result


def test_python_eval_timeout_kills_process():
    started = time.perf_counter()
    result = pyeval_mod.python_eval(
        "while True: pass", timeout_s=1.0)
    elapsed = time.perf_counter() - started
    assert "timed out" in result
    assert elapsed < 15  # hard kill, not the default 60s cap


def test_python_eval_rejects_empty_code():
    assert pyeval_mod.python_eval("").startswith("Error:")
    assert pyeval_mod.python_eval("   ").startswith("Error:")


def test_python_eval_output_cap():
    result = pyeval_mod.python_eval("print('x' * 100_000)")
    assert "truncated" in result
    assert len(result) < 5_000


def test_python_eval_exception_reports_status():
    result = pyeval_mod.python_eval("raise ValueError('boom')")
    assert "boom" in result
    assert "[exit 1]" in result


def test_python_eval_isolated_agents_only():
    """python_eval is absent from the main agent, present for isolated ones."""
    import src.live_ai as live_ai_mod
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    agent.isolated_tools = False
    main_names = [t.__name__ for t in agent._build_tools()]
    assert "python_eval" not in main_names

    isolated = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    isolated.isolated_tools = True
    isolated._messages = []
    isolated_names = [t.__name__ for t in isolated._build_tools()]
    assert "python_eval" in isolated_names


def test_pyeval_exported_from_tools_package():
    assert hasattr(tools_mod, "python_eval")


def test_python_eval_runs_in_configured_tmp_dir(tmp_path, monkeypatch):
    # With [cwd] tmp_dir_cwd configured, the sandbox scratch dir lives under it
    # (and is cleaned up afterwards), never in the process cwd.
    import src.common as common
    tmp_dir = tmp_path / "scratch"
    tmp_dir.mkdir()
    monkeypatch.setattr(common, "_tmp_dir", tmp_dir)
    result = pyeval_mod.python_eval("import os; print(os.getcwd())")
    assert "[exit 0]" in result
    assert str(tmp_dir) in result
    # the unique scratch subfolder was removed on exit
    assert list(tmp_dir.iterdir()) == []


def test_python_eval_falls_back_to_system_temp(monkeypatch):
    import src.common as common
    monkeypatch.setattr(common, "_tmp_dir", None)
    result = pyeval_mod.python_eval("print('ok')")
    assert "ok" in result and "[exit 0]" in result


def test_python_eval_killed_by_cancel_reports_cancelled(monkeypatch):
    """A python_eval subprocess killed by terminate_all() reports cancelled."""
    import subprocess

    import src.common as common

    class FakeProcess:
        args = ["python", "-I", "-c", "code"]
        returncode = None

        def __init__(self):
            self.killed = 0

        def communicate(self, timeout=None):
            common.terminate_all()  # cancel lands while the snippet runs
            return "", ""

        def kill(self):
            self.killed += 1

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    result = pyeval_mod.python_eval("while True: pass", timeout_s=5)
    assert "cancelled" in result.lower()
    assert process.killed == 1
    assert common.terminate_all() == 0  # nothing left registered


def test_python_eval_registers_and_unregisters(monkeypatch):
    """A normal snippet leaves nothing registered (no leak)."""
    import src.common as common

    result = pyeval_mod.python_eval("print('ok')")
    assert "[exit 0]" in result
    assert common.terminate_all() == 0
