"""Sandboxed Python evaluation tool (port of smolagents ``PythonInterpreterTool``).

Runs a generated snippet in a *subprocess* with a restricted environment:
a scratch working directory (the run's ``[cwd] tmp_dir_cwd``, falling back to
a fresh temporary directory), no inherited shell, hard timeout, capped output
and the exit status in the result. It is registered **only** for isolated
(background) agents — the interactive main agent already has ``run_command``
and is user-supervised.

Note: like smolagents' local executor this is *not* a security boundary
against hostile code; it contains accidental damage (infinite loops, output
flooding, cwd pollution) for a semi-trusted model.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from logging import getLogger

from src.common import configured_tmp_dir

__all__ = ["python_eval"]

_MAX_OUTPUT_CHARS = 4000
_DEFAULT_TIMEOUT_S = 10.0
_MAX_TIMEOUT_S = 60.0


def python_eval(code: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> str:
    """
    Evaluate a short Python snippet in a sandboxed subprocess and return its output.

    The snippet runs with a fresh temporary working directory and a hard
    timeout; stdout/stderr are captured (capped) and the exit status is
    reported. Use for quick calculations or data transformations instead of
    touching the user's system with run_command.

    Args:
        code: The Python source code to run (print() to produce output).
        timeout_s: Hard timeout in seconds (capped at 60).

    Returns:
        stdout + stderr (capped at 4000 chars) followed by an ``[exit N]``
        status line. Timeouts report ``[timeout]``; missing interpreter or
        OS errors return an ``Error:`` line instead of raising.
    """
    code = str(code or "")
    if not code.strip():
        return "Error: no code given"
    try:
        timeout_s = min(max(float(timeout_s), 0.1), _MAX_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout_s = _DEFAULT_TIMEOUT_S
    try:
        # Run in a unique scratch subfolder of the run's tmp dir (when
        # configured), so snippet files never pollute the process cwd and are
        # still cleaned up afterwards. Without a configured tmp dir, fall back
        # to the system temporary directory.
        parent = configured_tmp_dir()
        with tempfile.TemporaryDirectory(
                dir=str(parent) if parent is not None else None) as tmp_dir:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                # A minimal environment: no inherited shell variables.
                env={"SYSTEMROOT": __import__("os").environ.get(
                    "SYSTEMROOT", "")},
            )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout_s:g}s [timeout]"
    except Exception as exc:
        return f"Error: could not start the Python subprocess: {exc}"

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip()
    if len(output) > _MAX_OUTPUT_CHARS:
        output = (output[:_MAX_OUTPUT_CHARS]
                  + f" ... [truncated, {len(output)} chars total]")
    status = "[exit 0]" if completed.returncode == 0 \
        else f"[exit {completed.returncode}]"
    return (output + ("\n" if output else "") + status).strip()
