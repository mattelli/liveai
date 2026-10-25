"""System tools: platform info, date/time, working directory, shell commands."""

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.common import dump_func_call

__all__ = [
    "current_date_time", "exit_gracefully", "get_working_directory",
    "run_command", "spawn_process", "system_info",
]


def exit_gracefully() -> None:
    """
    Exits the program gracefully.
    Useful when user asks to quit or exit the application.
    """
    print("\nExiting gracefully. Bye! 👋")
    raise SystemExit(0)


@dump_func_call
def system_info() -> str:
    """
    Returns system information such as OS, CPU.
    Useful when user asks for system status or performance.
    """
    try:
        os_info = platform.platform()
        processor_info = platform.processor()
        platform_info = platform.uname()
        return f"{platform_info.system} {platform_info.release} ({os_info}), Processor: {processor_info}"
    except Exception as e:
        return f"Error fetching system info: {e}"


@dump_func_call
def current_date_time() -> str:
    """
    Returns the current date and time.
    Useful when user asks for time, date, or 'what day is it?'
    """
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%A, %B %d, %Y at %I:%M %p, %Z")


@dump_func_call
def get_working_directory() -> str:
    """
    Returns the current working directory.
    Useful when user asks for getting current directory files and directories,
    or when user asks for launching commands.
    """
    return str(Path.cwd().resolve())


@dump_func_call
def run_command(cmd: str, timeout_s: float = 30.0, cwd: str | None = None) -> str:
    """
    Run a command with a subprocess and wait for it to terminate.
    It returns either the command output or error as string.
    Useful when user asks to work with files, directories,
    or to execute commands for example:
        git, wget, curl, cd, mkdir, mv, cp, find, grep, python, ping
    and so on so fourth.
    """
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(cwd) if cwd is not None else Path.cwd(),
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,
        )
    except Exception as exc:
        return f"Error starting command: {exc}"
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return f"Command timed out after {timeout_s}s: {stderr}"
    if process.returncode != 0:
        return f"Return code [{process.returncode}]: {stderr}"
    return stdout


@dump_func_call
def spawn_process(cmd: str, cwd: str | None = None) -> str:
    """
    Spawn a process without waiting for it: the command keeps running in the
    background after this call returns. Returns the spawned Process ID (PID)
    so the caller can inspect or stop it later (e.g. with run_command), or
    an error message if the process could not be started.
    Useful when user asks to launch a long-running program
    (a server, a watcher, a player) without blocking the conversation.
    """
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=Path(cwd) if cwd is not None else Path.cwd(),
            shell=True,
        )
    except Exception as exc:
        return f"Error spawning process: {exc}"
    return f"Process started with PID {process.pid}"