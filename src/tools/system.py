"""System tools: platform info, date/time, working directory, shell commands."""

import os
import platform
import shutil
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
def spawn_process(cmd: str, cwd: str | None = None, background: bool = True,
                  timeout_s: float = 30.0) -> str:
    """
    Spawn a process.

    If ``background`` is True (default): the command runs detached from this
    application and the call returns immediately with the spawned Process
    ID (PID):

    - Windows: the command runs in a separate PowerShell window (a new
      console) that stays open after the command finishes.
    - Linux with a desktop environment: the command runs in a new terminal
      emulator window (gnome-terminal, konsole, xterm, ...) that stays open
      after the command finishes.
    - Linux without a desktop environment: the command runs detached with
      its output logged to a ``spawn_<timestamp>.log`` file in the working
      directory.

    If ``background`` is False: the command runs in the foreground and this
    call waits for it to terminate, returning its output or error as string
    (PowerShell on Windows, bash on Linux).

    ``cwd`` sets the working directory; ``timeout_s`` is only used when
    ``background`` is False.
    """
    workdir = Path(cwd) if cwd is not None else Path.cwd()

    if platform.system() == "Windows":
        return _spawn_windows(cmd, workdir, background, timeout_s)
    return _spawn_posix(cmd, workdir, background, timeout_s)


def _has_desktop_environment() -> bool:
    """Return True when a graphical desktop session is available (Linux)."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _spawn_posix(cmd: str, workdir: Path, background: bool, timeout_s: float) -> str:
    """Spawn a process on Linux/macOS using the bash shell."""
    if not background:
        # Foreground: wait for the command to finish and capture its output.
        try:
            process = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=workdir,
                text=True,
                encoding="utf-8",
                errors="replace",
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

    # Background: prefer a terminal window when a desktop environment exists.
    if _has_desktop_environment():
        terminal = _find_terminal_emulator()
        if terminal is not None:
            return _spawn_posix_terminal(terminal, cmd, workdir)

    # No desktop environment (or no terminal emulator found): run detached
    # with output redirected to a log file next to the working directory.
    return _spawn_posix_detached(cmd, workdir)


def _find_terminal_emulator() -> list[str] | None:
    """Find an available terminal emulator command, or None."""
    # Each entry: (command, how to append an extra command to run).
    candidates: list[tuple[str, list[str]]] = [
        ("gnome-terminal", ["--"]),
        ("konsole", ["-e"]),
        ("xfce4-terminal", ["-x"]),
        ("mate-terminal", ["-e"]),
        ("lxterminal", ["-e"]),
        ("xterm", ["-e"]),
        ("x-terminal-emulator", ["-e"]),
    ]
    for name, args in candidates:
        if shutil.which(name):
            return [name, *args]
    return None


def _spawn_posix_terminal(terminal: list[str], cmd: str, workdir: Path) -> str:
    """Open the command in a new terminal emulator window."""
    # Wrap the command so the terminal stays open after it finishes.
    wrapped = f"{cmd}; echo; echo 'Process finished. Press Enter to close...'; read"
    try:
        process = subprocess.Popen(
            [*terminal, "bash", "-c", wrapped],
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        return f"Error spawning process: {exc}"
    return (f"Process started in a new terminal window ({terminal[0]}) "
            f"with PID {process.pid}")


def _spawn_posix_detached(cmd: str, workdir: Path) -> str:
    """Run the command detached, logging its output to a file in ``workdir``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = workdir / f"spawn_{timestamp}.log"
    try:
        log_handle = open(log_file, "w", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=workdir,
            start_new_session=True,
        )
    except Exception as exc:
        return f"Error spawning process: {exc}"
    finally:
        # Popen keeps its own duplicated handle; safe to close ours.
        if "log_handle" in locals():
            log_handle.close()
    return (f"Process started detached with PID {process.pid}, "
            f"logging to {log_file}")


def _spawn_windows(cmd: str, workdir: Path, background: bool, timeout_s: float) -> str:
    """Spawn a process on Windows using PowerShell."""
    if background:
        try:
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoExit",
                    "-ExecutionPolicy", "Bypass",
                    "-Command", cmd,
                ],
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        except Exception as exc:
            return f"Error spawning process: {exc}"
        return f"Process started in a new PowerShell window with PID {process.pid}"

    # Foreground: wait for the command to finish and capture its output,
    # running through PowerShell for consistency with background mode.
    try:
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command", cmd,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=workdir,
            text=True,
            encoding="utf-8",
            errors="replace",
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