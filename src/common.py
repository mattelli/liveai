import tomllib
import json
import inspect
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from rich.console import Console
from rich.syntax import Syntax

Colors = Literal['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'reset']

def cprint(text: str, color: Colors = "reset", bold: bool = False, end: str = "\n", flush: bool = False) -> None:
    """
    Print text in the terminal with ANSI colors.
    Examples:
        cprint("Hello", "green")
        cprint("Warning", "yellow", bold=True)
        cprint("Error", "red", end="", flush=True)
    """
    color_map = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "grey": "90",
        "white": "37",
        "reset": "0",
    }
    color_code = color_map.get(color.lower(), color_map["reset"])
    style = "1;" if bold else ""
    print(f"\033[{style}{color_code}m{text}\033[0m", end=end, flush=flush)


class StreamPrinter:
    """
    Prints streamed model output with syntax-highlighted fenced code blocks.

    Plain text is printed in `text_color`. Text between ``` fences is buffered
    and rendered with Pygments (via Rich) once the closing fence arrives, using
    the language tag from the opening fence (defaults to "text").
    """

    def __init__(self, text_color: Colors = "cyan", code_theme: str = "monokai") -> None:
        self.console = Console()
        self.text_color = text_color
        self.code_theme = code_theme
        self._buffer = ""       # unprocessed (incomplete) text
        self._in_code = False
        self._code_lang = "text"
        self._code_lines: list[str] = []

    def feed(self, content: str) -> None:
        """Push a new streamed fragment. Prints any newly completed lines."""
        self._buffer += content
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line + "\n")

    def _process_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("```"):
            if self._in_code:
                self._in_code = False
                self._flush_code()
            else:
                self._in_code = True
                self._code_lang = stripped[3:].strip() or "text"
                self._code_lines = []
            return

        if self._in_code:
            self._code_lines.append(line)
        else:
            self.console.print(line, end="", style=self.text_color, highlight=False, soft_wrap=True)

    def _flush_code(self) -> None:
        code = "".join(self._code_lines).rstrip("\n")
        self._code_lines = []
        if not code:
            return
        self.console.print(
            Syntax(code, self._code_lang, theme=self.code_theme, line_numbers=False)
        )

    def flush(self) -> None:
        """Flush any buffered partial line / unterminated code block."""
        if self._buffer:
            self._process_line(self._buffer)
            self._buffer = ""
        if self._in_code:
            self._in_code = False
            self._flush_code()


def load_config(config_path: Path) -> SimpleNamespace:
    """
    Load config.toml and return an object with attributes from the config file.
    Paths are resolved relative to the config file's parent directory.
    """
    config_dir = config_path.parent.resolve()
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    def resolve_path(value, base_dir):
        """Resolve a path string relative to the config directory."""
        if isinstance(value, str):
            # Resolve relative paths against config directory
            return (base_dir / value).resolve()
        return value

    def namespace_from_dict(d: dict, base_dir: Path) -> SimpleNamespace:
        """Recursively convert a dict to a SimpleNamespace, resolving paths."""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = namespace_from_dict(v, base_dir)
            elif isinstance(v, list):
                result[k] = [resolve_path(item, base_dir) if isinstance(item, str) else item for item in v]
            elif k.endswith("_file") or k.endswith("_dir")  or k.endswith("_path"):
                result[k] = resolve_path(v, base_dir)
            else:
                result[k] = v
        return SimpleNamespace(**result)

    return namespace_from_dict(data, config_dir)


def read_json(file_path: Path):
    """
    Read JSON data from a file and return it as a Python object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: Path, data):
    """
    Write JSON data to a file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dump_func_call(func):
    """
    Decorator to print function call details with parameter names and shorted values.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        sel_args = [
            (name, str(value)[:50]) for name, value in bound_args.arguments.items()
            if "self" not in str(name)
        ]
        cprint(f"[CALL] {func.__name__}(" + ", ".join(f"{n}={v!r}" for n, v in sel_args) + ")", "grey", bold=True)
        return func(*args, **kwargs)
    return wrapper
