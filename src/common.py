import threading
import time
import tomllib
import json
import inspect
import re
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Literal

from rich.console import Console
from rich.syntax import Syntax

Colors = Literal['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'reset']

# Activity spinner: braille dots, redrawn every _SPIN_INTERVAL seconds
# while a generation produces no visible output.
_SPIN_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # braille dots
_SPIN_INTERVAL = 0.12
_SPIN_CLEAR = chr(13) + chr(27) + '[K'  # CR + erase to end of line

# Optional pre-output callback (used to clear the activity spinner before raw
# prints that bypass StreamPrinter, e.g. tool-call banners or round separators).
_output_hook: Callable[[], None] | None = None


def set_output_hook(hook: Callable[[], None] | None) -> None:
    """Register/unregister a callback invoked before every ``cprint``."""
    global _output_hook
    _output_hook = hook

def cprint(text: str, color: Colors = "reset", bold: bool = False, end: str = "\n", flush: bool = False) -> None:
    """
    Print text in the terminal with ANSI colors.
    Examples:
        cprint("Hello", "green")
        cprint("Warning", "yellow", bold=True)
        cprint("Error", "red", end="", flush=True)
    """
    if _output_hook is not None:
        _output_hook()
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


def prompt_input(label: str, color: Colors = "yellow") -> str | None:
    """Write label (flushed, no newline) and read a line.

    Returns the raw line (without trailing newline), or None on EOFError.
    """
    cprint(label, color, end="", flush=True)
    try:
        return input()
    except EOFError:
        return None


def crule(title: str = "", style: str = "dim") -> None:
    """Print a terminal-width horizontal rule carrying an optional title.

    The title is rendered as plain text (no rich markup parsing, so values
    like "[28k]" are safe). Honors the output hook (spinner suppression)
    like cprint, and falls back to a dashed line if rich is unavailable.
    """
    if _output_hook is not None:
        _output_hook()
    try:
        from rich.console import Console
        from rich.rule import Rule
        from rich.text import Text

        Console().print(Rule(Text(title) if title else "", style=style))
    except ImportError:  # pragma: no cover - rich is a hard dependency
        cprint("-" * 50)


class StreamPrinter:
    """
    Prints streamed model output with syntax-highlighted fenced code blocks.

    Plain text is printed in `text_color`; text between ``` fences is buffered
    and rendered with Pygments (via Rich) once the closing fence arrives, using
    the language tag from the opening fence (defaults to "text").

    Plain text is emitted as soon as it arrives, so single-line replies stream
    token by token instead of appearing all at once at ``flush()``. Only the
    ambiguous prefix at the start of a line (indentation plus up to two
    backticks) is held back until it can be classified as text or as a fence.
    """

    def __init__(self, text_color: Colors = "cyan", code_theme: str = "monokai") -> None:
        self.console = Console()
        self.text_color = text_color
        self.code_theme = code_theme
        self._line = ""              # current line tail not yet classified/emitted
        self._in_code = False
        self._code_lang = "text"
        self._code_lines: list[str] = []
        self._plain_started = False  # current line already partly emitted as text

        # Activity spinner: animates while real output is absent (prompt
        # processing, tool execution) or buffered (code blocks are held back
        # until their closing fence). Any real emission erases it.
        self._spin_active = False
        self._spin_suppressed = False  # paused around external (raw) prints
        self._spin_drawn = False
        self._spin_note = ""
        self._spin_t0 = 0.0
        self._dirty_line = False     # cursor is inside streamed text
        self._spin_lock = threading.Lock()
        self._spin_stop = threading.Event()
        self._spin_thread = None

    def feed(self, content: str) -> None:
        """Push a streamed fragment, emitting plain text as soon as it is safe."""
        start = 0
        while True:
            newline = content.find("\n", start)
            if newline == -1:
                self._feed_partial(content[start:])
                return
            self._feed_partial(content[start:newline])
            self._end_line()
            start = newline + 1

    def _feed_partial(self, text: str) -> None:
        if not text:
            return
        if self._in_code:
            self._line += text
        elif self._plain_started:
            self._emit(text)
        else:
            self._line += text
            if not self._could_be_fence(self._line):
                self._emit(self._line)
                self._line = ""
                self._plain_started = True

    @staticmethod
    def _could_be_fence(partial: str) -> bool:
        """True while `partial` could still become a ``` fence opening."""
        stripped = partial.lstrip()
        if stripped == "":
            return True  # only indentation so far
        backticks = len(stripped) - len(stripped.lstrip("`"))
        if backticks >= 3:
            return True  # confirmed fence; wait for the newline to classify it
        return backticks == len(stripped)  # only 1-2 backticks so far; may grow

    def _end_line(self) -> None:
        line, self._line = self._line, ""
        started, self._plain_started = self._plain_started, False
        stripped = line.strip()
        if self._in_code:
            if stripped.startswith("```"):
                self._in_code = False
                self._flush_code()
            else:
                self._code_lines.append(line + "\n")
            return
        if stripped.startswith("```"):
            self._in_code = True
            self._code_lang = stripped[3:].strip() or "text"
            self._code_lines = []
            return
        self._emit("\n" if started else line + "\n")

    def feed_thinking(self, text: str) -> None:
        """Stream chain-of-thought text in dim style ([brain] show_thinking).

        Printed directly, without touching the fenced-code/plain-text state
        of the main answer stream: reasoning always precedes the answer, and
        the line state is reset between rounds (mark_line_clean). Marks the
        line dirty so the spinner stays suppressed while thinking streams.
        """
        if not text:
            return
        with self._spin_lock:
            self._erase_spinner_locked()
            self.console.print(
                text, end="", style="dim", highlight=False, soft_wrap=True
            )
            self._dirty_line = not text.endswith(chr(10))

    def _emit(self, text: str) -> None:
        with self._spin_lock:
            self._erase_spinner_locked()
            self.console.print(
                text, end="", style=self.text_color, highlight=False, soft_wrap=True
            )
            self._dirty_line = not text.endswith(chr(10))

    def _flush_code(self) -> None:
        code = "".join(self._code_lines).rstrip("\n")
        self._code_lines = []
        if not code:
            return
        with self._spin_lock:
            self._erase_spinner_locked()
            self.console.print(
                Syntax(code, self._code_lang, theme=self.code_theme, line_numbers=False)
            )
            self._dirty_line = False

    def flush(self) -> None:
        """Flush any buffered partial line / unterminated code block."""
        line, self._line = self._line, ""
        self._plain_started = False
        if self._in_code:
            if line and not line.strip().startswith("```"):
                self._code_lines.append(line)
            self._in_code = False
            self._flush_code()
            return
        if line and not line.strip().startswith("```"):
            self._emit(line)

    # -------------------------------------------------- activity spinner

    def start_spinner(self, note: str = "working") -> None:
        """Animate the spinner until the next :meth:`stop_spinner`.

        The spinner only draws while the terminal line is clean; streamed
        text, code flushing and raw cprint output all erase or suppress
        it, so it never overwrites real content.
        """
        with self._spin_lock:
            self._spin_active = True
            self._spin_suppressed = False
            self._spin_drawn = False
            self._spin_note = note
            self._spin_t0 = time.monotonic()
            self._dirty_line = False
            self._spin_stop.clear()
        if self._spin_thread is None or not self._spin_thread.is_alive():
            self._spin_thread = threading.Thread(
                target=self._spin_loop, name="stream-spinner", daemon=True
            )
            self._spin_thread.start()
        set_output_hook(self._suppress_spinner_for_output)

    def stop_spinner(self) -> None:
        """Erase the spinner (if drawn) and stop the animation thread."""
        with self._spin_lock:
            self._spin_active = False
            self._spin_suppressed = False
            self._erase_spinner_locked()
        set_output_hook(None)
        self._spin_stop.set()
        thread = self._spin_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._spin_thread = None

    def note(self, text: str) -> None:
        """Update the spinner caption (e.g. prompt-processing progress)."""
        with self._spin_lock:
            self._spin_note = text
            self._spin_suppressed = False

    def mark_line_clean(self) -> None:
        """Declare the terminal line empty so the spinner may draw again."""
        with self._spin_lock:
            self._dirty_line = False

    def _spin_loop(self) -> None:
        while not self._spin_stop.wait(_SPIN_INTERVAL):
            with self._spin_lock:
                if (
                    self._spin_active
                    and not self._spin_suppressed
                    and not self._dirty_line
                ):
                    self._draw_spinner_locked()

    def _draw_spinner_locked(self) -> None:
        elapsed = time.monotonic() - self._spin_t0
        frame = _SPIN_FRAMES[int(elapsed / _SPIN_INTERVAL) % len(_SPIN_FRAMES)]
        note = self._spin_note
        if self._in_code:
            buffered = sum(len(line) for line in self._code_lines) + len(self._line)
            note = f"writing {self._code_lang} code ({buffered} chars)"
        self.console.file.write(_SPIN_CLEAR + frame + " " + note)
        self.console.file.flush()
        self._spin_drawn = True

    def _erase_spinner_locked(self) -> None:
        if self._spin_drawn:
            self.console.file.write(_SPIN_CLEAR)
            self.console.file.flush()
            self._spin_drawn = False

    def _suppress_spinner_for_output(self) -> None:
        """Output hook: erase the spinner before unrelated raw prints."""
        with self._spin_lock:
            self._erase_spinner_locked()
            self._spin_suppressed = True


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
        """Recursively convert a dict to a SimpleNamespace, resolving paths.

        Only scalar string values are path-resolved; strings inside TOML
        lists are kept verbatim (they are labels/prompts like agent_models
        entry names, not paths).
        """
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = namespace_from_dict(v, base_dir)
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


def normalize_extensions(value: str) -> list[str]:
    """Parse a comma separated extension list into lowercase dotted suffixes.

    ``"py, .MD ,txt"`` -> ``[".py", ".md", ".txt"]``.
    """
    result: list[str] = []
    for item in (value or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        result.append(item if item.startswith(".") else "." + item)
    return result


def safe_filename(symbol: str) -> str:
    """Turn ``'^GSPC'`` / ``'BRK/B'`` into a filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("._")
    return cleaned or "symbol"


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
