"""File tools: read/write text files, search and replace their content."""

import hashlib
import os
import tempfile
from pathlib import Path

from src.common import dump_func_call, normalize_extensions

__all__ = [
    "read_text_file", "replace_string_in_file",
    "search_in_file", "search_pattern_in_files",
    "write_text_file",
]


@dump_func_call
def read_text_file(file: str, start_line: int = 0, line_count: int = 0,
                   number_lines: bool = False,
                   max_file_bytes: int = 5_000_000,
                   checksum: bool = False) -> str:
    """
    Read a text file and return its contents as a string.
    Useful when user asks for content description.
    Optional start_line (1-based) and line_count read only a window of the
    file, e.g. after search_in_file / search_pattern_in_files report line
    numbers. number_lines prefixes each line with its number ("12: text"),
    useful to cite exact positions for replace_string_in_file.
    ``max_file_bytes`` (0 = unlimited) caps how many bytes are read;
    the read stops at the cap and the returned text is truncated.
    With ``checksum`` a ``[sha1:…]`` line of the returned text is appended,
    giving a reference digest to verify the content against the file on
    disk (detects transport/assembly corruption of large results).
    """
    file_path = Path(file).resolve()
    if not (file_path.exists() and file_path.is_file()):
        return f"Unable to read '{file}'"
    try:
        if start_line > 0 or line_count > 0 or number_lines:
            # Stream only the requested window instead of loading the whole
            # file into memory (important for large files).  newline=""
            # keeps the original line endings so a CRLF file is reported
            # as CRLF, consistent with the full read below.
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                raw_lines = f.readlines(max_file_bytes)
            eol = "\r\n" if (raw_lines and raw_lines[0].endswith("\r\n")) else "\n"
            lines = [line.rstrip("\r\n") for line in raw_lines]
        else:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                text = f.read()
    except Exception as exc:
        return f"Error: {exc!r}"
    if start_line > 0 or line_count > 0 or number_lines:
        start = max(start_line - 1, 0) if start_line > 0 else 0
        window = lines[start:start + line_count] if line_count > 0 else lines[start:]
        if number_lines:
            width = len(str(start + len(window)))
            window = [f"{start + i + 1:>{width}}: {line}"
                      for i, line in enumerate(window)]
        payload = eol.join(window)  # digest covers content, not UI markers
        if line_count > 0 and start + line_count < len(lines):
            window.append(
                f"… [windowed: lines {start + 1}-{start + line_count} of "
                f"{len(lines)} total — raise line_count or re-read for more]")
        if checksum:
            digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
            window.append(f"[sha1:{digest}]")
        return eol.join(window)
    if checksum:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        text += f"\n[sha1:{digest}]"
    return text


@dump_func_call
def write_text_file(file: str, content: str, overwrite: bool = False,
                    append: bool = False) -> str:
    """
    Write a text file with the given its path and string content.
    Useful when user asks for content generation, e.g. code generation.
    Existing files are only replaced when ``overwrite`` is True; with
    ``append`` the content is added to the end of an existing file instead
    (cheaper than resending the whole file for incremental updates).
    For modifying existing files prefer replace_string_in_file: it sends
    only the changed fragment, not the whole file.
    """
    dest_path = Path(file).resolve()
    if dest_path.exists():
        if dest_path.is_dir():
            return f"Error: '{dest_path}' is a directory."
        if append:
            try:
                with open(dest_path, "a", encoding="utf-8", newline="") as f:
                    f.write(content)
            except Exception as exc:
                return f"Error: {exc!r}"
            return f"Appended to '{dest_path.resolve()}'"
        if not overwrite:
            return "File already exists."
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
    except Exception as exc:
        return f"Error: {exc!r}"
    # Read-back verification: catch partial writes / disk-level corruption
    # at write time instead of silently committing a damaged file.
    try:
        with open(dest_path, "r", encoding="utf-8", newline="") as f:
            if f.read() != content:
                return (f"Error: verification failed for '{dest_path}' — "
                        "file content differs from what was written.")
    except Exception as exc:
        return f"Error: verification read failed for '{dest_path}': {exc!r}"
    return f"File created at '{dest_path.resolve()}' (verified)"


@dump_func_call
def search_in_file(pattern: str, file: str) -> str:
    """
    Search for a pattern in a specified file.
    Returns the line numbers where pattern is found.
    """
    file_path = Path(file).resolve()
    if not file_path.exists():
        return f"Error: '{file_path}' does not exist!"
    if file_path.is_dir():
        return f"Error: '{file_path}' is a directory!"
    found_lines: list[int] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                if pattern in line:
                    found_lines.append(line_num + 1)  # Line numbers are 1-based
    except Exception as e:
        return f"Error reading '{file_path}': {e}"

    if not found_lines:
        return f"Did not find pattern '{pattern}' in file '{file_path}' "
    return f"Found pattern '{pattern}' in file '{file_path}' at lines: " + ", ".join(map(str, found_lines)) + "\n"


@dump_func_call
def search_pattern_in_files(
        pattern: str,
        directory: str,
        include_extensions: str = "",
        exclude_extensions: str = "",
        exclude_glob: str | None = ".*",
        exclude_dirs: str = "target",
        max_file_bytes: int = 5_000_000,
    ) -> str:
    """
    Search for a pattern in files within a directory recursively.
    Returns a list of the file paths and the line numbers where pattern is found.
    ``include_extensions``/``exclude_extensions`` accept ``py`` or ``.py``.
    ``exclude_glob`` (default ``".*"``) skips hidden entries.
    ``exclude_dirs`` (default ``"target"``) skips whole build/dependency
    directories (comma-separated names, matched against any path part);
    pass ``""`` to disable.
    Files that are binary/undecodable or larger than ``max_file_bytes``
    (0 = unlimited) are skipped and reported instead of silently ignored.
    """
    include_extensions_list = normalize_extensions(include_extensions)
    exclude_extensions_list = normalize_extensions(exclude_extensions)
    excluded_dir_names = {d.strip().lower()
                          for d in exclude_dirs.split(",") if d.strip()}
    results: dict[str, list[int]] = {}
    skipped: list[str] = []
    search_dir = Path(directory).resolve()
    if not search_dir.exists():
        return f"'{search_dir}' does not exist!"
    if not search_dir.is_dir():
        return f"'{search_dir}' is not a directory!"
    excluded_items = []
    if exclude_glob:
        excluded_items = [f.name for f in search_dir.glob(exclude_glob)]
    for file_path in search_dir.rglob("*"):
        if file_path.is_dir():
            continue
        if any(item in file_path.parts for item in excluded_items):
            continue
        if any(part.lower() in excluded_dir_names
               for part in file_path.parts[:-1]):
            continue
        ext = file_path.suffix.lower()
        if include_extensions_list and ext not in include_extensions_list:
            continue
        if exclude_extensions_list and ext in exclude_extensions_list:
            continue
        try:
            if max_file_bytes and file_path.stat().st_size > max_file_bytes:
                skipped.append(f"{file_path} (exceeds max_file_bytes)")
                continue
            found_in_file: list[int] = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):  # streamed, not readlines()
                    if pattern in line:
                        found_in_file.append(line_num + 1)  # 1-based
            if found_in_file:
                results[file_path] = found_in_file
        except (UnicodeDecodeError, ValueError):
            skipped.append(f"{file_path} (binary/undecodable)")
            continue
        except OSError as e:
            skipped.append(f"{file_path} ({e})")
            continue
    if not results:
        result = f"Did not find file having pattern '{pattern}' in directory '{search_dir}' "
    else:
        result = (f"Found '{len(results)}' files having pattern '{pattern}' "
                  f"in directory '{search_dir}' :\n")
        for path, lines in results.items():
            result += f"- in file '{str(path)}' at lines: " + ", ".join(map(str, lines)) + "\n"
    if skipped:
        result += f"\nSkipped {len(skipped)} file(s):\n"
        for item in skipped:
            result += f"- {item}\n"
    return result


@dump_func_call
def replace_string_in_file(path: str, old_string: str, new_string: str) -> str:
    """
    Replace all literal occurrences of old_string in a file with new_string and count replacements.
    """
    _file_path = Path(path).resolve()
    if _file_path.is_dir():
        return f"Error: '{_file_path}' is a directory."
    if not old_string:
        return "Error: 'old_string' must not be empty."
    try:
        with open(_file_path, 'r', encoding='utf-8', newline='') as file:
            content = file.read()
        count = content.count(old_string)
        if count == 0:
            return (f"'{old_string}' not found in '{_file_path}': "
                    "nothing replaced.")
        content = content.replace(old_string, new_string)
        # Atomic write: never leave a truncated file behind on crash.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(_file_path.parent), prefix=_file_path.name + ".",
            suffix=".tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='') as tmp:
                tmp.write(content)
            os.replace(tmp_name, _file_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return f"'{count}' occurrences of '{old_string}' have been replaced by '{new_string}'"
    except FileNotFoundError:
        return f"Error: File '{_file_path}' not found."
    except Exception as e:
        return f"Error: {e}"