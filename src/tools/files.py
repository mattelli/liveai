"""File tools: read/write text files, search and replace their content."""

from pathlib import Path

from src.common import dump_func_call, normalize_extensions

__all__ = [
    "read_text_file", "replace_string_in_file",
    "search_in_file", "search_pattern_in_files",
    "write_text_file",
]


@dump_func_call
def read_text_file(file: str, start_line: int = 0, line_count: int = 0,
                   number_lines: bool = False) -> str:
    """
    Read a text file and return its contents as a string.
    Useful when user asks for content description.
    Optional start_line (1-based) and line_count read only a window of the
    file, e.g. after search_in_file / search_pattern_in_files report line
    numbers. number_lines prefixes each line with its number ("12: text"),
    useful to cite exact positions for replace_string_in_file.
    """
    file_path = Path(file).resolve()
    if not (file_path.exists() and file_path.is_file()):
        return f"Unable to read '{file}'"
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    if start_line > 0 or line_count > 0 or number_lines:
        lines = text.splitlines()
        start = max(start_line - 1, 0) if start_line > 0 else 0
        window = lines[start:start + line_count] if line_count > 0 else lines[start:]
        if number_lines:
            width = len(str(start + len(window)))
            window = [f"{start + i + 1:>{width}}: {line}"
                      for i, line in enumerate(window)]
        if line_count > 0 and start + line_count < len(lines):
            window.append(
                f"… [windowed: lines {start + 1}-{start + line_count} of "
                f"{len(lines)} total — raise line_count or re-read for more]")
        return "\n".join(window)
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
                with open(dest_path, "a", encoding="utf-8") as f:
                    f.write(content)
            except Exception as exc:
                return f"Error: {exc!r}"
            return f"Appended to '{dest_path.resolve()}'"
        if not overwrite:
            return "File already exists."
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    return f"File created at '{dest_path.resolve()}'"


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
    ) -> str:
    """
    Search for a pattern in files within a directory recursively.
    Returns a list of the file paths and the line numbers where pattern is found.
    ``include_extensions``/``exclude_extensions`` accept ``py`` or ``.py``.
    ``exclude_glob`` (default ``".*"``) skips hidden entries.
    """
    include_extensions_list = normalize_extensions(include_extensions)
    exclude_extensions_list = normalize_extensions(exclude_extensions)
    results: dict[str, list[int]] = {}
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
        ext = file_path.suffix.lower()
        if include_extensions_list and ext not in include_extensions_list:
            continue
        if exclude_extensions_list and ext in exclude_extensions_list:
            continue
        if any(item in file_path.parts for item in excluded_items):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines):
                    if pattern in line:
                        if file_path not in results:
                            results[file_path] = []
                        results[file_path].append(line_num + 1)  # Line numbers are 1-based
        except Exception:
            continue
    if not results:
        return f"Did not find file having pattern '{pattern}' in directory '{search_dir}' "
    formatted_res = f"Found '{len(results)}' files having pattern '{pattern}' in directory '{search_dir}' :\n"
    for path, lines in results.items():
        formatted_res += f"- in file '{str(path)}' at lines: " + ", ".join(map(str, lines)) + "\n"
    return formatted_res


@dump_func_call
def replace_string_in_file(path: str, old_string: str, new_string: str) -> str:
    """
    Replace all literal occurrences of old_string in a file with new_string and count replacements.
    """
    _file_path = Path(path).resolve()
    if _file_path.is_dir():
        return f"Error: '{_file_path}' is a directory."
    try:
        with open(_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        count = content.count(old_string)
        content = content.replace(old_string, new_string)
        with open(_file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"'{count}' occurrences of '{old_string}' have been replaced by '{new_string}'"
    except FileNotFoundError:
        return f"Error: File '{_file_path}' not found."
    except Exception as e:
        return f"Error: {e}"