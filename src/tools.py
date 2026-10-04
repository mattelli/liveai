import re
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

try:
    from src.common import dump_func_call
except Exception:
    from common import dump_func_call

__all__ = [
    "ValidYF", "exit_gracefully", "system_info", "current_date_time",
    "read_text_file", "write_text_file", "get_working_directory", "run_command",
    "webpage_headlines", "webpage_content", "web_search", "capture_image",
    "fetch_financial_data", "save_financial_chart", "search_in_file",
    "search_pattern_in_files", "replace_string_in_file", "download_file",
    "extract_archive",
]

CAPTURED_IMAGE: str = "captured_image.jpg"
REQUIRED_FINANCIAL_COLUMNS = ("Date", "Open", "Close")

# Test seams (module-level so the public tool signatures stay LLM-friendly).
_VIDEO_CAPTURE_FACTORY = None
_IMAGE_WRITER = None


class ValidYF:
    """
    Yahoo Finance time period and time interval definitions
    """
    period = Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    interval = Literal['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']


# ----------------------------------------------------------------- helpers


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


def extract_text(soup, tags, min_len: int = 20) -> list[str]:
    """Return the stripped text of matching tags longer than ``min_len``."""
    return [
        text
        for tag in soup.find_all(tags)
        if len(text := tag.get_text(strip=True)) > min_len
    ]


def format_search_results(results: list[str], n_results: int = 5) -> str:
    """Render search results as the numbered block returned to the model."""
    lines = ["\nSearch Results:\n"]
    for index, item in enumerate(results[:n_results], 1):
        lines.append(f"{index}) {item.replace(' › ', '/')}")
        lines.append(50 * "=")
    return "\n".join(lines) + "\n"


def select_financial_columns(data, symbol: str):
    """Reduce a yfinance frame to ``['Date', 'Open', 'Close']``.

    Accepts both flat and MultiIndex column layouts. Returns a DataFrame, or an
    error string when the data is empty or a required column is missing.
    """
    import pandas as pd

    if data is None or data.empty:
        return f"No data found for {symbol}."
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(level[0]) for level in frame.columns]
    date_col = "Date" if "Date" in frame.columns else (
        "Datetime" if "Datetime" in frame.columns else None
    )
    if date_col is None:
        return "Date is missing"
    frame = frame.rename(columns={date_col: "Date"})
    missing = [c for c in REQUIRED_FINANCIAL_COLUMNS if c not in frame.columns]
    if missing:
        return f"Not all the required columns {list(REQUIRED_FINANCIAL_COLUMNS)} are available."
    frame = frame[list(REQUIRED_FINANCIAL_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    if getattr(frame["Date"].dt, "tz", None) is not None:
        frame["Date"] = frame["Date"].dt.tz_localize(None)
    return frame


def safe_filename(symbol: str) -> str:
    """Turn ``'^GSPC'`` / ``'BRK/B'`` into a filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("._")
    return cleaned or "symbol"


def _fetch_soup(url: str, timeout_s: int = 10):
    """Fetch a URL and return a BeautifulSoup document, or an error string."""
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
    except Exception as exc:
        return f"Error fetching URL: {exc}"
    return BeautifulSoup(response.text, "html.parser")


def _create_chrome_driver():
    """Create a headless-ish Chrome WebDriver. Imported lazily."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--incognito")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(width=300, height=200)
    return driver


def _collect_results(driver, query: str, timeout_s: float) -> list[str]:
    """Drive a webdriver through a DuckDuckGo search and return result texts."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver.get("https://duckduckgo.com/")
    wait = WebDriverWait(driver, timeout_s)
    search_bar = wait.until(EC.element_to_be_clickable((By.NAME, "q")))
    search_bar.send_keys(query)
    search_bar.submit()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article")))
    elements = driver.find_elements(By.CSS_SELECTOR, "article")
    return [element.text.strip() for element in elements if element.text.strip()]


# Content types that complete a download name missing its extension (e.g. a
# bare ``https://host/photo`` URL answered with ``Content-Type: image/png``).
_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
    "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff",
    "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
    "video/x-msvideo": ".avi", "audio/mpeg": ".mp3", "audio/wav": ".wav",
    "audio/ogg": ".ogg", "application/pdf": ".pdf", "application/zip": ".zip",
}


def _suggested_name(url: str, headers) -> str:
    """Download name from Content-Disposition, else the URL path basename.

    Supports both ``filename="name.ext"`` and RFC 5987 ``filename*=UTF-8''...``
    headers; percent-escapes in the URL path are decoded.
    """
    from urllib.parse import unquote, urlparse

    header = (headers.get("content-disposition") or "") if headers else ""
    match = re.search(r"filename\*=utf-8''(?P<value>[^;]+)", header, re.IGNORECASE)
    if match:
        return unquote(match.group("value")).strip()
    match = re.search(r'filename\s*=\s*"?([^";]+)"?', header, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return unquote(Path(urlparse(url).path).name)


def _with_extension(name: str, content_type: str) -> str:
    """Complete ``name`` with the content-type extension when it has none."""
    if Path(name).suffix:
        return name
    media_type = (content_type or "").split(";")[0].strip().lower()
    return name + _CONTENT_TYPE_EXT.get(media_type, "")


# ------------------------------------------------------------------- tools


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
def read_text_file(file: str) -> str:
    """
    Read a text file and return its contents as a string.
    Useful when user asks for content description.
    """
    file_path = Path(file).resolve()
    if file_path.exists() and file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: {exc!r}"
    return f"Unable to read '{file}'"


@dump_func_call
def write_text_file(file: str, content: str, overwrite: bool = False) -> str:
    """
    Write a text file with the given its path and string content.
    Useful when user asks for content generation, e.g. code generation.
    Existing files are only replaced when ``overwrite`` is True.
    """
    dest_path = Path(file).resolve()
    if dest_path.exists():
        if dest_path.is_dir():
            return f"Error: '{dest_path}' is a directory."
        if not overwrite:
            return "File already exists."
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    return f"File created at '{dest_path.resolve()}'"


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
def webpage_headlines(url: str = "https://www.ansa.it/english/", relevant_text_length: int = 20, timeout_s: int = 10) -> str:
    """
    Automatic extraction of headlines from a URL. Returns headlines as string.
    Useful when user is asking to summarize webpage headlines given its URL.
    """
    soup = _fetch_soup(url, timeout_s)
    if isinstance(soup, str):
        return soup
    return f"\nFrom [{url}]\n" + "\n".join(extract_text(soup, ["h1", "h2"], relevant_text_length))


@dump_func_call
def webpage_content(url: str, relevant_text_length: int = 20, timeout_s: int = 10) -> str:
    """
    Automatic extraction paragraph content from a URL as string.
    Useful when user is asking to summarize the webpage content of a webpage given its URL.
    """
    soup = _fetch_soup(url, timeout_s)
    if isinstance(soup, str):
        return soup
    return f"\nFrom [{url}]\n" + "\n".join(extract_text(soup, ["p"], relevant_text_length))


@dump_func_call
def web_search(query: str, n_results: int = 5, timeout_s: float = 10.0) -> str:
    """
    Search web pages based on the query.
    Useful when user asks for search on a topic.
    """
    driver = _create_chrome_driver()
    try:
        results = _collect_results(driver, query, timeout_s)
        return format_search_results(results, n_results)
    except Exception as exc:
        return f"Error: {exc!r}"
    finally:
        driver.quit()


@dump_func_call
def download_file(
        url: str,
        out_dir: str | None = None,
        filename: str | None = None,
        overwrite: bool = False,
        timeout_s: float = 30.0,
    ) -> str:
    """
    Download a file such as a picture, movie, audio or any other file from a URL
    and save it on disk. The file is streamed in chunks to ``out_dir`` (default:
    current working directory) under ``filename`` when given, else the name
    suggested by the Content-Disposition header or the URL itself (missing
    extensions are completed from the Content-Type). Existing files are only
    replaced when ``overwrite`` is True. Returns the saved path and its size.
    """
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; live-assistant)"}
    try:
        response = requests.get(url, stream=True, timeout=timeout_s, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        return f"Error fetching URL: {exc}"

    headers = getattr(response, "headers", {}) or {}
    name = filename or _suggested_name(url, headers)
    name = _with_extension(name or f"downloaded_{datetime.now():%Y%m%d_%H%M%S}",
                           headers.get("content-type", ""))
    out_directory = Path(out_dir) if out_dir is not None else Path.cwd()
    dest_path = out_directory / safe_filename(name)
    if dest_path.exists():
        if dest_path.is_dir():
            return f"Error: '{dest_path}' is a directory."
        if not overwrite:
            return "File already exists."

    try:
        out_directory.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)
    except Exception as exc:
        return f"Error downloading: {exc}"
    return f"Saved '{dest_path.resolve()}' ({dest_path.stat().st_size:,} bytes)."


# ------------------------------------------------------------------ archives

_ZIP_SUFFIXES = (".zip",)
_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
_SUPPORTED_ARCHIVES = (
    "zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and gz"
)


def _archive_stem(path: Path) -> str:
    """``foo.tar.gz`` -> ``foo``; ``foo.zip`` -> ``foo``."""
    name = path.name.lower()
    for suffix in _TAR_SUFFIXES + _ZIP_SUFFIXES + (".7z", ".gz"):
        if name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _archive_kind(path: Path) -> str:
    """Map an archive path to its extractor: 'zip', 'tar', '7z', 'gz' or ''."""
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in _ZIP_SUFFIXES):
        return "zip"
    if any(name.endswith(suffix) for suffix in _TAR_SUFFIXES):
        return "tar"
    if name.endswith(".7z"):
        return "7z"
    if name.endswith(".gz"):
        return "gz"
    return ""


def _extract_zip(path: Path, dest: Path) -> int:
    """Extract a zip archive, rejecting entries that escape ``dest`` (zip slip)."""
    import zipfile

    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        root = dest.resolve()
        for member in members:
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive entry '{member.filename}'")
        for member in members:
            archive.extract(member, dest)
        return len(members)


def _extract_tar(path: Path, dest: Path) -> int:
    """Extract a tar archive; ``filter='data'`` blocks path traversal (PEP 706)."""
    import tarfile

    with tarfile.open(path) as archive:
        members = archive.getmembers()
        archive.extractall(dest, filter="data")
        return len(members)


def _extract_7z(path: Path, dest: Path) -> int:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "py7zr is required for .7z archives (install it with: pip install py7zr)"
        ) from exc
    with py7zr.SevenZipFile(path) as archive:
        entries = len(archive.list())
        archive.extractall(path=dest)
        return entries


def _extract_gz(path: Path, dest: Path) -> int:
    """Decompress a bare ``.gz`` file (not a tar) into ``dest/<stem>``."""
    import gzip
    import shutil

    target = dest / _archive_stem(path)
    with gzip.open(path, "rb") as source, open(target, "wb") as output:
        shutil.copyfileobj(source, output)
    return 1


@dump_func_call
def extract_archive(archive: str, out_dir: str | None = None, overwrite: bool = False) -> str:
    """
    Unzip / unpack a compressed archive and save its content into a folder.
    Useful when user asks to extract, unzip, unpack or decompress an archive.
    Supported formats: zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and
    single-file gz. By default the content is extracted into a new folder named
    after the archive (e.g. ``data.zip`` -> ``data/``); ``out_dir`` overrides it.
    A non-empty destination is only written when ``overwrite`` is True. Returns
    the destination folder and the number of extracted entries.
    """
    archive_path = Path(archive).resolve()
    if not archive_path.exists():
        return f"Error: '{archive_path}' does not exist!"
    if archive_path.is_dir():
        return f"Error: '{archive_path}' is a directory!"
    kind = _archive_kind(archive_path)
    if not kind:
        return f"Error: unsupported archive type '{archive_path.suffix}'. Supported: {_SUPPORTED_ARCHIVES}."

    if kind == "gz":  # single decompressed file, extracted beside the archive
        dest = Path(out_dir) if out_dir is not None else archive_path.parent
        if (dest / _archive_stem(archive_path)).exists() and not overwrite:
            return "File already exists."
    else:
        dest = Path(out_dir) if out_dir is not None else archive_path.parent / _archive_stem(archive_path)
        if dest.exists() and any(dest.iterdir()) and not overwrite:
            return f"Destination '{dest}' is not empty. Use overwrite=True to extract into it."
    try:
        dest.mkdir(parents=True, exist_ok=True)
        extractors = {
            "zip": _extract_zip, "tar": _extract_tar,
            "7z": _extract_7z, "gz": _extract_gz,
        }
        entries = extractors[kind](archive_path, dest)
    except Exception as exc:
        return f"Error extracting '{archive_path.name}': {exc}"
    return f"Extracted {entries} entries from '{archive_path.name}' to '{dest.resolve()}'."


@dump_func_call
def capture_image(out_file: str = CAPTURED_IMAGE, camera_index: int = 0) -> str:
    """
    Captures a frame from a camera and saves it as an image file.
    Useful when user asks for taking a picture.
    """
    capture_factory = _VIDEO_CAPTURE_FACTORY
    imwrite = _IMAGE_WRITER
    if capture_factory is None or imwrite is None:
        import cv2
        capture_factory = capture_factory or cv2.VideoCapture
        imwrite = imwrite or cv2.imwrite

    camera = capture_factory(camera_index)
    if not camera.isOpened():
        return f"Webcam '{camera_index}' not available"
    ret, frame = camera.read()
    camera.release()
    if not ret:
        return "Unable to capture frame"
    if not imwrite(out_file, frame):
        return f"Unable to save image '{out_file}'"
    return str(Path(out_file).resolve())


@dump_func_call
def fetch_financial_data(
        symbol: str,
        interval: ValidYF.interval = '1mo',
        period: ValidYF.period = '1y',
        start_date: str = None,  # YYYY-MM-DD
        end_date: str = None,  # YYYY-MM-DD
        to_string: bool = True
    ) -> "pd.DataFrame | str":
    """
    Fetch closing price of a stock symbol within a time period with a time interval.
    Useful when user asks for getting financial info of a financial product given its symbol. 
    """
    import yfinance as yf

    try:
        data = yf.download(symbol, period=period, interval=interval, start=start_date, end=end_date)
    except Exception as e:
        return f"Error: {e}"
    selected_data = select_financial_columns(data, symbol)
    if isinstance(selected_data, str):
        return selected_data
    if to_string:
        return selected_data.to_string()
    return selected_data


@dump_func_call
def save_financial_chart(
        symbol: str,
        interval: ValidYF.interval = '1mo',
        period: ValidYF.period = '1y',
        out_dir: str | None = None,
    ) -> str:
    """
    Fetch closing price of a stock symbol within a time period with a time interval.
    The closing price are plotted in a chart that is saved as image.
    Useful when user asks for plotting a financial product given its symbol. 
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    def set_axis():
        date_locator = mdates.AutoDateLocator()
        date_formatter = mdates.AutoDateFormatter(date_locator)
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.title(f'{symbol} Price Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)

    def set_chart():
        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

    try:
        data = fetch_financial_data(symbol, interval, period, to_string=False)
        if isinstance(data, str):
            return data
        out_directory = Path(out_dir) if out_dir is not None else Path.cwd()
        out_directory.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(
            data['Date'], data['Close'], label='Close Price', marker='o', linestyle='-', color='blue', markersize=5
        )
        set_axis()
        set_chart()
        plot_path = out_directory / f"{safe_filename(symbol)}.jpg"
        plt.savefig(fname=str(plot_path), format='jpg')
        plt.close()
        return f"Plot saved to '{plot_path}'"
    except Exception as e:
        return f"Error: {e}"


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


def __getattr__(name: str):
    """Lazily resolve optional heavy dependencies kept for API compatibility.

    ``requests`` is imported on first access (instead of at module import) so
    that importing ``src.tools`` stays cheap for callers that never fetch the
    web. Test seams such as ``tools.requests`` continue to work.
    """
    if name == "requests":
        import requests
        return requests
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    print(web_search("quantum computing wikipedia"))
    # print(webpage_content("https://www.nist.gov/quantum-information-science/quantum-computing-explained"))
    # print(save_financial_chart("STLA"))
