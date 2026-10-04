"""Web tools: DuckDuckGo search, page text extraction and file downloads."""

import re
from datetime import datetime
from pathlib import Path

from src.common import dump_func_call
from .files import safe_filename

__all__ = [
    "download_file", "extract_text", "format_search_results", "web_search",
    "webpage_content", "webpage_headlines",
]


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