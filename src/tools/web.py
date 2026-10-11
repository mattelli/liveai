"""Web tools: DDGS metasearch, webpage extraction and file downloads."""

import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from src.common import dump_func_call
from .files import safe_filename

__all__ = [
    "download_file", "extract_webpage", "format_search_results", "web_books",
    "web_images", "web_news", "web_search", "web_videos",
]


# Result fields rendered per DDGS search kind (https://github.com/deedy5/ddgs).
_FIELDS_BY_KIND = {
    "text": ("title", "href", "body"),
    "images": ("title", "image", "thumbnail", "width", "height", "source"),
    "videos": ("title", "content", "duration", "publisher", "uploader"),
    "news": ("date", "title", "body", "url", "source"),
    "books": ("title", "author", "publisher", "info", "url"),
}


def format_search_results(
    results: list[dict],
    n_results: int = 5,
    fields: Sequence[str] = ("title", "href", "body"),
) -> str:
    """Render DDGS result dicts as the numbered block returned to the model.

    Only ``fields`` present and non-empty in a result are shown.
    """
    lines = ["\nSearch Results:\n"]
    for index, item in enumerate(results[:n_results], 1):
        parts = [f"{field}: {item[field]}" for field in fields if item.get(field)]
        lines.append(f"{index}) " + (" | ".join(parts) if parts else str(item)))
        lines.append(50 * "=")
    return "\n".join(lines) + "\n"


def _ddgs_client(timeout_s: float = 10.0):
    """Return a lazily imported ``DDGS`` client (https://github.com/deedy5/ddgs).

    Test seam: tests patch this function (see ``tests/test_tools.py``) so no
    network access happens. ``DDGS`` accepts an integer timeout in seconds.
    """
    from ddgs import DDGS

    return DDGS(timeout=max(1, int(timeout_s)))


def _format_ddgs_results(kind: str, results: list[dict], n_results: int) -> str:
    """Render ``kind`` results with the fields defined in ``_FIELDS_BY_KIND``."""
    return format_search_results(results, n_results, _FIELDS_BY_KIND[kind])


def _run_ddgs_search(
    kind: str, query: str, n_results: int, timeout_s: float, **params
) -> str:
    """Run the DDGS ``kind`` search (text/images/videos/news/books) and render it."""
    try:
        results = getattr(_ddgs_client(timeout_s), kind)(query, max_results=n_results, **params)
    except Exception as exc:
        return f"Error: {exc}"
    if not results:
        return f"No results found for: {query}"
    return _format_ddgs_results(kind, results, n_results)


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
def extract_webpage(url: str, fmt: str = "text_markdown", timeout_s: float = 10.0) -> str:
    """
    Fetch a URL and extract its page content as markdown/plain/rich text
    through DDGS. Useful when user is asking to summarize or read the content
    of a webpage given its URL.
    """
    if fmt == "content":
        return "Error: fmt='content' returns raw bytes; use text_markdown, text_plain, text_rich or text."
    try:
        result = _ddgs_client(timeout_s).extract(url, fmt=fmt)
    except Exception as exc:
        return f"Error: {exc}"
    content = result.get("content")
    if content is None:
        return f"Error: no content extracted from {url}"
    if isinstance(content, bytes):
        content = content.decode(errors="replace")
    return f"\nFrom [{result.get('url', url)}]\n{content[:4000]}"


@dump_func_call
def web_search(
    query: str,
    n_results: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    timeout_s: float = 10.0,
) -> str:
    """
    Search web pages based on the query through the DDGS metasearch engines
    (bing, duckduckgo, google, ...).
    Useful when user asks for search on a topic.
    """
    try:
        results = _ddgs_client(timeout_s).text(
            query, region=region, safesearch=safesearch,
            timelimit=timelimit, max_results=n_results,
        )
    except Exception as exc:
        return f"Error: {exc}"
    if not results:
        return f"No results found for: {query}"
    return format_search_results(results, n_results)


@dump_func_call
def web_images(
    query: str,
    n_results: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    size: str | None = None,
    color: str | None = None,
    type_image: str | None = None,
    layout: str | None = None,
    license_image: str | None = None,
    timeout_s: float = 10.0,
) -> str:
    """
    Search pictures on the web: direct image URL, thumbnail, size and source.
    Useful when user asks to find pictures/images/photos of something; an
    image URL can be saved to disk with ``download_file``.
    """
    return _run_ddgs_search(
        "images", query, n_results, timeout_s,
        region=region, safesearch=safesearch, timelimit=timelimit,
        size=size, color=color, type_image=type_image,
        layout=layout, license_image=license_image,
    )


@dump_func_call
def web_videos(
    query: str,
    n_results: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    resolution: str | None = None,
    duration: str | None = None,
    timeout_s: float = 10.0,
) -> str:
    """
    Search videos on the web: video URL, duration, publisher and uploader.
    Useful when user asks to find videos of something; a video URL can be
    downloaded with ``download_file``.
    """
    return _run_ddgs_search(
        "videos", query, n_results, timeout_s,
        region=region, safesearch=safesearch, timelimit=timelimit,
        resolution=resolution, duration=duration,
    )


@dump_func_call
def web_news(
    query: str,
    n_results: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    timeout_s: float = 10.0,
) -> str:
    """
    Search recent news articles about a topic: date, source and summary.
    Useful when user asks for the latest news on a topic.
    """
    return _run_ddgs_search(
        "news", query, n_results, timeout_s,
        region=region, safesearch=safesearch, timelimit=timelimit,
    )


@dump_func_call
def web_books(query: str, n_results: int = 5, timeout_s: float = 10.0) -> str:
    """
    Search books by title, author or topic: author, publisher and format info.
    Useful when user asks to find a book.
    """
    return _run_ddgs_search("books", query, n_results, timeout_s)


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