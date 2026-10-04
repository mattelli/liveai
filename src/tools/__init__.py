"""Tool collection for the Live Assistant agent, split by topic.

- ``system``    : platform info, date/time, working directory, shell commands
- ``files``     : read/write/search/replace files + shared filename helpers
- ``web``       : DuckDuckGo search, page extraction, file downloads
- ``archives``  : extract zip/7z/tar/gz archives and zip folder contents
- ``camera``    : webcam frame capture
- ``image_gen`` : text-to-image generation (local Stable-Diffusion pipeline)
- ``finance``   : yfinance data download and chart plotting

Every public tool is re-exported here, so ``from src.tools import <tool>``
keeps working for callers (``src/live_ai.py``) and tests. Mutable test seams
(``_VIDEO_CAPTURE_FACTORY``, ``_create_chrome_driver``, ...) live in their
owning submodule and must be patched there, not on this package.
"""

from .system import (
    current_date_time, exit_gracefully, get_working_directory, run_command,
    system_info,
)
from .files import (
    normalize_extensions, read_text_file, replace_string_in_file, safe_filename,
    search_in_file, search_pattern_in_files, write_text_file,
)
from .web import (
    download_file, extract_text, format_search_results, webpage_content,
    webpage_headlines, web_search,
)
from .archives import extract_archive, zip_files
from .camera import CAPTURED_IMAGE, capture_image
from .image_gen import GENERATED_IMAGE, generate_image, unload_image_generator
from .finance import (
    ValidYF, fetch_financial_data, save_financial_chart, select_financial_columns,
)

__all__ = [
    "CAPTURED_IMAGE", "GENERATED_IMAGE", "ValidYF", "capture_image",
    "current_date_time", "download_file", "exit_gracefully", "extract_archive",
    "extract_text", "fetch_financial_data", "format_search_results",
    "generate_image", "get_working_directory", "normalize_extensions",
    "read_text_file", "replace_string_in_file", "run_command", "safe_filename",
    "save_financial_chart", "search_in_file", "search_pattern_in_files",
    "select_financial_columns", "system_info", "unload_image_generator",
    "web_search", "webpage_content", "webpage_headlines", "write_text_file",
    "zip_files",
]


def __getattr__(name: str):
    """Lazily resolve optional heavy dependencies kept for API compatibility.

    ``requests`` is imported on first access (instead of at package import) so
    that importing ``src.tools`` stays cheap for callers that never fetch the
    web. Test seams such as ``tools.requests`` continue to work.
    """
    if name == "requests":
        import requests
        return requests
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")