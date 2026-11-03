"""Tool collection for the Live Assistant agent, split by topic.

- ``system``    : platform info, date/time, working directory, shell commands,
                  background process spawning
- ``files``     : read/write/search/replace files
- ``web``       : DDGS metasearch, webpage extraction, file downloads
- ``archives``  : extract zip/7z/tar/gz archives and zip folder contents
- ``camera``    : webcam frame capture
- ``image_gen`` : text-to-image generation (local Stable-Diffusion pipeline)
- ``finance``   : yfinance data download and chart plotting
- ``summarize`` : extractive document summarization (LexRank + embeddings)
- ``docquery``  : in-document search with a CrossEncoder
- ``imgsearch`` : folder image search with a CLIP model
- ``corpora``   : named FAISS corpora (store + semantic text/image search)
- ``memory``    : personal memory + conversation history (recall/store/clear/compact)
- ``scheduler`` : schedule agent prompts for a future time (one-shot/recurring)
- ``speech``    : Piper text-to-speech toggle
- ``vision``    : image description through the multimodal chat model
- ``mcp``       : MCP (Model Context Protocol) servers as agent tools

Every public tool is re-exported here, so ``from src.tools import <tool>``
keeps working for callers (``src/live_ai.py``) and tests. Mutable test seams
(``_VIDEO_CAPTURE_FACTORY``, ``_SUMMARIZER_FACTORY``, ...) live in their
owning submodule and must be patched there, not on this package.

Config-driven modules follow the ``image_gen`` pattern: ``main.py`` calls
:func:`configure` once after ``load_config`` and every module keeps the
section it needs; session-bound state (chat history, LLM client, speech
device) is bound by ``Brain`` via the ``bind_*`` functions.
"""

from src.common import normalize_extensions, safe_filename

from .system import (
    current_date_time, exit_gracefully, get_working_directory, run_command,
    spawn_process, system_info,
)
from .files import (
    read_text_file, replace_string_in_file,
    search_in_file, search_pattern_in_files, write_text_file,
)
from .web import (
    download_file, extract_webpage, format_search_results, web_books,
    web_images, web_news, web_search, web_videos,
)
from .wiki import wikipedia_search
from .stt import transcribe_audio
from .pyeval import python_eval
from .archives import extract_archive, zip_files
from .camera import CAPTURED_IMAGE, capture_image
from .image_gen import GENERATED_IMAGE, generate_image, unload_image_generator
from .finance import (
    ValidYF, fetch_financial_data, save_financial_chart, select_financial_columns,
)
from .summarize import summarize_doc
from .docquery import query_document
from .imgsearch import image_search
from .corpora import list_corpora, query_corpus, store_corpus
from .memory import bind_summarizer, clear_chat, compact_chat, recall_memory, store_memory
from .scheduler import (
    cancel_scheduled_task, list_scheduled_tasks, schedule_task,
)
from .speech import allow_speech
from .vision import image_response
from .mcp import mcp_call_tool, mcp_list_tools

__all__ = [
    "CAPTURED_IMAGE", "GENERATED_IMAGE", "ValidYF", "allow_speech",
    "cancel_scheduled_task", "capture_image", "clear_chat", "compact_chat",
    "current_date_time", "download_file",
    "exit_gracefully", "extract_archive", "extract_webpage",
    "fetch_financial_data", "format_search_results", "generate_image",
    "get_working_directory", "image_response", "image_search",
    "list_corpora", "list_scheduled_tasks", "mcp_call_tool", "mcp_list_tools",
    "normalize_extensions", "query_corpus",
    "query_document", "read_text_file", "recall_memory",
    "replace_string_in_file", "run_command", "safe_filename",
    "save_financial_chart", "schedule_task", "search_in_file",
    "search_pattern_in_files", "select_financial_columns", "spawn_process",
    "store_corpus",
    "store_memory", "summarize_doc", "system_info", "unload_image_generator",
    "web_books", "python_eval", "transcribe_audio", "wikipedia_search", "web_images", "web_news", "web_search", "web_videos",
    "write_text_file", "zip_files",
]


def configure(config) -> None:
    """Inject the application config into every config-driven tool module.

    Called once by ``main.py`` after ``load_config``; each module keeps the
    section it needs and ignores the rest. ``vision`` is not in the list
    because it is purely session-bound (see ``bind_session``).
    """
    from . import (
        corpora, docquery, image_gen, imgsearch, mcp, memory, scheduler, speech,
        summarize,
    )
    for module in (corpora, docquery, image_gen, imgsearch, mcp, memory,
                   scheduler, speech, summarize):
        module.configure(config)


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