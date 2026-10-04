"""Document-summarization tool: extractive summaries via LexRank + embeddings.

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure`, the heavy ``Summarizer`` is built
lazily on first use and cached at module level (loading the embedding model is
slow), and tests override the ``_SUMMARIZER_FACTORY`` seam.
"""

from __future__ import annotations

from src.common import dump_func_call

__all__ = ["configure", "summarize_doc"]

_config_cache = None    # [summarization] section injected via configure() (None -> defaults)
_summarizer = None      # cached shared Summarizer instance
_SUMMARIZER_FACTORY = None  # test seam: factory(section) -> Summarizer


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Only the ``[summarization]`` section is kept; a missing
    section means the Summarizer defaults apply. Also drops the cached
    summarizer, so a config change is honored on the next call.
    """
    global _config_cache, _summarizer
    _config_cache = getattr(config, "summarization", None)
    _summarizer = None


def _shared_summarizer():
    """Build (once) and return the module-wide ``Summarizer``."""
    global _summarizer
    if _summarizer is None:
        from src.summarizer import Summarizer  # heavy: sentence-transformers

        factory = _SUMMARIZER_FACTORY or Summarizer
        cfg = _config_cache
        _summarizer = factory(
            model_path=cfg.model_path,
            device=getattr(cfg, "device", "cpu"),
            ratio=getattr(cfg, "ratio", 0.1),
            threshold=getattr(cfg, "threshold", 0.5),
            max_sentences=getattr(cfg, "max_sentences", 0),
        )
    return _summarizer


@dump_func_call
def summarize_doc(file: str, max_sentences: int = 0) -> str:
    """
    Summarize the content of a document into its most important sentences.
    Useful when the user asks to summarize, condense, shorten or outline a
    document. ``file`` is the path of the document to read; text, PDF, DOCX,
    XLSX, HTML and image files are supported via ``src.document_loader``.
    ``max_sentences`` = 0 uses the ratio configured in the
    ``[summarization]`` section.
    """
    try:
        return _shared_summarizer().summarize_file(file, max_sentences or None)
    except Exception as e:
        return f"Error: {e}"
