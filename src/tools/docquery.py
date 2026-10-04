"""In-document search tool: CrossEncoder scoring of query/passage pairs.

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure`, the heavy ``DocumentQuery`` is built
lazily on first use and cached at module level, and tests override the
``_DOCUMENT_QUERY_FACTORY`` seam.
"""

from __future__ import annotations

from src.common import dump_func_call

__all__ = ["configure", "query_document"]

_config_cache = None         # [document_query] section injected via configure()
_doc_query = None            # cached shared DocumentQuery instance
_DOCUMENT_QUERY_FACTORY = None  # test seam: factory(section) -> DocumentQuery


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Only the ``[document_query]`` section is kept; a missing
    section means the DocumentQuery defaults apply. Also drops the cached
    instance, so a config change is honored on the next call.
    """
    global _config_cache, _doc_query
    _config_cache = getattr(config, "document_query", None)
    _doc_query = None


def _shared_document_query():
    """Build (once) and return the module-wide ``DocumentQuery``."""
    global _doc_query
    if _doc_query is None:
        from src.document_query import (
            DEFAULT_CHUNK_SIZE as QUERY_CHUNK_SIZE,
            DEFAULT_TOP_K as QUERY_TOP_K,
            DocumentQuery,
        )

        factory = _DOCUMENT_QUERY_FACTORY or DocumentQuery
        cfg = _config_cache
        _doc_query = factory(
            model_path=cfg.model_path,
            device=getattr(cfg, "device", "cpu"),
            chunk_size=getattr(cfg, "chunk_size", QUERY_CHUNK_SIZE),
            top_k=getattr(cfg, "top_k", QUERY_TOP_K),
        )
    return _doc_query


@dump_func_call
def query_document(file: str, query: str, top_k: int = 0) -> str:
    """
    Search one document for the passages most relevant to a query.
    Useful when the user asks to find, locate or look up something inside a
    specific file without building a corpus. Any type supported by
    ``src.document_loader`` works: text, PDF (results keep page numbers),
    DOCX, XLSX, HTML, ... A CrossEncoder scores query/passage pairs.
    ``top_k`` = 0 uses the value configured in the ``[document_query]``
    section.
    """
    try:
        if not query or not query.strip():
            return "Error: a query is required."
        return _shared_document_query().query(file, query, top_k or None)
    except Exception as e:
        return f"Error: {e}"
