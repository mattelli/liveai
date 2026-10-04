"""Corpus tools: build and search named FAISS corpora (text + images).

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure`, the heavy ``EmbeddingCorpus`` is
built lazily on first use and cached at module level, and tests override the
``_EMBEDDING_CORPUS_FACTORY`` seam.
"""

from __future__ import annotations

from src.common import dump_func_call
from src.embedding_search import list_corpora as list_corpus_entries, resolve_corpus_dir

__all__ = ["configure", "list_corpora", "store_corpus", "query_corpus"]

_config_cache = None             # [embedding] section injected via configure()
_emb_corpus = None               # cached shared EmbeddingCorpus instance
_EMBEDDING_CORPUS_FACTORY = None  # test seam: factory(section) -> EmbeddingCorpus


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Only the ``[embedding]`` section is kept; a missing
    section means the EmbeddingCorpus defaults apply. Also drops the cached
    instance, so a config change is honored on the next call.
    """
    global _config_cache, _emb_corpus
    _config_cache = getattr(config, "embedding", None)
    _emb_corpus = None


def _shared_embedding_corpus():
    """Build (once) and return the module-wide ``EmbeddingCorpus``."""
    global _emb_corpus
    if _emb_corpus is None:
        from src.document_loader import IMAGE_EXTENSIONS
        from src.embedding_search import (
            DEFAULT_CHUNK_SIZE, DEFAULT_EXTENSIONS, DEFAULT_TOP_K,
            EmbeddingCorpus,
        )

        factory = _EMBEDDING_CORPUS_FACTORY or EmbeddingCorpus
        cfg = _config_cache
        _emb_corpus = factory(
            model_path=cfg.model_path,
            device=getattr(cfg, "device", "cpu"),
            extensions=getattr(cfg, "extensions", DEFAULT_EXTENSIONS),
            chunk_size=getattr(cfg, "chunk_size", DEFAULT_CHUNK_SIZE),
            top_k=getattr(cfg, "top_k", DEFAULT_TOP_K),
            image_model_path=getattr(cfg, "image_model_path", None),
            multimodal=getattr(cfg, "multimodal", False),
            image_extensions=getattr(cfg, "image_extensions", IMAGE_EXTENSIONS),
        )
    return _emb_corpus


@dump_func_call
def list_corpora() -> str:
    """
    List the available local corpora (name, mode, record counts, sources, built time).
    Useful before query_corpus when the user refers to a topic instead of a corpus name.
    """
    if _config_cache is None:
        return "Error: corpora directory is not configured."
    entries = list_corpus_entries(_config_cache.corpora_dir)
    if not entries:
        return "No corpora available."
    lines = [f"Available corpora ({len(entries)}):"]
    for e in entries:
        lines.append(
            f"- {e['name']}: mode={e['mode']}, records={e['records']} "
            f"(text {e['text_records']} / image {e['image_records']}), "
            f"sources={e['sources']}, built={e['built_at']}"
        )
    return "\n".join(lines)


@dump_func_call
def store_corpus(directory: str, corpus_name: str, overwrite: bool = False) -> str:
    """
    Build a named corpus from all supported files in a directory.
    Useful when the user asks to index, embed, or store documents for later semantic search.
    """
    try:
        if not corpus_name or not corpus_name.strip():
            return "Error: a corpus name is required (call list_corpora to see available corpora)."
        if _config_cache is None:
            return "Error: corpora directory is not configured."
        corpus_dir = resolve_corpus_dir(corpus_name, _config_cache.corpora_dir)
        return _shared_embedding_corpus().store(directory, corpus_dir, overwrite)
    except Exception as e:
        return f"Error: {e}"


@dump_func_call
def query_corpus(query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
    """
    Search a named corpus for the content most relevant to a query text.
    Useful when the user asks a question that should be answered from indexed documents
    or when looking for a matching image.
    ``kind`` restricts results to 'text' or 'image'; leave it empty to search both.
    """
    try:
        if not corpus_name or not corpus_name.strip():
            return "Error: a corpus name is required."
        if _config_cache is None:
            return "Error: corpora directory is not configured."
        corpus_dir = resolve_corpus_dir(corpus_name, _config_cache.corpora_dir)
        return _shared_embedding_corpus().query(corpus_dir, query, top_k or None, kind or None)
    except Exception as e:
        return f"Error: {e}"
