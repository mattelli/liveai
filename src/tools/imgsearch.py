"""Folder image-search tool: CLIP embeddings rank images by a text query.

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure`, the heavy ``ImageSearch`` (CLIP) is
built lazily on first use and cached at module level, and tests override the
``_IMAGE_SEARCH_FACTORY`` seam.
"""

from __future__ import annotations

from src.common import dump_func_call

__all__ = ["configure", "image_search"]

_config_cache = None      # [image_search] section injected via configure()
_img_search = None        # cached shared ImageSearch instance
_IMAGE_SEARCH_FACTORY = None  # test seam: factory(section) -> ImageSearch


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Only the ``[image_search]`` section is kept; a missing
    section means the ImageSearch defaults apply. Also drops the cached
    instance, so a config change is honored on the next call.
    """
    global _config_cache, _img_search
    _config_cache = getattr(config, "image_search", None)
    _img_search = None


def _shared_image_search():
    """Build (once) and return the module-wide ``ImageSearch``."""
    global _img_search
    if _img_search is None:
        from src.image_search import DEFAULT_TOP_K as IMAGE_SEARCH_TOP_K, ImageSearch

        factory = _IMAGE_SEARCH_FACTORY or ImageSearch
        cfg = _config_cache
        _img_search = factory(
            model_path=cfg.model_path,
            device=getattr(cfg, "device", "cpu"),
            top_k=getattr(cfg, "top_k", IMAGE_SEARCH_TOP_K),
        )
    return _img_search


@dump_func_call
def image_search(folder: str, query: str, top_k: int = 0) -> str:
    """
    Search the images inside a folder for the ones best matching a text query,
    using the CLIP model. Useful when the user asks to find, locate or show a
    picture/photo/screenshot in a folder by describing its content. Returns
    the top_k matches as '<rank>) [<cosine score>] <absolute image path>'
    lines, best match first; ``top_k`` = 0 uses the value configured in the
    ``[image_search]`` section (default 3).
    """
    try:
        if not query or not query.strip():
            return "Error: a query is required."
        hits = _shared_image_search().search(folder, query, top_k or None)
        if not hits:
            return f"No readable images found in '{folder}'."
        return "\n".join(
            f"{rank}) [{score:.3f}] {path}"
            for rank, (score, path) in enumerate(hits, start=1)
        )
    except Exception as e:
        return f"Error: {e}"
