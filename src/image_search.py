"""
Image search in a folder with a CLIP model.

The pipeline mirrors ``samples/sentence_transformers/image_search.py``: every
image under a directory is embedded with the CLIP model, the text query is
embedded into the same shared space, and cosine similarity (L2-normalized
vectors) ranks the images. The ``top_k`` best-matching image paths are returned
in descending similarity order.

Heavy dependencies (``sentence_transformers``, ``PIL``) are imported lazily so
this module can be imported -- and unit tested -- without loading a real model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.document_loader import IMAGE_EXTENSIONS
from src.embedding_search import normalize_rows
from src.summarizer import is_model_available

__all__ = [
    "DEFAULT_TOP_K",
    "ImageSearch",
]

DEFAULT_TOP_K = 3


class ImageSearch:
    """Ranks the images of a folder against a text query with a CLIP model.

    ``model_factory`` is a dependency-injection seam used by the unit tests to
    avoid loading a real model. Image embeddings are cached per (path, mtime),
    so repeated queries over an unchanged folder do not re-encode the images.
    Unreadable or corrupt image files are skipped.
    """

    def __init__(
        self,
        model_path,
        device: str = "cpu",
        top_k: int = DEFAULT_TOP_K,
        model_factory: Optional[Callable[[str, str], object]] = None,
        extensions=IMAGE_EXTENSIONS,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.top_k = top_k
        self._model_factory = model_factory
        self.extensions = tuple(
            ext if ext.startswith(".") else "." + ext
            for ext in (str(ext).lower().strip() for ext in extensions)
            if ext
        )
        self._model = None
        self._image_cache: dict[str, tuple[float, np.ndarray]] = {}

    # -------------------------------------------------------------- internals

    def _encoder(self):
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(str(self.model_path), self.device)
            else:
                if not is_model_available(self.model_path):
                    raise FileNotFoundError(
                        f"Image search model not found at '{self.model_path}'. "
                        "See README.md for how to store the model."
                    )
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(str(self.model_path), device=self.device)
        return self._model

    def _collect_images(self, directory: Path) -> list[Path]:
        """Every supported image file under ``directory`` (recursively, sorted)."""
        return sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        )

    def _encode_image(self, path: Path) -> Optional[np.ndarray]:
        """Embed one image; ``None`` when the file cannot be opened or read."""
        from PIL import Image

        try:
            key = str(path.resolve())
            mtime = path.stat().st_mtime
            cached = self._image_cache.get(key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            with Image.open(key) as image:
                payload = image.convert("RGB")
            vector = np.asarray(
                self._encoder().encode(payload, convert_to_numpy=True), dtype=float
            )
            self._image_cache[key] = (mtime, vector)
            return vector
        except Exception:
            return None

    def _resolve_k(self, top_k: Optional[int], n_images: int) -> int:
        k = top_k if top_k and int(top_k) > 0 else self.top_k
        if not k or int(k) <= 0:
            k = DEFAULT_TOP_K
        return max(1, min(int(k), n_images))

    # ----------------------------------------------------------------- public

    def search(self, directory, query: str, top_k: Optional[int] = None) -> list[tuple[float, str]]:
        """Return the ``(score, path)`` pairs of the ``top_k`` best-matching images.

        Images are ranked by cosine similarity to the embedded query, descending.
        Raises ``FileNotFoundError``/``NotADirectoryError`` for a bad ``directory``
        and ``ValueError`` for an empty query; returns ``[]`` when the folder has
        no readable supported images.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("No query provided.")

        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: '{root}'")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: '{root}'")

        kept: list[Path] = []
        vectors: list[np.ndarray] = []
        for path in self._collect_images(root):
            vector = self._encode_image(path)
            if vector is not None:
                kept.append(path)
                vectors.append(vector)
        if not kept:
            return []

        image_vecs = normalize_rows(np.asarray(vectors, dtype=float))
        query_vec = normalize_rows(
            np.asarray(self._encoder().encode([query], convert_to_numpy=True), dtype=float)
        )[0]

        scores = image_vecs @ query_vec
        k = self._resolve_k(top_k, len(kept))
        order = np.argsort(-scores)[:k]
        return [
            (float(scores[int(index)]), str(kept[int(index)].resolve()))
            for index in order
        ]