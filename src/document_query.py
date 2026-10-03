"""
In-document search with a CrossEncoder.

The pipeline mirrors
``samples/sentence_transformers/in_document_search_crossencoder.py``: the
document text is split into sentences, up to ``chunk_size`` consecutive
sentences form one passage, then a CrossEncoder scores every
``[query, passage]`` pair and the ``top_k`` highest-scoring passages are
returned in descending score order.

The document is read with ``src.document_loader``, so any supported file type
works: text-like files, PDF (passages keep their page number), DOCX, XLSX,
HTML and images.

Note: every query scores all passages of the document, so very large documents
make each query slower (the passage list itself is cached per file mtime).

Heavy dependencies (``sentence_transformers``, ``nltk``) are imported lazily so
this module can be imported -- and unit tested -- without loading a real model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.document_loader import load_document
from src.embedding_search import DEFAULT_CHUNK_SIZE, build_passages, has_content
from src.summarizer import is_model_available, split_sentences

__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_CHUNK_SIZE",
    "DocumentQuery",
]

DEFAULT_TOP_K = 5


class DocumentQuery:
    """Scores passages of one document against a query with a CrossEncoder.

    ``cross_encoder_factory`` and ``sentence_tokenizer`` are dependency-injection
    seams used by the unit tests to avoid loading a real model or NLTK data.
    Passages are rebuilt only when the file's mtime changes (per-process cache).
    """

    def __init__(
        self,
        model_path,
        device: str = "cpu",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        top_k: int = DEFAULT_TOP_K,
        cross_encoder_factory: Optional[Callable[[str, str], object]] = None,
        sentence_tokenizer: Optional[Callable[[str], list[str]]] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.chunk_size = chunk_size
        self.top_k = top_k
        self._cross_encoder_factory = cross_encoder_factory
        self._sentence_tokenizer = sentence_tokenizer
        self._cross_encoder = None
        self._passage_cache: dict[str, tuple[float, list[dict]]] = {}

    # -------------------------------------------------------------- internals

    def _tokenizer(self, text: str) -> list[str]:
        if self._sentence_tokenizer is not None:
            return self._sentence_tokenizer(text)
        return split_sentences(text)

    def _encoder(self):
        if self._cross_encoder is None:
            if self._cross_encoder_factory is not None:
                self._cross_encoder = self._cross_encoder_factory(
                    str(self.model_path), self.device
                )
            else:
                if not is_model_available(self.model_path):
                    raise FileNotFoundError(
                        f"Cross-encoder model not found at '{self.model_path}'. "
                        "See README.md for how to store the model."
                    )
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(
                    str(self.model_path), device=self.device
                )
        return self._cross_encoder

    def _passages(self, path: Path) -> list[dict]:
        """Load ``path`` and split its text into passage records.

        Raises ``FileNotFoundError``/``ValueError`` with a friendly message for
        unreadable files and image-only documents (load errors are captured by
        ``load_document``, not raised).
        """
        document = load_document(path)
        if document.error:
            raise FileNotFoundError(f"Unable to read '{document.source}': {document.error}")
        if document.kind == "image":
            raise ValueError(f"No searchable text in '{document.source}' (image file).")
        records: list[dict] = []
        # Paginated documents (PDF) expose per-page sections so each passage
        # keeps its real page number.
        sections = document.sections or [(document.pages, document.text)]
        for page, section_text in sections:
            for passage in build_passages([section_text], self.chunk_size, self._tokenizer):
                if not has_content(passage):
                    continue
                records.append({"text": passage, "page": page})
        return records

    def _cached_passages(self, path: Path) -> list[dict]:
        """Rebuild the passage list only when the file's mtime changed."""
        key = str(path.resolve())
        mtime = path.stat().st_mtime
        cached = self._passage_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        records = self._passages(path)
        self._passage_cache[key] = (mtime, records)
        return records

    def _resolve_k(self, top_k: Optional[int], n_passages: int) -> int:
        k = top_k if top_k and top_k > 0 else self.top_k
        if not k or int(k) <= 0:
            k = DEFAULT_TOP_K
        return max(1, min(int(k), n_passages))

    # ----------------------------------------------------------------- public

    def query(self, path, query: str, top_k: Optional[int] = None) -> str:
        """Return the ``top_k`` passages most relevant to ``query``.

        Passages are scored ``[query, passage]`` pair by pair with the
        CrossEncoder and returned in descending score order (highest first),
        with page numbers for paginated documents.
        """
        query = (query or "").strip()
        if not query:
            return "No query provided."

        path = Path(path)
        if not path.exists() or not path.is_file():
            return f"Unable to read '{path}': file not found."

        try:
            records = self._cached_passages(path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return str(exc)

        if not records:
            return f"No searchable text found in '{path.name}'."

        k = self._resolve_k(top_k, len(records))
        inputs = [[query, record["text"]] for record in records]
        scores = np.asarray(self._encoder().predict(inputs), dtype=float).ravel()
        order = np.argsort(-scores)[:k]

        lines = [f"Top {k} matches for '{query}' in '{path.name}':"]
        for rank, idx in enumerate(order, start=1):
            record = records[int(idx)]
            where = f" (page {record['page']})" if record.get("page") else ""
            lines.append(f"{rank}) [{float(scores[idx]):.3f}]{where} {record['text']}")
        return "\n".join(lines)