"""
Extractive text summarization.

The pipeline mirrors ``samples/sentence_transformers/text_summarization.py``:
split the input into sentences, embed each sentence, compute the pairwise
cosine similarity, run LexRank to score centrality, then return the most
central sentences in their original document order.

Input can be raw text (``summarize``) or any file type supported by
``src.document_loader`` -- text-like, PDF, DOCX, XLSX, HTML and images
(``summarize_file``). Heavy dependencies (``sentence_transformers``, ``nltk``)
are imported lazily so this module can be imported -- and unit tested -- without
downloading or loading a real model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.lex_rank import degree_centrality_scores

__all__ = [
    "DEFAULT_RATIO",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_SENTENCES",
    "MAX_INPUT_CHARS",
    "split_sentences",
    "is_model_available",
    "Summarizer",
]

DEFAULT_RATIO = 0.1
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_SENTENCES = 0  # 0 -> derive the count from the ratio
MAX_INPUT_CHARS = 20000  # guard against pathological documents

# Files that must be present in a saved SentenceTransformer folder.
_MODEL_MARKERS = (
    "modules.json",
    "config_sentence_transformers.json",
    "pytorch_model.bin",
    "model.safetensors",
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _regex_split(text: str) -> list[str]:
    """Naive sentence splitter used when NLTK data is unavailable."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    Uses NLTK when available (and its ``punkt`` data is installed); otherwise
    falls back to a regex splitter so the tool never depends on a runtime
    network download.
    """
    try:
        import nltk

        return [s for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        return _regex_split(text)


def is_model_available(model_path) -> bool:
    """True when ``model_path`` is a directory containing a saved model."""
    path = Path(model_path)
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _MODEL_MARKERS)


class Summarizer:
    """Loads a sentence-embedding model lazily and summarizes text.

    ``model_factory`` and ``sentence_tokenizer`` are dependency-injection seams
    used by the unit tests to avoid loading a real model or NLTK data.
    """

    def __init__(
        self,
        model_path,
        device: str = "cpu",
        ratio: float = DEFAULT_RATIO,
        threshold: float = DEFAULT_THRESHOLD,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        model_factory: Optional[Callable[[str, str], object]] = None,
        sentence_tokenizer: Optional[Callable[[str], list[str]]] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.ratio = ratio
        self.threshold = threshold
        self.max_sentences = max_sentences
        self._model_factory = model_factory
        self._sentence_tokenizer = sentence_tokenizer
        self._model = None

    # -------------------------------------------------------------- internals

    def _tokenize(self, text: str) -> list[str]:
        if self._sentence_tokenizer is not None:
            return [s for s in self._sentence_tokenizer(text) if s.strip()]
        return split_sentences(text)

    def _embedder(self):
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(str(self.model_path), self.device)
            else:
                if not is_model_available(self.model_path):
                    raise FileNotFoundError(
                        f"Summarization model not found at '{self.model_path}'. "
                        "See README.md for how to store the model."
                    )
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    str(self.model_path), device=self.device
                )
        return self._model

    def _resolve_k(self, max_sentences: Optional[int], n_sentences: int) -> int:
        if max_sentences is not None and max_sentences > 0:
            k = max_sentences
        elif self.max_sentences > 0:
            k = self.max_sentences
        else:
            k = max(1, round(n_sentences * self.ratio))
        return max(1, min(int(k), n_sentences))

    # ----------------------------------------------------------------- public

    def summarize_file(self, path, max_sentences: Optional[int] = None) -> str:
        """Summarize the text extracted from a document of any supported type.

        ``src.document_loader.load_document`` extracts plain text from text-like
        files, PDF, DOCX, XLSX, HTML and images; load errors are reported as a
        message instead of being raised.
        """
        from src.document_loader import load_document  # import-safe, lazy deps

        document = load_document(path)
        if document.error:
            return f"Unable to read '{document.source}': {document.error}"
        if document.kind == "image":
            return f"No extractable text in '{document.source}' (image file)."
        return self.summarize(document.text, max_sentences=max_sentences)

    def summarize(self, text: str, max_sentences: Optional[int] = None) -> str:
        """Return an extractive summary of ``text`` as a single string."""
        text = (text or "").strip()
        if not text:
            return "No text to summarize."
        if len(text) > MAX_INPUT_CHARS:
            text = text[:MAX_INPUT_CHARS]

        sentences = self._tokenize(text)
        if len(sentences) <= 1:
            return text

        k = self._resolve_k(max_sentences, len(sentences))

        embeddings = np.asarray(self._embedder().encode(sentences), dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        similarity = embeddings @ embeddings.T
        scores = degree_centrality_scores(similarity, self.threshold)

        # Keep the top-k by centrality, then restore document order.
        top_indices = sorted(np.argsort(-scores)[:k])
        return " ".join(sentences[i] for i in top_indices).strip()
