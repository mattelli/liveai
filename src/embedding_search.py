"""
Semantic search over a local text corpus.

The pipeline mirrors ``samples/sentence_transformers/semantic_search_quora_faiss.py``:
collect passages from a directory of text files, embed them with a
``SentenceTransformer``, L2-normalize the embeddings (so inner product equals
cosine similarity), persist them to a single pickle, then answer queries with a
FAISS exact inner-product index.

Heavy dependencies (``faiss``, ``sentence_transformers``) are imported lazily so
this module can be imported -- and unit tested -- without loading a real model or
building a real index.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.summarizer import is_model_available, split_sentences

__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EXTENSIONS",
    "collect_texts",
    "normalize_rows",
    "build_passages",
    "load_corpus",
    "save_corpus",
    "EmbeddingCorpus",
]

DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 3
DEFAULT_EXTENSIONS = (".txt", ".md")


def _normalize_extensions(value) -> tuple[str, ...]:
    """Turn ``"txt, md"`` or ``("txt", ".md")`` into ``(".txt", ".md")``."""
    items = value.split(",") if isinstance(value, str) else list(value)
    result: list[str] = []
    for item in items:
        item = str(item).strip().lower()
        if not item:
            continue
        result.append(item if item.startswith(".") else "." + item)
    return tuple(result)


def collect_texts(directory, extensions=DEFAULT_EXTENSIONS) -> list[str]:
    """Read every supported text file under ``directory`` (recursively).

    Returns the file contents sorted by path for deterministic corpus builds.
    Raises ``FileNotFoundError`` / ``NotADirectoryError`` for a bad directory.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: '{root}'")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: '{root}'")

    suffixes = _normalize_extensions(extensions)
    texts: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if content.strip():
            texts.append(content)
    return texts


def normalize_rows(matrix) -> np.ndarray:
    """L2-normalize each row; zero rows are left as zeros (norm guarded)."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def build_passages(texts, chunk_size=DEFAULT_CHUNK_SIZE, tokenizer=None) -> list[str]:
    """Split each text into sentences and group ``chunk_size`` per passage."""
    splitter = tokenizer or split_sentences
    size = max(1, int(chunk_size))
    passages: list[str] = []
    for text in texts:
        sentences = [s.strip() for s in splitter(text) if s.strip()]
        for start in range(0, len(sentences), size):
            passage = " ".join(sentences[start:start + size]).strip()
            if passage:
                passages.append(passage)
    return passages


def load_corpus(path) -> dict:
    """Load a corpus pickle and validate its shape."""
    with open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "passages" not in payload or "embeddings" not in payload:
        raise ValueError(f"Invalid corpus file: '{path}'")
    return payload


def save_corpus(path, payload: dict) -> None:
    """Write a corpus pickle, creating parent directories as needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(payload, handle)


def _default_index_factory(embeddings):
    """Build an exact FAISS inner-product index over normalized embeddings."""
    import faiss

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    return index


class EmbeddingCorpus:
    """Builds and queries a local corpus of embedded passages.

    ``model_factory``, ``index_factory`` and ``sentence_tokenizer`` are
    dependency-injection seams used by the unit tests to avoid loading a real
    model or building a real FAISS index.
    """

    def __init__(
        self,
        model_path,
        device: str = "cpu",
        extensions=DEFAULT_EXTENSIONS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        top_k: int = DEFAULT_TOP_K,
        model_factory: Optional[Callable[[str, str], object]] = None,
        index_factory: Optional[Callable[[np.ndarray], object]] = None,
        sentence_tokenizer: Optional[Callable[[str], list[str]]] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.extensions = _normalize_extensions(extensions)
        self.chunk_size = chunk_size
        self.top_k = top_k
        self._model_factory = model_factory
        self._index_factory = index_factory
        self._sentence_tokenizer = sentence_tokenizer
        self._model = None

    # -------------------------------------------------------------- internals

    def _tokenizer(self, text: str) -> list[str]:
        if self._sentence_tokenizer is not None:
            return self._sentence_tokenizer(text)
        return split_sentences(text)

    def _embedder(self):
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(str(self.model_path), self.device)
            else:
                if not is_model_available(self.model_path):
                    raise FileNotFoundError(
                        f"Embedding model not found at '{self.model_path}'. "
                        "See README.md for how to store the model."
                    )
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    str(self.model_path), device=self.device
                )
        return self._model

    def _encode(self, texts) -> np.ndarray:
        embeddings = np.asarray(
            self._embedder().encode(list(texts), convert_to_numpy=True), dtype=float
        )
        return normalize_rows(embeddings)

    # ----------------------------------------------------------------- public

    def store(self, directory: str, output_file: str, overwrite: bool = False) -> str:
        """Build a corpus from ``directory`` and persist it to ``output_file``."""
        output = Path(output_file).resolve()
        if output.exists() and not overwrite:
            return "File already exists."

        texts = collect_texts(directory, self.extensions)
        if not texts:
            return f"No text files found in '{directory}'"

        passages = build_passages(texts, self.chunk_size, self._tokenizer)
        if not passages:
            return "No passages extracted from the input files."

        embeddings = self._encode(passages)
        payload = {
            "version": 1,
            "model_path": str(self.model_path),
            "passages": passages,
            "embeddings": embeddings,
        }
        save_corpus(output, payload)
        return f"Corpus with {len(passages)} passages saved to '{output}'"

    def query(self, corpus_file: str, query: str, top_k: Optional[int] = None) -> str:
        """Return the ``top_k`` passages most relevant to ``query``."""
        query = (query or "").strip()
        if not query:
            return "No query provided."

        path = Path(corpus_file).resolve()
        if not path.exists():
            return f"Corpus not found at '{path}'. Build it first with store_corpus."

        payload = load_corpus(path)
        passages = list(payload.get("passages") or [])
        if not passages:
            return "Corpus is empty."

        embeddings = np.asarray(payload.get("embeddings"), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(passages):
            return f"Invalid corpus file: '{path}'."

        query_vec = self._encode([query])[0]
        if query_vec.shape[0] != embeddings.shape[1]:
            return (
                "Query embedding size does not match the corpus; "
                "rebuild the corpus with the configured model."
            )

        index = (self._index_factory or _default_index_factory)(embeddings)
        k = int(top_k) if top_k else int(self.top_k)
        k = max(1, min(k, len(passages)))

        scores, ids = index.search(query_vec.reshape(1, -1), k)
        lines = [f"Top {k} matches for '{query}':"]
        rank = 0
        for score, idx in zip(np.ravel(scores), np.ravel(ids)):
            idx = int(idx)
            if idx < 0:
                continue
            rank += 1
            lines.append(f"{rank}) [{float(score):.3f}] {passages[idx]}")
        return "\n".join(lines)
