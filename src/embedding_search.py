"""
Semantic search over a local document corpus.

The pipeline mirrors ``samples/sentence_transformers/semantic_search_quora_faiss.py``:
collect text passages (and images) from a directory, embed them, L2-normalize the
embeddings (so inner product equals cosine similarity), persist them to a single
pickle, then answer queries with a FAISS exact inner-product index.

Two modes:

- **text**       – text passages are embedded with ``model_path`` and image files
                   are skipped.
- **multimodal** – text passages and images are embedded into one shared space
                   with the CLIP model at ``image_model_path``, so ``query`` can
                   return images as well.

Per-file summaries are appended to a markdown file next to the corpus pickle and
are deliberately **not** stored in the pickle.

Heavy dependencies (``faiss``, ``sentence_transformers``, ``PIL``) are imported
lazily so this module can be imported -- and unit tested -- without loading a
real model or building a real index.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.document_loader import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, iter_documents
from src.summarizer import is_model_available, split_sentences

__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_SUMMARY_NAME",
    "collect_texts",
    "normalize_rows",
    "build_passages",
    "load_corpus",
    "save_corpus",
    "EmbeddingCorpus",
]

DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 3
DEFAULT_EXTENSIONS = SUPPORTED_EXTENSIONS
DEFAULT_SUMMARY_NAME = "SUMMARY.md"


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

    Legacy helper kept for text-only callers/tests. Returns file contents sorted
    by path. Raises ``FileNotFoundError`` / ``NotADirectoryError``.
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
    """Load a v1 (``passages``) or v2 (``records``) corpus pickle and validate it."""
    with open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    valid = (
        isinstance(payload, dict)
        and "embeddings" in payload
        and ("records" in payload or "passages" in payload)
    )
    if not valid:
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
    """Builds and queries a local corpus of embedded passages and images.

    ``model_factory``, ``image_model_factory``, ``index_factory``,
    ``summarizer_factory`` and ``sentence_tokenizer`` are dependency-injection
    seams used by the unit tests to avoid loading real models/indexes.
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
        image_model_path=None,
        multimodal: bool = False,
        image_extensions=IMAGE_EXTENSIONS,
        summary_name: str = DEFAULT_SUMMARY_NAME,
        summarizer_factory: Optional[Callable[[str, str], object]] = None,
        image_model_factory: Optional[Callable[[str, str], object]] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.extensions = _normalize_extensions(extensions)
        self.chunk_size = chunk_size
        self.top_k = top_k
        self._model_factory = model_factory
        self._index_factory = index_factory
        self._sentence_tokenizer = sentence_tokenizer
        self.image_model_path = Path(image_model_path) if image_model_path else None
        self.multimodal = multimodal
        self.image_extensions = _normalize_extensions(image_extensions)
        self.summary_name = summary_name
        self._summarizer_factory = summarizer_factory
        self._image_model_factory = image_model_factory
        self._model = None
        self._image_model = None
        self._summarizer = None

    # -------------------------------------------------------------- internals

    def _tokenizer(self, text: str) -> list[str]:
        if self._sentence_tokenizer is not None:
            return self._sentence_tokenizer(text)
        return split_sentences(text)

    def _plain_text_embedder(self):
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

    # Backwards-compatible alias (Phase 1 name).
    _embedder = _plain_text_embedder

    def _clip_embedder(self):
        if self._image_model is None:
            if self._image_model_factory is not None:
                self._image_model = self._image_model_factory(
                    str(self.image_model_path), self.device
                )
            else:
                if not self.image_model_path or not is_model_available(self.image_model_path):
                    raise FileNotFoundError(
                        f"Image embedding model not found at '{self.image_model_path}'. "
                        "See README.md for how to store the model."
                    )
                from sentence_transformers import SentenceTransformer

                self._image_model = SentenceTransformer(
                    str(self.image_model_path), device=self.device
                )
        return self._image_model

    def _encode_texts(self, texts) -> np.ndarray:
        embeddings = np.asarray(
            self._plain_text_embedder().encode(list(texts), convert_to_numpy=True),
            dtype=float,
        )
        return normalize_rows(embeddings)

    def _encode_clip_texts(self, texts) -> np.ndarray:
        embeddings = np.asarray(
            self._clip_embedder().encode(list(texts), convert_to_numpy=True), dtype=float
        )
        return normalize_rows(embeddings)

    def _encode_images(self, image_paths) -> np.ndarray:
        from PIL import Image

        images = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (1, 1)))
        embeddings = np.asarray(
            self._clip_embedder().encode(images, convert_to_numpy=True), dtype=float
        )
        return normalize_rows(embeddings)

    def _embed_records_multimodal(self, records) -> np.ndarray:
        if not records:
            return np.zeros((0, 0), dtype=float)
        text_ids = [i for i, record in enumerate(records) if record["kind"] == "text"]
        image_ids = [i for i, record in enumerate(records) if record["kind"] == "image"]
        text_embeddings = (
            self._encode_clip_texts([records[i]["text"] for i in text_ids])
            if text_ids
            else None
        )
        image_embeddings = (
            self._encode_images([records[i]["image_path"] for i in image_ids])
            if image_ids
            else None
        )
        dim = text_embeddings.shape[1] if text_embeddings is not None else image_embeddings.shape[1]
        embeddings = np.zeros((len(records), dim), dtype=float)
        for position, index in enumerate(text_ids):
            embeddings[index] = text_embeddings[position]
        for position, index in enumerate(image_ids):
            embeddings[index] = image_embeddings[position]
        return embeddings

    def _encode_query(self, query: str, mode: str) -> np.ndarray:
        encoder = self._clip_embedder() if mode == "multimodal" else self._plain_text_embedder()
        embeddings = np.asarray(encoder.encode([query], convert_to_numpy=True), dtype=float)
        return normalize_rows(embeddings)[0]

    def _summarizer_instance(self):
        if self._summarizer is None:
            if self._summarizer_factory is not None:
                self._summarizer = self._summarizer_factory(str(self.model_path), self.device)
            else:
                from src.summarizer import TextSummarizer

                self._summarizer = TextSummarizer(self.model_path, device=self.device)
        return self._summarizer

    def _summarize_document(self, document) -> str:
        if document.error:
            return f"ERROR: {document.error}"
        if document.text and document.text.strip():
            try:
                return self._summarizer_instance().summarize(document.text).strip()
            except Exception as exc:
                return f"ERROR: {exc}"
        if document.images:
            return f"Image file '{document.source}' indexed as an image record."
        return "No extractable text content."

    def _append_summaries(self, output: Path, documents, n_text: int, n_image: int) -> Path:
        summary_path = output.parent / self.summary_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            f"## Corpus build {timestamp} — {output.name}",
            "",
            f"- Files: {len(documents)}  |  text records: {n_text}  |  image records: {n_image}",
            "",
        ]
        for document in documents:
            lines.append(f"### {document.source}")
            lines.append(self._summarize_document(document))
            lines.append("")
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return summary_path

    def _build_records(self, documents) -> list[dict]:
        records: list[dict] = []
        for document in documents:
            if document.error:
                continue
            if document.kind == "text" and document.text and document.text.strip():
                for passage in build_passages([document.text], self.chunk_size, self._tokenizer):
                    records.append(
                        {
                            "kind": "text",
                            "source": document.source,
                            "text": passage,
                            "page": document.pages,
                        }
                    )
            if self.multimodal:
                for image_path in document.images:
                    records.append(
                        {
                            "kind": "image",
                            "source": document.source,
                            "text": document.source,
                            "image_path": image_path,
                        }
                    )
        return records

    # ----------------------------------------------------------------- public

    def store(self, directory: str, output_file: str, overwrite: bool = False) -> str:
        """Build a corpus from ``directory`` and persist it to ``output_file``."""
        output = Path(output_file).resolve()
        if output.exists() and not overwrite:
            return "File already exists."

        image_dir = output.parent / f"{output.stem}_images"
        documents = iter_documents(directory, self.extensions, image_dir=image_dir)
        if not documents:
            return f"No supported files found in '{directory}'"

        records = self._build_records(documents)
        if not records:
            return "No passages extracted from the input files."

        if self.multimodal:
            embeddings = self._embed_records_multimodal(records)
        else:
            embeddings = self._encode_texts([record["text"] for record in records])

        payload = {
            "version": 2,
            "mode": "multimodal" if self.multimodal else "text",
            "text_model_path": str(self.model_path),
            "image_model_path": str(self.image_model_path) if self.image_model_path else "",
            "records": records,
            "embeddings": embeddings,
        }
        save_corpus(output, payload)

        n_text = sum(1 for record in records if record["kind"] == "text")
        n_image = sum(1 for record in records if record["kind"] == "image")
        summary_path = self._append_summaries(output, documents, n_text, n_image)
        return (
            f"Corpus with {n_text} text / {n_image} image records saved to '{output}'; "
            f"summaries appended to '{summary_path}'"
        )

    def query(self, corpus_file: str, query: str, top_k: Optional[int] = None) -> str:
        """Return the ``top_k`` records most relevant to ``query``."""
        query = (query or "").strip()
        if not query:
            return "No query provided."

        path = Path(corpus_file).resolve()
        if not path.exists():
            return f"Corpus not found at '{path}'. Build it first with store_corpus."

        payload = load_corpus(path)
        if "records" in payload:
            return self._query_v2(path, payload, query, top_k)
        return self._query_v1(path, payload, query, top_k)

    # --------------------------------------------------------------- query impl

    def _rank(self, embeddings: np.ndarray, query_vec: np.ndarray, top_k, total: int):
        index = (self._index_factory or _default_index_factory)(embeddings)
        k = int(top_k) if top_k else int(self.top_k)
        k = max(1, min(k, total))
        scores, ids = index.search(query_vec.reshape(1, -1), k)
        return k, np.ravel(scores), np.ravel(ids)

    def _query_v2(self, path, payload, query, top_k) -> str:
        records = list(payload.get("records") or [])
        if not records:
            return "Corpus is empty."
        embeddings = np.asarray(payload.get("embeddings"), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            return f"Invalid corpus file: '{path}'."

        mode = payload.get("mode", "text")
        query_vec = self._encode_query(query, mode)
        if query_vec.shape[0] != embeddings.shape[1]:
            return (
                "Query embedding size does not match the corpus; "
                "rebuild the corpus with the configured model."
            )

        k, scores, ids = self._rank(embeddings, query_vec, top_k, len(records))
        lines = [f"Top {k} matches for '{query}':"]
        rank = 0
        for score, idx in zip(scores, ids):
            idx = int(idx)
            if idx < 0:
                continue
            record = records[idx]
            source = record.get("source", "")
            rank += 1
            hit = f"{rank}) [{float(score):.3f}] ({source}) {record.get('text') or source}"
            if record.get("kind") == "image":
                hit += f" [image] {record.get('image_path', '')}".rstrip()
            lines.append(hit)
        return "\n".join(lines)

    def _query_v1(self, path, payload, query, top_k) -> str:
        passages = list(payload.get("passages") or [])
        if not passages:
            return "Corpus is empty."
        embeddings = np.asarray(payload.get("embeddings"), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(passages):
            return f"Invalid corpus file: '{path}'."

        query_vec = self._encode_texts([query])[0]
        if query_vec.shape[0] != embeddings.shape[1]:
            return (
                "Query embedding size does not match the corpus; "
                "rebuild the corpus with the configured model."
            )

        k, scores, ids = self._rank(embeddings, query_vec, top_k, len(passages))
        lines = [f"Top {k} matches for '{query}':"]
        rank = 0
        for score, idx in zip(scores, ids):
            idx = int(idx)
            if idx < 0:
                continue
            rank += 1
            lines.append(f"{rank}) [{float(score):.3f}] {passages[idx]}")
        return "\n".join(lines)
