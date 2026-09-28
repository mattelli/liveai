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

Per-file summaries were replaced by ``PROCESSED.md`` next to the corpus pickle:
a build header followed by the absolute path of every processed document. The
list is deliberately **not** stored in the pickle.

Heavy dependencies (``faiss``, ``sentence_transformers``, ``PIL``) are imported
lazily so this module can be imported -- and unit tested -- without loading a
real model or building a real index.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
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
    "DEFAULT_CORPORA_DIR",
    "CORPUS_FILE",
    "METADATA_FILE",
    "PROCESSED_FILE",
    "IMAGES_DIR",
    "MIN_ALNUM_CHARS",
    "sanitize_corpus_name",
    "resolve_corpus_dir",
    "save_metadata",
    "load_metadata",
    "list_corpora",
    "collect_texts",
    "normalize_rows",
    "build_passages",
    "has_content",
    "load_corpus",
    "save_corpus",
    "EmbeddingCorpus",
]

DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 3
DEFAULT_EXTENSIONS = SUPPORTED_EXTENSIONS
DEFAULT_CORPORA_DIR = ".res/corpora"
CORPUS_FILE = "corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_FILE = "processed.md"
IMAGES_DIR = "images"
MIN_ALNUM_CHARS = 4  # drop passages with fewer letters/digits (TOC leaders, noise)

_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
_CORPUS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def sanitize_corpus_name(name: str) -> str:
    """Validate a corpus name; raise ``ValueError`` for empty/path-like names."""
    cleaned = (name or "").strip()
    if not cleaned or cleaned in (".", "..") or not _CORPUS_NAME_RE.match(cleaned):
        raise ValueError(f"invalid corpus name: {name!r}")
    return cleaned


def resolve_corpus_dir(corpus_name: str, corpora_dir) -> Path:
    """Return the folder for ``corpus_name`` under ``corpora_dir``."""
    return Path(corpora_dir) / sanitize_corpus_name(corpus_name)


def save_metadata(corpus_dir, metadata: dict) -> None:
    """Write ``metadata.json`` inside ``corpus_dir``."""
    path = Path(corpus_dir) / METADATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(corpus_dir) -> dict:
    """Read ``metadata.json`` from ``corpus_dir``."""
    path = Path(corpus_dir) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Corpus metadata not found: '{path}'")
    return json.loads(path.read_text(encoding="utf-8"))


def list_corpora(corpora_dir) -> list[dict]:
    """Return every ``<corpora_dir>/*/metadata.json`` entry, sorted by name."""
    root = Path(corpora_dir)
    if not root.is_dir():
        return []
    entries: list[dict] = []
    for metadata_file in root.glob(f"*/{METADATA_FILE}"):
        try:
            entries.append(json.loads(metadata_file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(entries, key=lambda entry: entry.get("name", ""))


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


def has_content(text, min_alnum: int = MIN_ALNUM_CHARS) -> bool:
    """True when ``text`` has at least ``min_alnum`` letters/digits.

    Filters out junk passages such as TOC dot leaders (``. . .``), punctuation
    runs and near-empty table cells.
    """
    return len(_ALNUM.findall(text or "")) >= max(1, int(min_alnum))


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


def prune_orphan_images(image_dir, records) -> int:
    """Delete image files under ``image_dir`` that no record references.

    Deduplication keeps only the first copy of identical images; this removes the
    leftover files so the image folder matches the corpus exactly.
    """
    directory = Path(image_dir)
    if not directory.exists():
        return 0
    referenced = {
        Path(record["image_path"]).resolve()
        for record in records
        if record.get("image_path")
    }
    removed = 0
    for path in directory.iterdir():
        if path.is_file() and path.resolve() not in referenced:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


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
        self._image_model_factory = image_model_factory
        self._model = None
        self._image_model = None

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

    def _append_processed(self, corpus_dir, documents, directory, n_text: int, n_image: int) -> Path:
        corpus_dir = Path(corpus_dir)
        processed_path = corpus_dir / PROCESSED_FILE
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        root = Path(directory).resolve()
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            f"## Corpus build {timestamp} — {corpus_dir.name}",
            "",
            f"- Files: {len(documents)}  |  text records: {n_text}  |  image records: {n_image}",
            "",
        ]
        for document in documents:
            absolute = (root / document.source).resolve()
            suffix = f" — ERROR: {document.error}" if document.error else ""
            lines.append(f"- {absolute}{suffix}")
        lines.append("")
        with open(processed_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return processed_path

    def _build_records(self, documents) -> list[dict]:
        records: list[dict] = []
        seen_image_hashes: set = set()
        for document in documents:
            if document.error:
                continue
            if document.kind == "text":
                # Paginated documents (PDF) expose per-page sections so each
                # passage keeps its real page number.
                sections = document.sections or [(document.pages, document.text)]
                for page, section_text in sections:
                    for passage in build_passages([section_text], self.chunk_size, self._tokenizer):
                        if not has_content(passage):
                            continue
                        records.append(
                            {
                                "kind": "text",
                                "source": document.source,
                                "text": passage,
                                "page": page,
                            }
                        )
            if self.multimodal:
                for image_path in document.images:
                    try:
                        digest = hashlib.md5(Path(image_path).read_bytes()).hexdigest()
                    except OSError:
                        continue
                    if digest in seen_image_hashes:
                        continue
                    seen_image_hashes.add(digest)
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

    def store(self, directory: str, corpus_dir, overwrite: bool = False) -> str:
        """Build a named corpus folder from ``directory``."""
        corpus_dir = Path(corpus_dir)
        pkl_path = corpus_dir / CORPUS_FILE
        if pkl_path.exists() and not overwrite:
            return "Corpus already exists."

        image_dir = corpus_dir / IMAGES_DIR
        if image_dir.exists():
            shutil.rmtree(image_dir, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        documents = iter_documents(directory, self.extensions, image_dir=image_dir)
        if not documents:
            return f"No supported content found in '{directory}'."

        records = self._build_records(documents)
        if not records:
            return f"No supported content found in '{directory}'."

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
        save_corpus(pkl_path, payload)
        prune_orphan_images(image_dir, records)

        n_text = sum(1 for record in records if record["kind"] == "text")
        n_image = sum(1 for record in records if record["kind"] == "image")
        self._append_processed(corpus_dir, documents, directory, n_text, n_image)
        save_metadata(corpus_dir, {
            "name": corpus_dir.name,
            "version": 2,
            "mode": payload["mode"],
            "text_model_path": payload["text_model_path"],
            "image_model_path": payload["image_model_path"],
            "source_dir": str(Path(directory).resolve()),
            "records": len(records),
            "text_records": n_text,
            "image_records": n_image,
            "sources": len({record["source"] for record in records if record.get("source")}),
            "built_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
        return (
            f"Corpus '{corpus_dir.name}' saved to '{corpus_dir}' "
            f"({n_text} text / {n_image} image records)."
        )

    def query(self, corpus_dir, query: str, top_k: Optional[int] = None, kind: Optional[str] = None) -> str:
        """Return the ``top_k`` records most relevant to ``query``.

        ``kind`` restricts the ranking to ``"text"`` or ``"image"`` records;
        ``None``/empty searches all records.
        """
        query = (query or "").strip()
        if not query:
            return "No query provided."

        kind = (kind or "").strip().lower() or None
        if kind not in (None, "text", "image"):
            return f"Invalid kind '{kind}'. Use 'text' or 'image'."

        path = Path(corpus_dir) / CORPUS_FILE
        if not path.exists():
            return (
                f"Corpus not found at '{path}'. Use list_corpora() for available "
                "names, or store_corpus(...) to build it."
            )

        payload = load_corpus(path)
        if "records" in payload:
            return self._query_v2(path, payload, query, top_k, kind)
        return self._query_v1(path, payload, query, top_k, kind)

    # --------------------------------------------------------------- query impl

    def _rank(self, embeddings: np.ndarray, query_vec: np.ndarray, top_k, total: int):
        index = (self._index_factory or _default_index_factory)(embeddings)
        k = int(top_k) if top_k else int(self.top_k)
        k = max(1, min(k, total))
        scores, ids = index.search(query_vec.reshape(1, -1), k)
        return k, np.ravel(scores), np.ravel(ids)

    def _query_v2(self, path, payload, query, top_k, kind=None) -> str:
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

        if kind:
            candidates = [i for i, record in enumerate(records) if record.get("kind", "text") == kind]
        else:
            candidates = list(range(len(records)))
        if not candidates:
            return f"No {kind} records found in the corpus."

        k, scores, ids = self._rank(embeddings[candidates], query_vec, top_k, len(candidates))
        label = f"{kind} " if kind else ""
        lines = [f"Top {k} {label}matches for '{query}':"]
        rank = 0
        for score, local in zip(scores, ids):
            local = int(local)
            if local < 0:
                continue
            record = records[candidates[local]]
            source = record.get("source", "")
            rank += 1
            hit = f"{rank}) [{float(score):.3f}] ({source}) {record.get('text') or source}"
            if record.get("kind") == "image":
                hit += f" [image] {record.get('image_path', '')}".rstrip()
            lines.append(hit)
        return "\n".join(lines)

    def _query_v1(self, path, payload, query, top_k, kind=None) -> str:
        if kind == "image":
            return "No image records found in the corpus."
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
