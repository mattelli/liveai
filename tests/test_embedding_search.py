import importlib
import sys

import numpy as np
import pytest

import src.embedding_search as emb
from src.embedding_search import (
    EmbeddingCorpus,
    build_passages,
    collect_texts,
    load_corpus,
    normalize_rows,
    save_corpus,
)


class FakeModel:
    """Deterministic text/image embedder (handles str and PIL-like images)."""

    def __init__(self, mapping=None, dim=4):
        self.mapping = mapping or {}
        self.dim = dim
        self.text_calls = 0
        self.image_calls = 0

    def _fallback(self, text):
        vec = np.zeros(self.dim)
        vec[sum(map(ord, text)) % self.dim] = 1.0
        return vec

    def _image_vector(self, image):
        vec = np.zeros(self.dim)
        vec[sum(image.size) % self.dim] = 1.0
        return vec

    def encode(self, items, convert_to_numpy=True):
        vectors = []
        for item in items:
            if isinstance(item, str):
                self.text_calls += 1
                vectors.append(self.mapping.get(item, self._fallback(item)))
            else:
                self.image_calls += 1
                vectors.append(self._image_vector(item))
        return np.array(vectors, dtype=float)


class FakeIndex:
    """Exact inner-product search, mimicking ``faiss.IndexFlatIP.search``."""

    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=float)

    def search(self, query_vec, k):
        scores = self.embeddings @ np.ravel(query_vec)
        order = np.argsort(-scores)[:k]
        return scores[order].reshape(1, -1), order.reshape(1, -1)


def make_corpus(passages, vectors, *, chunk_size=1, multimodal=False, **kwargs):
    model = FakeModel(mapping=dict(zip(passages, vectors)), dim=len(vectors[0]))
    corpus = EmbeddingCorpus(
        "model",
        chunk_size=chunk_size,
        model_factory=lambda path, device: model,
        image_model_factory=lambda path, device: model,
        index_factory=lambda embeddings: FakeIndex(embeddings),
        sentence_tokenizer=lambda text: list(passages),
        multimodal=multimodal,
        image_model_path="clip",
        **kwargs,
    )
    return corpus, model


def write_doc(tmp_path, name="doc.txt", content="ignored", subdir="docs"):
    directory = tmp_path / subdir
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(content, encoding="utf-8")
    return directory


def write_png(path):
    from PIL import Image

    Image.new("RGB", (2, 2), color="blue").save(path)


# ------------------------------------------------------------- lazy imports


def test_import_is_lazy():
    for name in ("faiss", "sentence_transformers", "torch", "pymupdf", "docx", "openpyxl"):
        sys.modules.pop(name, None)
    sys.modules.pop("src.embedding_search", None)
    importlib.import_module("src.embedding_search")
    for name in ("faiss", "sentence_transformers", "torch", "pymupdf", "docx", "openpyxl"):
        assert name not in sys.modules, f"{name} imported at module load"


# ----------------------------------------------------------------- helpers


def test_normalize_rows_unit_norm_and_zero_safe():
    result = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(result[0]), 1.0)
    assert np.allclose(result[1], [0.0, 0.0])


def test_collect_texts_reads_supported_files(tmp_path):
    directory = write_doc(tmp_path, "a.txt", "alpha")
    (directory / "b.md").write_text("beta", encoding="utf-8")
    (directory / "c.bin").write_text("nope", encoding="utf-8")
    assert collect_texts(directory, extensions=(".txt", ".md")) == ["alpha", "beta"]


def test_collect_texts_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_texts(tmp_path / "nope")


def test_collect_texts_not_a_dir_raises(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        collect_texts(target)


def test_build_passages_chunks():
    tokenizer = lambda text: ["s1", "s2", "s3", "s4"]  # noqa: E731
    assert build_passages(["doc"], chunk_size=2, tokenizer=tokenizer) == ["s1 s2", "s3 s4"]


def test_build_passages_size_one():
    tokenizer = lambda text: ["s1", "s2", "s3"]  # noqa: E731
    assert build_passages(["doc"], chunk_size=1, tokenizer=tokenizer) == ["s1", "s2", "s3"]


# ------------------------------------------------------------------- store


def test_store_writes_records(tmp_path):
    directory = write_doc(tmp_path)
    passages = ["alpha.", "beta.", "gamma."]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corpus, _ = make_corpus(passages, vectors)
    output = tmp_path / "corpus.pkl"

    status = corpus.store(str(directory), str(output))
    assert "3 text" in status and output.exists()

    payload = load_corpus(output)
    assert payload["version"] == 2
    assert [r["text"] for r in payload["records"]] == passages
    embeddings = np.asarray(payload["embeddings"])
    assert embeddings.shape == (3, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_store_refuses_existing_by_default(tmp_path):
    directory = write_doc(tmp_path)
    output = tmp_path / "corpus.pkl"
    output.write_bytes(b"sentinel")
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    assert corpus.store(str(directory), str(output)) == "File already exists."
    assert output.read_bytes() == b"sentinel"


def test_store_overwrite_true(tmp_path):
    directory = write_doc(tmp_path)
    output = tmp_path / "corpus.pkl"
    output.write_bytes(b"sentinel")
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    status = corpus.store(str(directory), str(output), overwrite=True)
    assert "1 text" in status
    assert [r["text"] for r in load_corpus(output)["records"]] == ["alpha."]


def test_store_no_supported_files_returns_message(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    assert "No supported files" in corpus.store(str(directory), str(tmp_path / "c.pkl"))


def test_store_missing_model_raises_friendly_error(tmp_path):
    directory = write_doc(tmp_path)
    corpus = EmbeddingCorpus(
        tmp_path / "missing-model",
        sentence_tokenizer=lambda text: ["alpha.", "beta."],
    )
    with pytest.raises(FileNotFoundError) as exc:
        corpus.store(str(directory), str(tmp_path / "c.pkl"))
    assert "missing-model" in str(exc.value)


def test_store_v2_writes_records_not_processed_list(tmp_path):
    directory = write_doc(tmp_path)
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    output = tmp_path / "corpus.pkl"
    corpus.store(str(directory), str(output))

    payload = load_corpus(output)
    assert payload["version"] == 2 and "records" in payload
    assert "processed" not in payload and "summary" not in payload


def test_store_appends_processed_md(tmp_path):
    directory = write_doc(tmp_path)
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    corpus.store(str(directory), str(tmp_path / "corpus1.pkl"))
    corpus.store(str(directory), str(tmp_path / "corpus2.pkl"))

    processed_path = tmp_path / "PROCESSED.md"
    text = processed_path.read_text(encoding="utf-8")
    assert text.count("## Corpus build") == 2
    assert str((directory / "doc.txt").resolve()) in text


def test_store_image_records_in_multimodal(tmp_path):
    directory = write_doc(tmp_path)
    write_png(directory / "pic.png")
    passages = ["alpha.", "beta."]
    vectors = [[1, 0, 0], [0, 1, 0]]
    corpus, model = make_corpus(passages, vectors, multimodal=True)
    output = tmp_path / "corpus.pkl"

    status = corpus.store(str(directory), str(output))
    assert "1 image" in status
    payload = load_corpus(output)
    kinds = [r["kind"] for r in payload["records"]]
    assert "image" in kinds
    assert model.image_calls >= 1
    embeddings = np.asarray(payload["embeddings"])
    assert embeddings.shape[0] == len(payload["records"])
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_store_text_mode_skips_images(tmp_path):
    directory = write_doc(tmp_path)
    write_png(directory / "pic.png")
    corpus, model = make_corpus(["alpha.", "beta."], [[1, 0, 0], [0, 1, 0]], multimodal=False)
    output = tmp_path / "corpus.pkl"

    corpus.store(str(directory), str(output))
    payload = load_corpus(output)
    assert all(r["kind"] == "text" for r in payload["records"])
    assert model.image_calls == 0


def test_missing_image_model_raises_friendly_error(tmp_path):
    directory = write_doc(tmp_path)
    corpus = EmbeddingCorpus(
        "model",
        multimodal=True,
        image_model_path=tmp_path / "missing-clip",
        sentence_tokenizer=lambda text: ["alpha.", "beta."],
    )
    with pytest.raises(FileNotFoundError) as exc:
        corpus.store(str(directory), str(tmp_path / "c.pkl"))
    assert "missing-clip" in str(exc.value)


# ------------------------------------------------------------------- query


def test_query_returns_ranked_hits(tmp_path):
    passages = ["alpha.", "beta.", "gamma."]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corpus, model = make_corpus(passages, vectors)
    model.mapping["find beta"] = [0, 1, 0]
    output = tmp_path / "corpus.pkl"
    corpus.store(str(write_doc(tmp_path)), str(output))

    result = corpus.query(str(output), "find beta")
    lines = result.splitlines()
    assert lines[0].startswith("Top 3 matches")
    assert lines[1].startswith("1) [1.000]") and "beta." in lines[1]


def test_query_v2_formats_sources(tmp_path):
    directory = write_doc(tmp_path)
    write_png(directory / "pic.png")
    passages = ["alpha.", "beta."]
    vectors = [[1, 0, 0], [0, 1, 0]]
    corpus, model = make_corpus(passages, vectors, multimodal=True)
    model.mapping["query"] = [0, 1, 0]
    output = tmp_path / "corpus.pkl"
    corpus.store(str(directory), str(output))

    result = corpus.query(str(output), "query", top_k=10)
    assert "(doc.txt)" in result
    assert "[image]" in result


def test_query_reads_v1_corpus(tmp_path):
    passages = ["alpha.", "beta."]
    vectors = [[1, 0, 0], [0, 1, 0]]
    corpus, model = make_corpus(passages, vectors)
    model.mapping["find beta"] = [0, 1, 0]
    output = tmp_path / "old.pkl"
    save_corpus(output, {"passages": passages, "embeddings": normalize_rows(np.array(vectors, dtype=float))})

    result = corpus.query(str(output), "find beta")
    assert "beta." in result and "(" not in result.splitlines()[1]


def test_query_missing_corpus_returns_message(tmp_path):
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    assert "Corpus not found" in corpus.query(str(tmp_path / "nope.pkl"), "hi")


def test_query_empty_query_returns_message(tmp_path):
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    assert corpus.query(str(tmp_path / "c.pkl"), "   ") == "No query provided."


def test_query_dimension_mismatch_returns_message(tmp_path):
    output = tmp_path / "corpus.pkl"
    save_corpus(output, {"passages": ["alpha."], "embeddings": np.ones((1, 3))})
    corpus = EmbeddingCorpus(
        "m",
        model_factory=lambda path, device: FakeModel(dim=4),
        index_factory=lambda embeddings: FakeIndex(embeddings),
    )
    assert "does not match the corpus" in corpus.query(str(output), "hi")


def test_model_loaded_once_across_store_and_query(tmp_path):
    counter = {"n": 0}

    def factory(path, device):
        counter["n"] += 1
        return FakeModel(mapping={"alpha.": [1, 0, 0], "hi": [1, 0, 0]})

    passages = ["alpha."]
    corpus = EmbeddingCorpus(
        "m",
        model_factory=factory,
        index_factory=lambda embeddings: FakeIndex(embeddings),
        sentence_tokenizer=lambda text: list(passages),
    )
    output = tmp_path / "corpus.pkl"
    corpus.store(str(write_doc(tmp_path)), str(output))
    corpus.query(str(output), "hi")
    assert counter["n"] == 1


def test_query_uses_instance_top_k_when_not_overridden(tmp_path):
    passages = ["alpha.", "beta.", "gamma."]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corpus, model = make_corpus(passages, vectors, top_k=2)
    model.mapping["q"] = [1, 0, 0]
    output = tmp_path / "corpus.pkl"
    corpus.store(str(write_doc(tmp_path)), str(output))

    result = corpus.query(str(output), "q")
    assert result.splitlines()[0].startswith("Top 2 matches")
    assert len(result.splitlines()) == 3  # header + 2 hits


# ------------------------------------------------- content filter / page numbers


def test_has_content_filters_junk():
    assert emb.has_content("alpha") is True
    assert emb.has_content(". . .") is False
    assert emb.has_content("a.") is False
    assert emb.has_content("12") is False
    assert emb.has_content("1234") is True


def test_store_drops_low_content_passages(tmp_path):
    directory = write_doc(tmp_path)
    passages = [". . .", "Real content about qubits."]
    vectors = [[1, 0, 0], [0, 1, 0]]
    corpus, _ = make_corpus(passages, vectors)
    output = tmp_path / "corpus.pkl"

    status = corpus.store(str(directory), str(output))
    payload = load_corpus(output)
    assert [r["text"] for r in payload["records"]] == ["Real content about qubits."]
    assert "1 text" in status


def test_store_uses_pdf_section_pages(tmp_path, monkeypatch):
    from src.document_loader import Document

    documents = [
        Document(
            source="doc.pdf",
            kind="text",
            text="alpha.\nbeta.",
            pages=10,
            sections=[(3, "alpha."), (7, "beta.")],
        )
    ]
    monkeypatch.setattr(
        emb, "iter_documents", lambda directory, extensions, image_dir=None: documents
    )

    model = FakeModel(mapping={"alpha.": [1, 0, 0], "beta.": [0, 1, 0]}, dim=3)
    corpus = EmbeddingCorpus(
        "m",
        model_factory=lambda path, device: model,
        index_factory=lambda embeddings: FakeIndex(embeddings),
        sentence_tokenizer=lambda text: [text],
    )
    output = tmp_path / "corpus.pkl"
    corpus.store(str(tmp_path), str(output))

    payload = load_corpus(output)
    assert [r["text"] for r in payload["records"]] == ["alpha.", "beta."]
    assert [r["page"] for r in payload["records"]] == [3, 7]


# --------------------------------------------------------------- kind filter


def _multimodal_corpus(tmp_path):
    directory = write_doc(tmp_path)
    write_png(directory / "pic.png")
    passages = ["alpha.", "beta."]
    vectors = [[1, 0, 0], [0, 1, 0]]
    corpus, model = make_corpus(passages, vectors, multimodal=True, top_k=10)
    model.mapping["query"] = [0, 1, 0]
    output = tmp_path / "corpus.pkl"
    corpus.store(str(directory), str(output))
    return corpus, output


def test_query_image_kind_returns_only_images(tmp_path):
    corpus, output = _multimodal_corpus(tmp_path)
    result = corpus.query(str(output), "query", kind="image")
    assert "image matches" in result.splitlines()[0]
    assert "[image]" in result
    assert "(doc.txt)" not in result


def test_query_text_kind_returns_only_text(tmp_path):
    corpus, output = _multimodal_corpus(tmp_path)
    result = corpus.query(str(output), "query", kind="text")
    assert "text matches" in result.splitlines()[0]
    assert "(doc.txt)" in result
    assert "[image]" not in result


def test_query_kind_is_case_insensitive(tmp_path):
    corpus, output = _multimodal_corpus(tmp_path)
    result = corpus.query(str(output), "query", kind="IMAGE")
    assert "[image]" in result


def test_query_invalid_kind_returns_message(tmp_path):
    corpus, output = _multimodal_corpus(tmp_path)
    result = corpus.query(str(output), "query", kind="audio")
    assert "Invalid kind" in result and "text" in result and "image" in result


def test_query_image_kind_on_v1_corpus(tmp_path):
    output = tmp_path / "old.pkl"
    save_corpus(output, {"passages": ["alpha."], "embeddings": np.eye(1, 3)})
    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]])
    assert corpus.query(str(output), "alpha", kind="image") == "No image records found in the corpus."


# ----------------------------------------------------- image quality filters


def test_store_dedupes_identical_images(tmp_path):
    directory = write_doc(tmp_path)
    write_png(directory / "a.png")
    write_png(directory / "b.png")  # identical bytes to a.png
    from PIL import Image

    Image.new("RGB", (2, 2), color="red").save(directory / "c.png")

    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]], multimodal=True)
    output = tmp_path / "corpus.pkl"
    corpus.store(str(directory), str(output))

    images = [r for r in load_corpus(output)["records"] if r["kind"] == "image"]
    assert len(images) == 2  # a/b collapse into one; c is distinct


def test_store_clears_stale_image_dir(tmp_path):
    stale = tmp_path / "corpus_images"
    stale.mkdir()
    (stale / "junk.png").write_bytes(b"old")

    corpus, _ = make_corpus(["alpha."], [[1, 0, 0]], multimodal=True)
    corpus.store(str(write_doc(tmp_path)), str(tmp_path / "corpus.pkl"))

    assert not (stale / "junk.png").exists()


def test_prune_orphan_images(tmp_path):
    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    kept = image_dir / "kept.png"
    orphan = image_dir / "orphan.png"
    write_png(kept)
    write_png(orphan)

    removed = emb.prune_orphan_images(image_dir, [{"kind": "image", "image_path": str(kept)}])

    assert removed == 1
    assert kept.exists() and not orphan.exists()
