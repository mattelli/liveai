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


class FakeEmbedder:
    """Deterministic text -> vector embedder with an optional explicit mapping."""

    def __init__(self, mapping=None, dim=4):
        self.mapping = mapping or {}
        self.dim = dim
        self.calls = 0

    def _fallback(self, text):
        vec = np.zeros(self.dim)
        vec[sum(map(ord, text)) % self.dim] = 1.0
        return vec

    def encode(self, texts, convert_to_numpy=True):
        self.calls += 1
        return np.array(
            [self.mapping.get(text, self._fallback(text)) for text in texts],
            dtype=float,
        )


class FakeIndex:
    """Exact inner-product search, mimicking ``faiss.IndexFlatIP.search``."""

    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=float)

    def search(self, query_vec, k):
        scores = self.embeddings @ np.ravel(query_vec)
        order = np.argsort(-scores)[:k]
        return scores[order].reshape(1, -1), order.reshape(1, -1)


def make_corpus(passages, vectors, *, chunk_size=1, **kwargs):
    embedder = FakeEmbedder(mapping=dict(zip(passages, vectors)), dim=len(vectors[0]))
    corpus = EmbeddingCorpus(
        "model",
        chunk_size=chunk_size,
        model_factory=lambda path, device: embedder,
        index_factory=lambda embeddings: FakeIndex(embeddings),
        sentence_tokenizer=lambda text: list(passages),
        **kwargs,
    )
    return corpus, embedder


def write_doc(tmp_path, name="doc.txt", content="ignored", subdir="docs"):
    directory = tmp_path / subdir
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(content, encoding="utf-8")
    return directory


# ------------------------------------------------------------- lazy imports


def test_import_is_lazy():
    for name in ("faiss", "sentence_transformers", "torch"):
        sys.modules.pop(name, None)
    sys.modules.pop("src.embedding_search", None)
    importlib.import_module("src.embedding_search")
    for name in ("faiss", "sentence_transformers", "torch"):
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
    assert collect_texts(directory) == ["alpha", "beta"]


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
    assert build_passages(["doc"], chunk_size=2, tokenizer=tokenizer) == [
        "s1 s2",
        "s3 s4",
    ]


def test_build_passages_size_one():
    tokenizer = lambda text: ["s1", "s2", "s3"]  # noqa: E731
    assert build_passages(["doc"], chunk_size=1, tokenizer=tokenizer) == ["s1", "s2", "s3"]


# ------------------------------------------------------------------- store


def test_store_writes_pickle(tmp_path):
    directory = write_doc(tmp_path)
    passages = ["a.", "b.", "c."]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corpus, _ = make_corpus(passages, vectors)
    output = tmp_path / "corpus.pkl"

    status = corpus.store(str(directory), str(output))
    assert "3 passages" in status and output.exists()

    payload = load_corpus(output)
    assert payload["passages"] == passages
    embeddings = np.asarray(payload["embeddings"])
    assert embeddings.shape == (3, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_store_refuses_existing_by_default(tmp_path):
    directory = write_doc(tmp_path)
    output = tmp_path / "corpus.pkl"
    output.write_bytes(b"sentinel")
    corpus, _ = make_corpus(["a."], [[1, 0, 0, 0]])
    assert corpus.store(str(directory), str(output)) == "File already exists."
    assert output.read_bytes() == b"sentinel"


def test_store_overwrite_true(tmp_path):
    directory = write_doc(tmp_path)
    output = tmp_path / "corpus.pkl"
    output.write_bytes(b"sentinel")
    corpus, _ = make_corpus(["a."], [[1, 0, 0, 0]])
    status = corpus.store(str(directory), str(output), overwrite=True)
    assert "1 passages" in status
    assert load_corpus(output)["passages"] == ["a."]


def test_store_no_text_files_returns_message(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    corpus, _ = make_corpus(["a."], [[1, 0, 0, 0]])
    assert "No text files" in corpus.store(str(directory), str(tmp_path / "c.pkl"))


def test_store_missing_model_raises_friendly_error(tmp_path):
    directory = write_doc(tmp_path)
    corpus = EmbeddingCorpus(
        tmp_path / "missing-model",
        sentence_tokenizer=lambda text: ["a.", "b."],
    )
    with pytest.raises(FileNotFoundError) as exc:
        corpus.store(str(directory), str(tmp_path / "c.pkl"))
    assert "missing-model" in str(exc.value)


# ------------------------------------------------------------------- query


def test_query_returns_ranked_hits(tmp_path):
    passages = ["alpha.", "beta.", "gamma."]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corpus, embedder = make_corpus(passages, vectors)
    embedder.mapping["find beta"] = [0, 1, 0]  # query closest to "beta."
    output = tmp_path / "corpus.pkl"
    corpus.store(str(write_doc(tmp_path)), str(output))

    result = corpus.query(str(output), "find beta")
    lines = result.splitlines()
    assert lines[0].startswith("Top 3 matches")
    assert lines[1].startswith("1) [1.000]") and "beta." in lines[1]


def test_query_missing_corpus_returns_message(tmp_path):
    corpus, _ = make_corpus(["a."], [[1, 0, 0, 0]])
    assert "Corpus not found" in corpus.query(str(tmp_path / "nope.pkl"), "hi")


def test_query_empty_query_returns_message(tmp_path):
    corpus, _ = make_corpus(["a."], [[1, 0, 0, 0]])
    assert corpus.query(str(tmp_path / "c.pkl"), "   ") == "No query provided."


def test_query_dimension_mismatch_returns_message(tmp_path):
    output = tmp_path / "corpus.pkl"
    save_corpus(output, {"passages": ["a."], "embeddings": np.ones((1, 3))})
    corpus = EmbeddingCorpus(
        "m",
        model_factory=lambda path, device: FakeEmbedder(dim=4),
        index_factory=lambda embeddings: FakeIndex(embeddings),
    )
    assert "does not match the corpus" in corpus.query(str(output), "hi")


def test_model_loaded_once_across_store_and_query(tmp_path):
    counter = {"n": 0}

    def factory(path, device):
        counter["n"] += 1
        return FakeEmbedder(mapping={"a.": [1, 0, 0, 0], "hi": [1, 0, 0, 0]})

    passages = ["a."]
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
    passages = ["a.", "b.", "c."]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corpus, embedder = make_corpus(passages, vectors, top_k=2)
    embedder.mapping["q"] = [1, 0, 0]
    output = tmp_path / "corpus.pkl"
    corpus.store(str(write_doc(tmp_path)), str(output))

    result = corpus.query(str(output), "q")
    assert result.splitlines()[0].startswith("Top 2 matches")
    assert len(result.splitlines()) == 3  # header + 2 hits
