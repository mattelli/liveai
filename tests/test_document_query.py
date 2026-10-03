import numpy as np
import pytest

import src.document_query as document_query
from src.document_query import DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, DocumentQuery
from src.document_loader import Document


class FakeCrossEncoder:
    """Scores every [query, passage] pair with a deterministic value."""

    def __init__(self, scores=None):
        self.calls = 0
        self.seen_inputs = None
        self._scores = scores

    def predict(self, inputs):
        self.calls += 1
        self.seen_inputs = list(inputs)
        if self._scores is not None:
            return np.asarray(self._scores, dtype=float)
        return np.arange(len(inputs), dtype=float)


def make_query(scores=None, sentences=None, **kwargs):
    """Build a DocumentQuery with a fake encoder and a fixed tokenizer.

    ``sentences`` is what the tokenizer returns for every call (independent of
    the input text). ``kwargs.setdefault("chunk_size", 1)`` keeps one passage
    per sentence so the passage count is predictable.
    """
    kwargs.setdefault("chunk_size", 1)
    fake = FakeCrossEncoder(scores=scores)
    instance = DocumentQuery(
        "model",
        cross_encoder_factory=lambda path, device: fake,
        sentence_tokenizer=lambda text: list(sentences or []),
        **kwargs,
    )
    return instance, fake


def make_text_file(tmp_path, name="doc.txt"):
    target = tmp_path / name
    target.write_text("some document body", encoding="utf-8")
    return target


# ------------------------------------------------------------------- ranking


def test_query_ranks_passages_by_score(tmp_path, monkeypatch):
    sentences = [f"passage {i} with words" for i in range(4)]
    target = make_text_file(tmp_path)
    monkeypatch.setattr(
        document_query, "load_document", lambda path: Document(source="doc.txt", kind="text", text="body")
    )
    instance, _ = make_query(scores=[0.5, 9.0, 0.1, 4.0], sentences=sentences, top_k=2)
    result = instance.query(target, "find things")
    assert result.splitlines()[0] == "Top 2 matches for 'find things' in 'doc.txt':"
    assert "passage 1 " in result.splitlines()[1] and "[9.000]" in result.splitlines()[1]
    assert "passage 3 " in result.splitlines()[2] and "[4.000]" in result.splitlines()[2]


def test_query_defaults_to_configured_top_k(tmp_path, monkeypatch):
    sentences = [f"passage {i} with words" for i in range(6)]
    target = make_text_file(tmp_path)
    monkeypatch.setattr(
        document_query, "load_document", lambda path: Document(source="doc.txt", kind="text", text="body")
    )
    instance, _ = make_query(sentences=sentences, top_k=2)
    assert instance.query(target, "q").splitlines()[0] == "Top 2 matches for 'q' in 'doc.txt':"


def test_query_top_k_override_and_clamp(tmp_path, monkeypatch):
    sentences = ["alpha beta gamma", "delta epsilon", "zeta eta theta"]
    target = make_text_file(tmp_path)
    monkeypatch.setattr(
        document_query, "load_document", lambda path: Document(source="doc.txt", kind="text", text="body")
    )
    instance, _ = make_query(sentences=sentences, top_k=1)
    lines = instance.query(target, "q", top_k=99).splitlines()
    assert lines[0] == "Top 3 matches for 'q' in 'doc.txt':"  # clamped to passage count


def test_query_scores_every_passage_pair(tmp_path, monkeypatch):
    sentences = ["sentence one here", "sentence two here", "sentence three here"]
    target = make_text_file(tmp_path, name="d.md")
    monkeypatch.setattr(
        document_query, "load_document", lambda path: Document(source="d.md", kind="text", text="body")
    )
    instance, fake = make_query(sentences=sentences)
    instance.query(target, "lookup")
    assert fake.seen_inputs == [["lookup", s] for s in sentences]


# ------------------------------------------------------------ input handling


def test_query_empty_query_message(tmp_path):
    target = make_text_file(tmp_path)
    instance, _ = make_query(sentences=["a full sentence"])
    assert instance.query(target, "  ") == "No query provided."
    assert instance.query(target, "") == "No query provided."


def test_query_missing_file_message(tmp_path):
    instance, _ = make_query(sentences=["a full sentence"])
    assert instance.query(tmp_path / "nope.txt", "q") == (
        f"Unable to read '{tmp_path / 'nope.txt'}': file not found."
    )


def test_query_unsupported_file_type_message(tmp_path):
    target = tmp_path / "legacy.doc"
    target.write_text("x", encoding="utf-8")
    instance, _ = make_query(sentences=["a full sentence"])
    assert instance.query(target, "q").startswith("Unable to read 'legacy.doc':")


def test_query_image_file_message(tmp_path, monkeypatch):
    target = tmp_path / "pic.png"
    target.write_bytes(b"png")
    monkeypatch.setattr(
        document_query, "load_document", lambda path: Document(source="pic.png", kind="image", text="pic.png")
    )
    instance, _ = make_query(sentences=["a full sentence"])
    assert instance.query(target, "q") == "No searchable text in 'pic.png' (image file)."


def test_query_document_without_sentences(tmp_path, monkeypatch):
    target = make_text_file(tmp_path, name="d.txt")
    monkeypatch.setattr(
        document_query, "load_document", lambda path: Document(source="d.txt", kind="text", text=". . .")
    )
    instance, _ = make_query(sentences=[])  # tokenizer finds no sentences
    assert instance.query(target, "q") == "No searchable text found in 'd.txt'."


# --------------------------------------------------------------------- pages


def test_query_pdf_passages_keep_page_numbers(tmp_path, monkeypatch):
    sections = [(1, "first page. Second sentence."), (3, "third page sentence.")]
    target = make_text_file(tmp_path, name="paper.pdf")
    monkeypatch.setattr(
        document_query,
        "load_document",
        lambda path: Document(source="paper.pdf", kind="text", text="joined", sections=sections),
    )
    instance, _ = make_query(sentences=["sentence one", "sentence two"])  # 2 per section
    result = instance.query(target, "q")
    assert "(page 1)" in result and "(page 3)" in result


# -------------------------------------------------------------------- caching


def test_passages_cached_until_mtime_changes(tmp_path, monkeypatch):
    calls = {"load": 0}

    def fake_load(path):
        calls["load"] += 1
        return Document(source="doc.txt", kind="text", text="one sentence. two sentences. three sentences.")

    monkeypatch.setattr(document_query, "load_document", fake_load)
    instance, _ = make_query(sentences=["sentence one", "sentence two", "sentence three"])
    target = make_text_file(tmp_path)

    instance.query(target, "q1")
    instance.query(target, "q2")
    assert calls["load"] == 1  # same mtime -> passages never rebuilt

    import os

    os.utime(target, (1_000_000_000, 1_000_000_000))  # force a distinct mtime
    instance.query(target, "q3")
    assert calls["load"] == 2


# ----------------------------------------------------------------- model seam


def test_missing_model_raises_friendly_error(monkeypatch):
    monkeypatch.setattr(document_query, "is_model_available", lambda path: False)
    instance = DocumentQuery("missing-model", sentence_tokenizer=lambda text: ["alpha.", "beta gamma."])
    with pytest.raises(FileNotFoundError) as exc:
        instance._encoder()
    assert "missing-model" in str(exc.value)


def test_defaults_are_sample_compatible():
    assert DEFAULT_CHUNK_SIZE == 3  # 3 consecutive sentences per passage
    assert DEFAULT_TOP_K == 5