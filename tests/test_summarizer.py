import importlib
import sys

import numpy as np
import pytest

import src.summarizer as summarizer
from src.summarizer import MAX_INPUT_CHARS, TextSummarizer


class FakeEmbedder:
    """Deterministic embedder; only the similarity matrix must be square."""

    def __init__(self, matrix=None):
        self.calls = 0
        self._matrix = matrix

    def encode(self, sentences):
        self.calls += 1
        if self._matrix is not None:
            return np.asarray(self._matrix, dtype=float)
        return np.eye(len(sentences))


def make_summarizer(sentences, scores=None, monkeypatch=None, **kwargs):
    fake = FakeEmbedder()
    instance = TextSummarizer(
        "model",
        model_factory=lambda path, device: fake,
        sentence_tokenizer=lambda text: list(sentences),
        **kwargs,
    )
    if scores is not None:
        monkeypatch.setattr(
            summarizer,
            "degree_centrality_scores",
            lambda sim, threshold: np.asarray(scores, dtype=float),
        )
    return instance, fake


# ------------------------------------------------------------- summarize_file


def test_summarize_file_text_document(tmp_path, monkeypatch):
    sentences = ["s0.", "s1.", "s2.", "s3."]
    scores = [0.9, 0.1, 0.95, 0.0]
    instance, _ = make_summarizer(sentences, scores=scores, monkeypatch=monkeypatch, ratio=0.5)
    target = tmp_path / "notes.txt"
    target.write_text(" ".join(sentences), encoding="utf-8")
    assert instance.summarize_file(target) == "s0. s2."


def test_summarize_file_tex_document(tmp_path, monkeypatch):
    instance, _ = make_summarizer(
        ["plain words here."], scores=[1.0], monkeypatch=monkeypatch
    )
    target = tmp_path / "paper.tex"
    target.write_text("\\section{Intro}\nplain words here.\n% hidden comment", encoding="utf-8")
    # Single-sentence input: summarize returns the TeX-stripped text verbatim.
    result = instance.summarize_file(target)
    assert "plain words here." in result
    assert "hidden comment" not in result and "section" not in result


def test_summarize_file_reports_load_errors(tmp_path, monkeypatch):
    instance, _ = make_summarizer(["a."], monkeypatch=monkeypatch)
    result = instance.summarize_file(tmp_path / "missing.md")
    assert result.startswith("Unable to read 'missing.md':")


def test_summarize_file_image_document(tmp_path, monkeypatch):
    instance, _ = make_summarizer(["a."], monkeypatch=monkeypatch)
    target = tmp_path / "pic.png"
    target.write_bytes(b"not-a-real-png")
    # Images carry no summarizable text; report that instead of summarizing
    # the file name that the loader uses as placeholder text.
    assert instance.summarize_file(target) == (
        "No extractable text in 'pic.png' (image file)."
    )

# ------------------------------------------------------------- lazy imports


def test_import_summarizer_is_lazy():
    for name in ("sentence_transformers", "nltk", "torch"):
        sys.modules.pop(name, None)
    sys.modules.pop("src.summarizer", None)
    importlib.import_module("src.summarizer")
    for name in ("sentence_transformers", "nltk", "torch"):
        assert name not in sys.modules, f"{name} imported at module load"


# ------------------------------------------------------------ sentence split


def test_split_sentences_with_injected_tokenizer():
    instance = TextSummarizer("p", sentence_tokenizer=lambda text: ["a", "b"])
    assert instance._tokenize("whatever") == ["a", "b"]


def test_split_sentences_regex_fallback(monkeypatch):
    import nltk

    def boom(text):
        raise LookupError("no punkt data")

    monkeypatch.setattr(nltk, "sent_tokenize", boom)
    assert summarizer.split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


# ------------------------------------------------------------------ summarize


def test_summarize_empty_returns_message():
    instance = TextSummarizer("p", sentence_tokenizer=lambda text: [])
    assert instance.summarize("") == "No text to summarize."
    assert instance.summarize("   ") == "No text to summarize."


def test_summarize_single_sentence_returns_input():
    instance = TextSummarizer(
        "p", sentence_tokenizer=lambda text: ["only one sentence."]
    )
    assert instance.summarize("only one sentence.") == "only one sentence."


def test_summarize_respects_ratio(monkeypatch):
    sentences = [f"s{i}." for i in range(10)]
    scores = [0.1] * 10
    scores[5] = 1.0
    instance, _ = make_summarizer(sentences, scores=scores, monkeypatch=monkeypatch, ratio=0.1)
    assert instance.summarize(" ".join(sentences)) == sentences[5]


def test_summarize_respects_max_sentences_override(monkeypatch):
    sentences = [f"s{i}." for i in range(6)]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    instance, _ = make_summarizer(sentences, scores=scores, monkeypatch=monkeypatch, ratio=0.1)
    result = instance.summarize(" ".join(sentences), max_sentences=2)
    assert len(result.split()) == 2
    assert result == "s0. s1."


def test_summarize_preserves_document_order(monkeypatch):
    sentences = ["s0.", "s1.", "s2.", "s3."]
    scores = [0.9, 0.1, 0.95, 0.0]
    instance, _ = make_summarizer(sentences, scores=scores, monkeypatch=monkeypatch, ratio=0.5)
    # Centrality order is s2, s0; the summary must be restored to document order.
    assert instance.summarize(" ".join(sentences)) == "s0. s2."


def test_summarize_truncates_long_input(monkeypatch):
    seen = {}

    def tokenizer(text):
        seen["length"] = len(text)
        return ["a.", "b."]

    instance = TextSummarizer(
        "m",
        model_factory=lambda path, device: FakeEmbedder(),
        sentence_tokenizer=tokenizer,
    )
    monkeypatch.setattr(
        summarizer, "degree_centrality_scores", lambda sim, threshold: np.array([1.0, 0.5])
    )
    instance.summarize("x" * (MAX_INPUT_CHARS + 500))
    assert seen["length"] == MAX_INPUT_CHARS


def test_missing_model_raises_friendly_error(tmp_path, monkeypatch):
    instance = TextSummarizer(
        tmp_path / "missing-model",
        sentence_tokenizer=lambda text: ["a.", "b.", "c."],
    )
    monkeypatch.setattr(
        summarizer,
        "degree_centrality_scores",
        lambda sim, threshold: np.array([1.0, 0.5, 0.1]),
    )
    with pytest.raises(FileNotFoundError) as exc:
        instance.summarize("a. b. c.")
    assert "missing-model" in str(exc.value)


def test_model_is_loaded_once(monkeypatch):
    counter = {"n": 0}

    def factory(path, device):
        counter["n"] += 1
        return FakeEmbedder()

    sentences = ["a.", "b.", "c."]
    instance = TextSummarizer(
        "m",
        model_factory=factory,
        sentence_tokenizer=lambda text: list(sentences),
    )
    monkeypatch.setattr(
        summarizer,
        "degree_centrality_scores",
        lambda sim, threshold: np.array([1.0, 0.5, 0.1]),
    )
    instance.summarize("a. b. c.")
    instance.summarize("a. b. c.")
    assert counter["n"] == 1


def test_is_model_available(tmp_path):
    assert summarizer.is_model_available(tmp_path / "nope") is False
    (tmp_path / "empty").mkdir()
    assert summarizer.is_model_available(tmp_path / "empty") is False
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "modules.json").write_text("{}", encoding="utf-8")
    assert summarizer.is_model_available(saved) is True


def test_summarize_integrates_with_real_lex_rank():
    sentences = [
        "The cat sat on the mat.",
        "The dog barked loudly.",
        "A qubit can be in superposition.",
        "Qubits can be entangled with each other.",
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    class Embedder:
        def encode(self, sents):
            return vectors

    instance = TextSummarizer(
        "m",
        ratio=0.5,
        threshold=0.5,
        model_factory=lambda path, device: Embedder(),
        sentence_tokenizer=lambda text: list(sentences),
    )
    result = instance.summarize(" ".join(sentences))
    # The two qubit sentences form the densest cluster; both are selected and
    # returned in document order.
    assert result == "A qubit can be in superposition. Qubits can be entangled with each other."
