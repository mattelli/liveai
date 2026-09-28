import pytest

from src.embedding_search import (
    list_corpora,
    load_metadata,
    resolve_corpus_dir,
    sanitize_corpus_name,
    save_metadata,
)


def test_sanitize_accepts_valid_names():
    for name in ("autosar", "quora_v2", "a.b-c", "Corpus1"):
        assert sanitize_corpus_name(name) == name
    assert sanitize_corpus_name("  autosar  ") == "autosar"


def test_sanitize_rejects_invalid_names():
    for name in ("", "   ", ".", "..", "a/b", "a\\b", "a b", "-bad", ".hidden"):
        with pytest.raises(ValueError):
            sanitize_corpus_name(name)


def test_resolve_corpus_dir(tmp_path):
    assert resolve_corpus_dir("autosar", tmp_path) == tmp_path / "autosar"
    assert resolve_corpus_dir("  autosar  ", tmp_path) == tmp_path / "autosar"


def test_save_and_load_metadata_roundtrip(tmp_path):
    corpus_dir = tmp_path / "autosar"
    metadata = {"name": "autosar", "mode": "text", "records": 3}
    save_metadata(corpus_dir, metadata)
    assert load_metadata(corpus_dir) == metadata


def test_load_metadata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "nope")


def test_list_corpora_empty_root_returns_empty(tmp_path):
    assert list_corpora(tmp_path / "missing") == []
    assert list_corpora(tmp_path) == []


def test_list_corpora_reads_and_sorts(tmp_path):
    save_metadata(tmp_path / "beta", {"name": "beta", "records": 2})
    save_metadata(tmp_path / "alpha", {"name": "alpha", "records": 1})
    (tmp_path / "junk").mkdir()  # no metadata.json -> ignored

    assert [entry["name"] for entry in list_corpora(tmp_path)] == ["alpha", "beta"]
