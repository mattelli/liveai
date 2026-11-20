import json
import pickle
from types import SimpleNamespace

import numpy as np

import playground.inspect_corpus as inspect_corpus


def make_payload():
    return {
        "version": 2,
        "mode": "text",
        "text_model_path": "m",
        "image_model_path": "",
        "records": [
            {"kind": "text", "source": "a.md", "text": "hello world", "page": None},
            {"kind": "image", "source": "a.png", "text": "a.png", "image_path": "/tmp/a.png"},
        ],
        "embeddings": np.eye(2, 3, dtype=float),
    }


def write_corpus(tmp_path, name="autosar"):
    corpus_dir = tmp_path / name
    corpus_dir.mkdir(parents=True)
    with open(corpus_dir / "corpus.pkl", "wb") as handle:
        pickle.dump(make_payload(), handle)
    return name


def test_as_records_v2_passthrough():
    payload = make_payload()
    assert inspect_corpus.as_records(payload) == payload["records"]


def test_as_records_v1_passages():
    payload = {"passages": ["one", "two"], "embeddings": np.eye(2)}
    records = inspect_corpus.as_records(payload)
    assert [record["text"] for record in records] == ["one", "two"]
    assert all(record["kind"] == "text" for record in records)


def test_summarize_counts():
    payload = make_payload()
    info = inspect_corpus.summarize(payload, payload["records"], np.asarray(payload["embeddings"]))
    assert info["records"] == 2
    assert info["text_records"] == 1 and info["image_records"] == 1
    assert info["sources"] == 2
    assert info["embeddings_shape"] == [2, 3]


def test_select_records_filters():
    records = make_payload()["records"]

    assert [i for i, _ in inspect_corpus.select_records(
        records, SimpleNamespace(kind="image", source=None, grep=None))] == [1]
    assert [i for i, _ in inspect_corpus.select_records(
        records, SimpleNamespace(kind=None, source="a.md", grep=None))] == [0]
    assert [i for i, _ in inspect_corpus.select_records(
        records, SimpleNamespace(kind=None, source=None, grep="HELLO"))] == [0]
    assert [i for i, _ in inspect_corpus.select_records(
        records, SimpleNamespace(kind=None, source=None, grep=None))] == [0, 1]


def test_checks_reports_normalization():
    payload = make_payload()
    checks = dict(inspect_corpus._checks(payload["records"], np.asarray(payload["embeddings"])))
    assert checks["rows match records"] is True
    assert checks["L2-normalized"] is True


def test_main_requires_corpus_or_list(tmp_path, capsys):
    assert inspect_corpus.main(["--corpora-dir", str(tmp_path)]) == 1
    assert "--corpus" in capsys.readouterr().err


def test_main_missing_corpus_returns_error(tmp_path, capsys):
    assert inspect_corpus.main(["--corpus", "nope", "--corpora-dir", str(tmp_path)]) == 1
    assert "Corpus not found" in capsys.readouterr().err


def test_main_invalid_name_returns_error(tmp_path, capsys):
    assert inspect_corpus.main(["--corpus", "a/b", "--corpora-dir", str(tmp_path)]) == 1
    assert "Invalid corpus name" in capsys.readouterr().err


def test_main_list_reads_metadata(tmp_path, capsys):
    corpus_dir = tmp_path / "autosar"
    corpus_dir.mkdir()
    (corpus_dir / "metadata.json").write_text(
        json.dumps({
            "name": "autosar", "mode": "text", "records": 3,
            "text_records": 3, "image_records": 0, "sources": 1, "built_at": "2026-01-01",
        }),
        encoding="utf-8",
    )

    assert inspect_corpus.main(["--list", "--corpora-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "autosar" in out and "Available corpora (1)" in out


def test_main_list_empty(tmp_path, capsys):
    assert inspect_corpus.main(["--list", "--corpora-dir", str(tmp_path)]) == 0
    assert "No corpora available." in capsys.readouterr().out


def test_main_json_lists_items(tmp_path, capsys):
    name = write_corpus(tmp_path)

    assert inspect_corpus.main(
        ["--corpus", name, "--corpora-dir", str(tmp_path), "--json", "--limit", "1"]
    ) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["records"] == 2  # total count
    assert out["shown"] == 1
    assert len(out["items"]) == 1 and out["items"][0]["index"] == 0


def test_main_record_json(tmp_path, capsys):
    name = write_corpus(tmp_path)

    assert inspect_corpus.main(
        ["--corpus", name, "--corpora-dir", str(tmp_path), "--record", "1", "--json"]
    ) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["kind"] == "image" and record["image_path"] == "/tmp/a.png"


def test_main_record_out_of_range(tmp_path, capsys):
    name = write_corpus(tmp_path)

    assert inspect_corpus.main(
        ["--corpus", name, "--corpora-dir", str(tmp_path), "--record", "99"]
    ) == 1
    assert "out of range" in capsys.readouterr().err
