from logging import getLogger
from types import SimpleNamespace

import pytest

pytest.importorskip("lmstudio", reason="live_ai imports lmstudio at module level")

import src.live_ai as live_ai_mod  # noqa: E402


def _bare_brain(speech_gen):
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(speech_gen=speech_gen)
    return brain


def test_allow_speech_forwards_config_and_closes(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)
            self.closed = 0

        def close(self):
            self.closed += 1

    monkeypatch.setattr(live_ai_mod, "SpeechGenerator", Recorder)

    brain = _bare_brain(SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.3, speech_speed=1.5,
        speaker_id=2, noise_scale=0.4, noise_w_scale=0.7, normalize_audio=True,
    ))

    assert brain.allow_speech(True) == "Speech Generator initialized"
    speech = brain.speech
    assert seen["speaker_id"] == 2 and seen["noise_scale"] == 0.4
    assert seen["noise_w_scale"] == 0.7 and seen["normalize_audio"] is True
    assert seen["speech_speed"] == 1.5

    assert brain.allow_speech(False) == "Speech Generator released"
    assert speech.closed == 1
    assert not hasattr(brain, "speech")


def test_allow_speech_without_optional_keys(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)

        def close(self):
            pass

    monkeypatch.setattr(live_ai_mod, "SpeechGenerator", Recorder)

    brain = _bare_brain(SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.5, speech_speed=1.0,
    ))
    brain.allow_speech(True)
    assert seen["speaker_id"] is None
    assert seen["noise_scale"] == 1.0
    assert seen["noise_w_scale"] == 1.0
    assert seen["normalize_audio"] is False


def _bare_summarization_brain(**overrides):
    cfg = dict(model_path="m", device="cpu", ratio=0.1, threshold=0.5, max_sentences=0)
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(summarization=SimpleNamespace(**cfg))
    return brain


def test_summarize_text_reads_file_and_caches(tmp_path, monkeypatch):
    seen = {"calls": [], "count": 0}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def summarize(self, text, max_sentences=None):
            seen["calls"].append((text, max_sentences))
            return "SUMMARY"

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)

    first = tmp_path / "first.txt"
    first.write_text("long text", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("more text", encoding="utf-8")

    brain = _bare_summarization_brain(ratio=0.2, threshold=0.4, max_sentences=3)
    assert brain.summarize_text(str(first)) == "SUMMARY"
    assert brain.summarize_text(str(second)) == "SUMMARY"

    assert seen["count"] == 1  # summarizer created once and cached
    assert seen["init"]["model_path"] == "m"
    assert seen["init"]["ratio"] == 0.2
    assert seen["init"]["threshold"] == 0.4
    assert seen["init"]["max_sentences"] == 3
    assert seen["calls"] == [("long text", None), ("more text", None)]


def test_summarize_text_missing_file_returns_message(tmp_path):
    brain = _bare_summarization_brain()
    out = brain.summarize_text(str(tmp_path / "nope.txt"))
    assert "Unable to read" in out


def test_summarize_text_forwards_max_sentences(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize(self, text, max_sentences=None):
            seen["max"] = max_sentences
            return "S"

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    brain = _bare_summarization_brain()
    assert brain.summarize_text(str(target), max_sentences=2) == "S"
    assert seen["max"] == 2


def test_summarize_text_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize(self, text, max_sentences=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "TextSummarizer", Recorder)
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    brain = _bare_summarization_brain()
    out = brain.summarize_text(str(target))
    assert out.startswith("Error:") and "boom" in out


def test_agent_build_tools_includes_summarize_text():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "summarize_text" in names


def test_agent_build_tools_keeps_existing_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    for name in ("system_info", "read_text_file", "web_search", "image_response"):
        assert name in names


def _bare_embedding_brain(**overrides):
    cfg = dict(model_path="m", device="cpu", extensions="txt, md",
               chunk_size=3, top_k=5, corpus_path="corpus.pkl")
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(embedding=SimpleNamespace(**cfg))
    return brain


def test_store_corpus_uses_config_and_caches(monkeypatch):
    seen = {"count": 0}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def store(self, directory, output, overwrite=False):
            seen["store"] = (directory, output, overwrite)
            return "STORED"

        def query(self, corpus_file, query, top_k=None, kind=None):
            raise AssertionError("query not expected")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain()
    assert brain.store_corpus("/docs") == "STORED"
    assert brain.store_corpus("/docs") == "STORED"

    assert seen["count"] == 1  # corpus object created once and cached
    assert seen["init"]["model_path"] == "m"
    assert seen["init"]["chunk_size"] == 3
    assert seen["init"]["top_k"] == 5
    assert seen["store"] == ("/docs", "corpus.pkl", False)


def test_store_corpus_forwards_explicit_file_and_overwrite(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, output, overwrite=False):
            seen["store"] = (directory, output, overwrite)
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain()
    assert brain.store_corpus("/docs", corpus_file="/tmp/c.pkl", overwrite=True) == "STORED"
    assert seen["store"] == ("/docs", "/tmp/c.pkl", True)


def test_query_corpus_forwards_query_and_top_k(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_file, query, top_k=None, kind=None):
            seen["query"] = (corpus_file, query, top_k)
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain()
    assert brain.query_corpus("hello", top_k=2) == "RESULTS"
    assert seen["query"] == ("corpus.pkl", "hello", 2)


def test_query_corpus_uses_explicit_corpus_file(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_file, query, top_k=None, kind=None):
            seen["query"] = (corpus_file, query, top_k)
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain()
    brain.query_corpus("hello", corpus_file="/tmp/c.pkl")
    assert seen["query"] == ("/tmp/c.pkl", "hello", None)


def test_query_corpus_forwards_kind(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_file, query, top_k=None, kind=None):
            seen["query"] = (corpus_file, query, top_k, kind)
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain()
    assert brain.query_corpus("cluster diagram", kind="image") == "RESULTS"
    assert seen["query"] == ("corpus.pkl", "cluster diagram", None, "image")


def test_store_corpus_reports_errors(monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, output, overwrite=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().store_corpus("/docs")
    assert out.startswith("Error:") and "boom" in out


def test_query_corpus_reports_errors(monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_file, query, top_k=None, kind=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().query_corpus("hi")
    assert out.startswith("Error:") and "boom" in out


def test_agent_build_tools_includes_embedding_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "store_corpus" in names and "query_corpus" in names


def test_agent_build_tools_keeps_summarize_text():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "summarize_text" in names


def test_store_corpus_passes_multimodal_config(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def store(self, directory, output, overwrite=False):
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(
        image_model_path="clip", multimodal=True, processed_name="PROC.md",
        image_extensions="png, jpg",
    )
    assert brain.store_corpus("/docs") == "STORED"
    assert seen["init"]["image_model_path"] == "clip"
    assert seen["init"]["multimodal"] is True
    assert seen["init"]["processed_name"] == "PROC.md"
    assert seen["init"]["image_extensions"] == "png, jpg"


def test_embedding_corpus_defaults_when_config_is_minimal(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def query(self, corpus_file, query, top_k=None, kind=None):
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    # Phase 1 style config: no image_model_path/multimodal/processed_name keys.
    brain = _bare_embedding_brain()
    brain.query_corpus("hi")
    assert seen["init"]["multimodal"] is False
    assert seen["init"]["image_model_path"] is None
    assert seen["init"]["processed_name"] == "PROCESSED.md"
