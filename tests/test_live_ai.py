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
               chunk_size=3, top_k=5, corpora_dir="/res/corpora")
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(embedding=SimpleNamespace(**cfg))
    return brain


def test_store_corpus_requires_name(monkeypatch):
    import inspect

    params = inspect.signature(live_ai_mod.Brain.store_corpus).parameters
    assert params["corpus_name"].default is inspect.Parameter.empty

    seen = {"store": 0}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, *args, **kwargs):
            seen["store"] += 1
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().store_corpus("/docs", "")
    assert out.startswith("Error: a corpus name is required")
    assert seen["store"] == 0


def test_query_corpus_requires_name(monkeypatch):
    import inspect

    params = inspect.signature(live_ai_mod.Brain.query_corpus).parameters
    assert params["corpus_name"].default is inspect.Parameter.empty

    seen = {"query": 0}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, *args, **kwargs):
            seen["query"] += 1
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().query_corpus("hello", "")
    assert out.startswith("Error: a corpus name is required")
    assert seen["query"] == 0


def test_store_corpus_uses_named_corpus(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, corpus_dir, overwrite=False):
            seen["store"] = (directory, str(corpus_dir), overwrite)
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(corpora_dir=str(tmp_path))
    assert brain.store_corpus("/docs", "autosar", overwrite=True) == "STORED"
    assert seen["store"] == ("/docs", str(tmp_path / "autosar"), True)


def test_store_corpus_invalid_name_returns_error(monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, *args, **kwargs):
            raise AssertionError("store must not be called")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain().store_corpus("/docs", "a/b")
    assert out.startswith("Error:") and "corpus name" in out.lower()


def test_query_corpus_resolves_name_and_forwards(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_dir, query, top_k=None, kind=None):
            seen["query"] = (str(corpus_dir), query, top_k, kind)
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(corpora_dir=str(tmp_path))
    assert brain.query_corpus("hello", "autosar", kind="image", top_k=2) == "RESULTS"
    assert seen["query"] == (str(tmp_path / "autosar"), "hello", 2, "image")


def test_list_corpora_formats_entries(monkeypatch):
    entries = [
        {"name": "autosar", "mode": "multimodal", "records": 10, "text_records": 8,
         "image_records": 2, "sources": 3, "built_at": "2026-01-01 00:00:00 UTC"},
        {"name": "quora", "mode": "text", "records": 5, "text_records": 5,
         "image_records": 0, "sources": 1, "built_at": "2026-01-02 00:00:00 UTC"},
    ]
    monkeypatch.setattr(live_ai_mod, "list_corpora", lambda corpora_dir: entries)
    out = _bare_embedding_brain().list_corpora()
    assert "Available corpora (2)" in out
    assert "autosar: mode=multimodal" in out and "quora: mode=text" in out


def test_list_corpora_empty_returns_message(monkeypatch):
    monkeypatch.setattr(live_ai_mod, "list_corpora", lambda corpora_dir: [])
    assert _bare_embedding_brain().list_corpora() == "No corpora available."


def test_store_corpus_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, corpus_dir, overwrite=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain(corpora_dir=str(tmp_path)).store_corpus("/docs", "autosar")
    assert out.startswith("Error:") and "boom" in out


def test_query_corpus_reports_errors(tmp_path, monkeypatch):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_dir, query, top_k=None, kind=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    out = _bare_embedding_brain(corpora_dir=str(tmp_path)).query_corpus("hi", "autosar")
    assert out.startswith("Error:") and "boom" in out


def test_agent_build_tools_includes_embedding_tools():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "store_corpus" in names and "query_corpus" in names


def test_camel_tool_definition_advertises_and_translates():
    def query_corpus(query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
        """Search a named corpus."""
        return f"{query}|{corpus_name}|{kind}|{top_k}"

    definition = live_ai_mod._camel_tool_definition(query_corpus)
    assert set(definition["parameters"]) == {"query", "corpusName", "kind", "topK"}
    assert definition["parameters"]["corpusName"] is str
    assert definition["parameters"]["topK"] == {"type": int, "default": 0}
    # The model emits camelCase; the implementation maps it to the real names.
    assert definition["implementation"](
        query="q", corpusName="autosar", kind="image", topK=2
    ) == "q|autosar|image|2"


def test_camel_tool_definition_schema_uses_camel_case():
    from lmstudio.json_api import ChatResponseEndpoint

    def query_corpus(query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
        """Search a named corpus."""
        return "ok"

    definition = live_ai_mod._camel_tool_definition(query_corpus)
    tools, client_map = ChatResponseEndpoint.parse_tools([definition])
    schema = tools.to_dict()["tools"][0]["function"]["parameters"]
    assert schema["required"] == ["query", "corpusName"]
    assert "corpusName" in schema["properties"] and "topK" in schema["properties"]
    assert "corpus_name" not in schema["properties"]


def test_agent_build_tools_includes_list_corpora():
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "list_corpora" in names


def test_store_corpus_passes_multimodal_config(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def store(self, directory, corpus_dir, overwrite=False):
            return "STORED"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(
        corpora_dir=str(tmp_path),
        image_model_path="clip", multimodal=True, image_extensions="png, jpg",
    )
    assert brain.store_corpus("/docs", "autosar") == "STORED"
    assert seen["init"]["image_model_path"] == "clip"
    assert seen["init"]["multimodal"] is True
    assert seen["init"]["image_extensions"] == "png, jpg"
    assert "processed_name" not in seen["init"]


def test_embedding_corpus_defaults_when_config_is_minimal(tmp_path, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def query(self, corpus_dir, query, top_k=None, kind=None):
            return "RESULTS"

    monkeypatch.setattr(live_ai_mod, "EmbeddingCorpus", Recorder)
    brain = _bare_embedding_brain(corpora_dir=str(tmp_path))
    brain.query_corpus("hi", "autosar")
    assert seen["init"]["multimodal"] is False
    assert seen["init"]["image_model_path"] is None
    assert "processed_name" not in seen["init"]
