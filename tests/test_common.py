import io
from pathlib import Path

from rich.console import Console

from src.common import StreamPrinter, cprint, load_config, prompt_input


def test_prompt_input_returns_line(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "hello")
    assert prompt_input("You: ") == "hello"


def test_prompt_input_none_on_eof(monkeypatch):
    def boom():
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    assert prompt_input("You: ") is None


def test_prompt_input_flushes_and_keeps_label(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "x")
    prompt_input("You: ")
    out = capsys.readouterr().out
    assert "You: " in out


def test_cprint_forwards_flush(capsys):
    cprint("abc", "red", end="", flush=True)
    captured = capsys.readouterr()
    assert "abc" in captured.out


def _capture_printer():
    """StreamPrinter writing into an in-memory console (deterministic)."""
    sp = StreamPrinter()
    sp.console = Console(file=io.StringIO(), width=80)
    return sp


def _printed(sp) -> str:
    return sp.console.file.getvalue()


def test_stream_printer_flush_partial(capsys):
    sp = _capture_printer()
    sp.feed("partial")
    sp.flush()
    assert "partial" in _printed(sp)


def test_stream_printer_flush_idempotent(capsys):
    sp = _capture_printer()
    sp.feed("a")
    sp.flush()
    n = len(_printed(sp))
    sp.flush()
    assert len(_printed(sp)) == n


def test_stream_printer_unterminated_code_block():
    sp = _capture_printer()
    sp.feed("```py\nx=1")
    sp.flush()  # must not raise
    assert "x=1" in _printed(sp)


def test_stream_printer_emits_plain_text_incrementally():
    sp = _capture_printer()
    for token in ["Hello", " there", ", how", " are", " you?"]:
        sp.feed(token)
        # No newline and no flush: output must keep growing while streaming.
    assert _printed(sp) == "Hello there, how are you?"


def test_stream_printer_single_line_streams_before_flush():
    sp = _capture_printer()
    sp.feed("Hello")
    assert _printed(sp) == "Hello"  # already emitted, without flush()
    sp.feed(" world")
    assert _printed(sp) == "Hello world"
    sp.flush()
    assert _printed(sp) == "Hello world"


def test_stream_printer_fence_split_across_fragments():
    sp = _capture_printer()
    sp.feed("``")
    assert _printed(sp) == ""  # ambiguous prefix held back
    sp.feed("`py\n")
    assert _printed(sp) == ""  # fence opened; nothing emitted as plain text
    sp.feed("x=1\n")
    assert _printed(sp) == ""  # code buffered until the closing fence
    sp.feed("```\n")
    out = _printed(sp)
    assert "x=1" in out
    assert "```" not in out  # fences themselves are never printed


def test_stream_printer_plain_text_then_code_block():
    sp = _capture_printer()
    sp.feed("answer:\n")
    sp.feed("```py\nx=1\n```\n")
    out = _printed(sp)
    assert "answer:" in out
    assert "x=1" in out
    assert "```" not in out


def test_stream_printer_inline_backticks_are_plain():
    sp = _capture_printer()
    sp.feed("use ``code`` here\n")
    assert "use ``code`` here" in _printed(sp)
    assert sp._in_code is False


def test_stream_printer_two_backticks_line_is_plain():
    sp = _capture_printer()
    sp.feed("``\n")
    assert "``" in _printed(sp)
    assert sp._in_code is False


def test_stream_printer_indent_then_text():
    sp = _capture_printer()
    sp.feed("  ")  # indentation could still become a fence
    assert _printed(sp) == ""
    sp.feed("hello\n")
    assert "  hello" in _printed(sp)


def test_load_config_without_input_block(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[brain]\nmodel = 'x'\n")
    ns = load_config(cfg)  # must not raise
    assert not hasattr(ns, "input")


def test_load_config_resolves_input_history_file(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[input]\nhistory_file = ".chat_history"\n')
    ns = load_config(cfg)
    assert ns.input.history_file == (tmp_path / ".chat_history").resolve()


def test_load_config_resolves_summarization_model_path(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[summarization]\nmodel_path = ".res/models/st"\n')
    ns = load_config(cfg)
    assert ns.summarization.model_path == (tmp_path / ".res/models/st").resolve()


def test_real_config_has_summarization_block():
    cfg_path = Path(__file__).parent.parent / "config.toml"
    ns = load_config(cfg_path)
    sm = ns.summarization
    assert sm.model_path.is_absolute()
    for key in ("device", "ratio", "threshold", "max_sentences"):
        assert hasattr(sm, key)


def test_load_config_resolves_corpora_dir(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[embedding]\nmodel_path = ".res/models/st"\ncorpora_dir = ".res/corpora"\n')
    ns = load_config(cfg)
    assert ns.embedding.model_path == (tmp_path / ".res/models/st").resolve()
    assert ns.embedding.corpora_dir == (tmp_path / ".res/corpora").resolve()


def test_real_config_has_embedding_block():
    cfg_path = Path(__file__).parent.parent / "config.toml"
    ns = load_config(cfg_path)
    emb = ns.embedding
    assert emb.model_path.is_absolute()
    assert emb.corpora_dir.is_absolute()
    for key in ("device", "extensions", "chunk_size", "top_k"):
        assert hasattr(emb, key)


def test_real_config_has_corpora_keys():
    cfg_path = Path(__file__).parent.parent / "config.toml"
    ns = load_config(cfg_path)
    emb = ns.embedding
    assert emb.corpora_dir.is_absolute()
    assert not hasattr(emb, "default_corpus")
    assert not hasattr(emb, "corpus_path")
    assert not hasattr(emb, "processed_name")


def test_real_config_has_multimodal_embedding_keys():
    cfg_path = Path(__file__).parent.parent / "config.toml"
    ns = load_config(cfg_path)
    emb = ns.embedding
    assert emb.image_model_path.is_absolute()
    assert isinstance(emb.multimodal, bool)
    assert emb.image_extensions
    assert "pdf" in emb.extensions and "docx" in emb.extensions
