import io
import pytest
from pathlib import Path

from rich.console import Console

from src.common import _SPIN_FRAMES, StreamPrinter, cprint, load_config, prompt_input


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


# -------------------------------------------------------- markdown rendering


def _capture_markdown_printer():
    """StreamPrinter in markdown mode writing into an in-memory console."""
    sp = StreamPrinter(markdown=True)
    sp.console = Console(file=io.StringIO(), width=80)
    return sp


def test_markdown_printer_buffers_until_flush():
    sp = _capture_markdown_printer()
    sp.feed("# Hello")
    # Nothing emitted while streaming: markdown needs the whole response.
    assert _printed(sp) == ""
    sp.feed("\n\nSome **bold** text")
    assert _printed(sp) == ""
    sp.flush()
    out = _printed(sp)
    assert "Hello" in out
    assert "bold" in out
    assert "#" not in out  # heading marker consumed by the renderer


def test_markdown_printer_renders_heading_and_list():
    sp = _capture_markdown_printer()
    sp.feed("# Title\n\n- item one\n- item two\n")
    sp.flush()
    out = _printed(sp)
    assert "Title" in out
    assert "item one" in out
    assert "item two" in out


def test_markdown_printer_renders_code_block():
    sp = _capture_markdown_printer()
    sp.feed("```py\nx = 1\n```\n")
    sp.flush()
    out = _printed(sp)
    assert "x = 1" in out
    assert "```" not in out


def test_markdown_printer_empty_flush_is_noop():
    sp = _capture_markdown_printer()
    sp.feed("   \n")
    sp.flush()  # must not raise
    assert _printed(sp) == ""


def test_markdown_printer_renders_table():
    sp = _capture_markdown_printer()
    sp.feed("| a | b |\n|---|---|\n| 1 | 2 |\n")
    sp.flush()
    out = _printed(sp)
    assert "a" in out and "b" in out
    assert "1" in out and "2" in out


def test_markdown_printer_blank_line_between_thinking_and_answer():
    """A 💭 dump is separated from the Markdown answer by a blank line.

    Markdown mode buffers the whole stream and renders once at flush(), so
    the thinking/answer separator must fire there (the streaming path does
    it in _emit, which markdown mode never reaches).
    """
    sp = _capture_markdown_printer()
    sp.feed_thinking("pondering")
    sp.feed("# Hello")
    sp.flush()
    out = _printed(sp)
    assert "pondering" in out
    assert "Hello" in out
    assert out.index("pondering") < out.rindex("Hello")
    assert out.count("\n\n") == 1


def test_markdown_printer_no_separator_without_thinking():
    """Without a 💭 dump the Markdown answer gets no leading blank line."""
    sp = _capture_markdown_printer()
    sp.feed("# Hello")
    sp.flush()
    assert "\n\n" not in _printed(sp)


def test_markdown_printer_separator_fires_once_across_flushes():
    """The separator is consumed by the first flush; later flushes are plain."""
    sp = _capture_markdown_printer()
    sp.feed_thinking("pondering")
    sp.feed("first")
    sp.flush()
    sp.feed("second")
    sp.flush()
    out = _printed(sp)
    assert out.count("\n\n") == 1


def test_markdown_printer_non_markdown_default_streams():
    """Without markdown=True the printer keeps streaming plain text."""
    sp = _capture_printer()
    sp.feed("Hello **world**")
    assert _printed(sp) == "Hello **world**"  # raw literal text, streamed


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


def test_real_config_has_image_gen_block():
    cfg_path = Path(__file__).parent.parent / "config.toml"
    ns = load_config(cfg_path)
    igm = ns.image_gen
    assert igm.out_dir.is_absolute()  # resolved relative to the config file
    for key in ("t2i_model", "i2i_model", "device", "negative_prompt",
                "width", "height", "steps", "seed", "guidance", "denoise"):
        assert hasattr(igm, key)


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


# ------------------------------------------------------------- activity spinner


def test_spinner_draws_only_on_clean_line():
    sp = _capture_printer()
    sp._spin_active = True
    sp._spin_note = "thinking"
    with sp._spin_lock:
        sp._draw_spinner_locked()
    drawn = sp.console.file.getvalue()
    assert drawn.startswith("\r\x1b[K")
    assert any(frame in drawn for frame in _SPIN_FRAMES) and "thinking" in drawn
    assert sp._spin_drawn is True

    # real output erases the spinner and marks the line dirty
    sp._emit("hello")
    assert sp._dirty_line is True and sp._spin_drawn is False

    # while the line is dirty the refresh loop must not draw again
    n = len(sp.console.file.getvalue())
    with sp._spin_lock:
        if sp._spin_active and not sp._spin_suppressed and not sp._dirty_line:
            sp._draw_spinner_locked()
    assert len(sp.console.file.getvalue()) == n


def test_spinner_reports_buffered_code():
    sp = _capture_printer()
    sp.feed("```python\nx = 1\n")
    with sp._spin_lock:
        sp._draw_spinner_locked()
    assert "writing python code" in sp.console.file.getvalue()


def test_spinner_stop_erases_and_clears_hook():
    sp = _capture_printer()
    sp._spin_active = True
    with sp._spin_lock:
        sp._draw_spinner_locked()
    sp.stop_spinner()
    # the erase sequence was written after the last draw
    assert sp.console.file.getvalue().count("\r\x1b[K") >= 2
    from src.common import _output_hook
    assert _output_hook is None
    assert sp._spin_active is False


def test_cprint_invokes_output_hook(capsys):
    from src.common import set_output_hook
    calls = []
    set_output_hook(lambda: calls.append(1))
    try:
        cprint("banner", "grey")
    finally:
        set_output_hook(None)
    assert calls == [1]
    assert "banner" in capsys.readouterr().out


# -------------------------------------------- tool-call block (dump_func_call)

import src.common as common_mod  # noqa: E402


def _plain():
    """Neutralize the spinner output hook during block-print tests."""
    common_mod.set_output_hook(None)


def test_dump_func_call_success_green_multiline(capsys, monkeypatch):
    monkeypatch.setattr(common_mod, "_call_lines_max", 8)
    _plain()

    @common_mod.dump_func_call
    def search(query, n_results=5):
        return "ok"

    assert search("etf list") == "ok"
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == ""  # blank separator line before the block
    body = [line for line in lines if line]
    assert len(body) == 4  # header + two parameters + status
    assert "\x1b[1;90;48;5;22m[CALL] search(" in body[0]
    assert "query = 'etf list'" in body[1]
    assert "n_results = 5" in body[2]
    assert "-> ok (" in body[3]
    # grey foreground on dark-green background (256-color 22), bold
    assert all("\x1b[1;90;48;5;22m" in line for line in body)


def test_dump_func_call_failure_red_block(capsys, monkeypatch):
    monkeypatch.setattr(common_mod, "_call_lines_max", 8)
    _plain()

    @common_mod.dump_func_call
    def boom(path):
        raise ValueError("bad path")

    with pytest.raises(ValueError):
        boom("/nope")
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line]
    assert any("failed in" in line for line in lines)
    assert any("bad path" in line for line in lines)
    assert any("path = '/nope'" in line for line in lines)
    # grey foreground on dark-red background (256-color 88)
    assert all("\x1b[1;90;48;5;88m" in line for line in lines)


def test_dump_func_call_caps_block_lines(capsys, monkeypatch):
    monkeypatch.setattr(common_mod, "_call_lines_max", 5)
    _plain()

    @common_mod.dump_func_call
    def many(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8):
        return "ok"

    many()
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line]
    assert len(lines) == 5  # header + 2 params + elision + status
    assert any("… (+6 more parameters)" in line for line in lines)


def test_dump_func_call_skipped_when_call_lines_max_zero(capsys, monkeypatch):
    """[ui] call_lines_max = 0 disables the [CALL] dump entirely."""
    monkeypatch.setattr(common_mod, "_call_lines_max", 0)
    _plain()

    @common_mod.dump_func_call
    def silent(x, y=2):
        return "ok"

    assert silent(1) == "ok"
    assert capsys.readouterr().out == ""  # nothing printed, no separator either

    @common_mod.dump_func_call
    def silent_boom(x):
        raise ValueError("bad")

    with pytest.raises(ValueError):
        silent_boom(1)
    assert capsys.readouterr().out == ""  # failures stay silent too


def test_dump_func_call_self_and_long_values(capsys, monkeypatch):
    monkeypatch.setattr(common_mod, "_call_lines_max", 8)
    _plain()

    class Holder:
        @common_mod.dump_func_call
        def method(self, content, keep):
            return "ok"

    Holder().method("x" * 500, keep="y")
    out = capsys.readouterr().out
    assert "self" not in out
    long_line = next(line for line in out.splitlines() if "content" in line)
    assert len(long_line) < 200  # value truncated, block stays narrow
    assert "…" in long_line


def test_configure_ui_parses_call_lines_max(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setattr(common_mod, "_call_lines_max", 8)
    common_mod.configure_ui(NS(ui=NS(call_lines_max=3)))
    assert common_mod._call_lines_max == 3
    common_mod.configure_ui(NS(ui=NS(call_lines_max="12")))
    assert common_mod._call_lines_max == 12
    common_mod.configure_ui(NS(ui=NS(call_lines_max="bogus")))  # ignored
    assert common_mod._call_lines_max == 12
    common_mod.configure_ui(NS())  # no [ui] at all -> keep
    assert common_mod._call_lines_max == 12
    common_mod.configure_ui(NS(ui=NS(call_lines_max=0)))  # explicit 0 -> dump off
    assert common_mod._call_lines_max == 0


def test_stream_printer_blank_line_between_thinking_and_answer(capsys):
    """The first answer text after a 💭 dump starts on a fresh line."""
    printer = StreamPrinter()
    sink = io.StringIO()
    printer.console.file = sink
    printer.feed_thinking("pondering")
    printer.feed("the answer")
    out = sink.getvalue()
    # thinking text, blank separator, then the answer
    assert "pondering" in out
    assert "\n\n" in out
    assert out.index("pondering") < out.rindex("the answer")
    # the separator fires only once: subsequent answer chunks are contiguous
    printer.feed(" more")
    out = sink.getvalue()
    assert "the answer more" in out
    assert out.count("\n\n") == 1
