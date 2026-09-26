import io

from rich.console import Console

from src.common import StreamPrinter, cprint, prompt_input


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
