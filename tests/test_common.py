import io
import pytest
from pathlib import Path

from rich.console import Console

from src.common import (
    _SPIN_FRAMES, StreamPrinter, cprint, load_config, prompt_input, truncate_text,
)


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


def test_load_config_resolves_input_history_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # _cwd paths are per launch directory
    cfg = tmp_path / "config.toml"
    cfg.write_text('[input]\nhistory_file_cwd = ".chat_history"\n')
    ns = load_config(cfg)
    # Non-data_dir _cwd keys resolve under the data root (cwd/.liveai), not the
    # config directory.
    assert ns.input.history_file_cwd == (tmp_path / ".liveai" / ".chat_history").resolve()


def test_load_config_cwd_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[brain]\nmodel = 'x'\n")
    ns = load_config(cfg)
    assert ns.cwd.data_dir_cwd == (tmp_path / ".liveai").resolve()
    assert ns.cwd.tmp_dir_cwd == (tmp_path / ".liveai" / "tmp").resolve()
    assert ns.cwd.behaviour_dir_cwd == (tmp_path / ".liveai" / "behaviour").resolve()


def test_load_config_cwd_keys_resolve_under_data_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[brain]\nmemory_path_cwd = "memory.log"\nsessions_dir_cwd = "sessions"\n'
        '[agent]\nmonitor_log_file_cwd = "agents.jsonl"\n'
        '[input]\nhistory_file_cwd = "input.log"\n'
        '[delegation.scheduler]\npersist_file_cwd = "schedule.json"\n'
    )
    ns = load_config(cfg)
    root = (tmp_path / ".liveai").resolve()
    assert ns.brain.memory_path_cwd == root / "memory.log"
    assert ns.brain.sessions_dir_cwd == root / "sessions"
    assert ns.agent.monitor_log_file_cwd == root / "agents.jsonl"
    assert ns.input.history_file_cwd == root / "input.log"
    assert ns.delegation.scheduler.persist_file_cwd == root / "schedule.json"


def test_load_config_data_dir_cwd_is_the_root(tmp_path, monkeypatch):
    # data_dir_cwd resolves against the launch dir; the other _cwd keys follow
    # it, so a custom data root moves every state file with it.
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[cwd]\ndata_dir_cwd = ".state"\n'
        '[brain]\nmemory_path_cwd = "memory.log"\n'
    )
    ns = load_config(cfg)
    assert ns.cwd.data_dir_cwd == (tmp_path / ".state").resolve()
    assert ns.brain.memory_path_cwd == (tmp_path / ".state" / "memory.log").resolve()


def test_load_config_assets_still_resolve_against_config_dir(tmp_path, monkeypatch):
    # The launch directory differs from the config directory: assets must
    # follow the config file, _cwd state must follow the launch directory.
    launch = tmp_path / "launch"
    launch.mkdir()
    cfg_dir = tmp_path / "proj"
    cfg_dir.mkdir()
    monkeypatch.chdir(launch)
    cfg = cfg_dir / "config.toml"
    cfg.write_text(
        '[brain]\nmemory_path_cwd = "memory.log"\n'
        '[embedding]\nmodel_path = ".res/models/st"\ncorpora_dir = ".res/corpora"\n'
        '[image_gen]\nout_dir = ".out_images"\n'
    )
    ns = load_config(cfg)
    assert ns.embedding.model_path == (cfg_dir / ".res/models/st").resolve()
    assert ns.embedding.corpora_dir == (cfg_dir / ".res/corpora").resolve()
    assert ns.image_gen.out_dir == (cfg_dir / ".out_images").resolve()
    assert ns.brain.memory_path_cwd == (launch / ".liveai" / "memory.log").resolve()


def test_load_config_custom_cwd_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[cwd]\ndata_dir_cwd = ".state"\ntmp_dir_cwd = "scratch"\n'
        'behaviour_dir_cwd = "rules"\n'
        '[brain]\nmemory_path_cwd = "memory.log"\n'
    )
    ns = load_config(cfg)
    root = (tmp_path / ".state").resolve()
    assert ns.cwd.data_dir_cwd == root
    assert ns.cwd.tmp_dir_cwd == root / "scratch"
    assert ns.cwd.behaviour_dir_cwd == root / "rules"
    assert ns.brain.memory_path_cwd == root / "memory.log"


def test_ensure_data_dirs_creates_folders(tmp_path, monkeypatch):
    from src.common import ensure_data_dirs
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[brain]\nmodel = 'x'\n")
    ns = load_config(cfg)
    ensure_data_dirs(ns)
    assert (tmp_path / ".liveai").is_dir()
    assert (tmp_path / ".liveai" / "tmp").is_dir()
    assert (tmp_path / ".liveai" / "behaviour").is_dir()


def test_ensure_data_dirs_tolerates_missing_cwd():
    from types import SimpleNamespace
    from src.common import ensure_data_dirs
    ensure_data_dirs(SimpleNamespace())  # no [cwd] -> no-op, no raise


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



# ------------------------------------------------ truncate_text (shared helper)


def test_truncate_text_unchanged_under_limit():
    assert truncate_text("hello", 4000) == "hello"


def test_truncate_text_unchanged_when_disabled():
    text = "x" * 50_000
    assert truncate_text(text, 0) == text
    assert truncate_text(text, -5) == text


def test_truncate_text_keeps_head_and_tail():
    text = "HEAD" + "m" * 10_000 + "TAIL"
    result = truncate_text(text, 4000)
    assert len(result) < len(text)
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert f"[truncated, {len(text)} chars total]" in result


def test_truncate_text_head_tail_split_is_three_quarters():
    text = "A" * 10_000 + "Z" * 10_000
    result = truncate_text(text, 4000)
    assert result.startswith("A" * 3000)      # head = limit * 3 // 4
    assert result.endswith("Z" * 1000)        # tail = limit // 4
    assert "A" * 3001 not in result           # middle content dropped


def test_truncate_text_reports_original_length():
    text = "y" * 12_345
    result = truncate_text(text, 1000)
    assert "[truncated, 12345 chars total]" in result


def test_truncate_text_cuts_are_clean_at_marker():
    """Head/tail cuts are stripped so no fragment can merge into the marker."""
    text = "A" * 10_000 + "   \n" + "Z" * 10_000
    result = truncate_text(text, 4000)
    marker_line = f"\n… [truncated, {len(text)} chars total] …\n"
    assert marker_line in result
    head, _, tail = result.partition(marker_line)
    assert head.endswith("A")          # head cut stripped to clean payload
    assert tail.startswith("Z")        # tail cut stripped to clean payload


# ------------------------------------------------- process registry (Task 6)


def test_process_registry_terminates_registered_processes():
    """terminate_all() kills every registered process and clears the registry."""
    import src.common as common

    class FakeProcess:
        def __init__(self):
            self.killed = 0

        def kill(self):
            self.killed += 1

    first, second = FakeProcess(), FakeProcess()
    common.register_process(first)
    common.register_process(second)
    assert common.terminate_all() == 2
    assert first.killed == 1 and second.killed == 1
    assert common.terminate_all() == 0  # registry cleared


def test_process_registry_unregister_removes_process():
    """A finished process is no longer terminated."""
    import src.common as common

    class FakeProcess:
        def __init__(self):
            self.killed = 0

        def kill(self):
            self.killed += 1

    process = FakeProcess()
    common.register_process(process)
    common.unregister_process(process)
    assert common.terminate_all() == 0
    assert process.killed == 0


def test_process_registry_marks_terminated_processes():
    """A killed process is flagged so its tool can report a cancel."""
    import src.common as common

    class FakeProcess:
        def kill(self):
            pass

    process = FakeProcess()
    assert common.process_terminated(process) is False
    common.register_process(process)
    common.terminate_all()
    assert common.process_terminated(process) is True


def test_process_registry_terminate_all_never_raises():
    """A process that cannot be killed is skipped, not fatal."""
    import src.common as common

    class BrokenProcess:
        pid = None  # no pid -> plain kill() path

        def kill(self):
            raise RuntimeError("already gone")

    common.register_process(BrokenProcess())
    assert common.terminate_all() == 0  # counted as not killed, no raise


def test_terminate_all_kills_the_process_tree_on_windows(monkeypatch):
    """On Windows the whole tree is killed (taskkill /F /T), not just the shell."""
    import src.common as common

    calls = []

    class FakeProcess:
        pid = 4321

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(common.sys, "platform", "win32")
    monkeypatch.setattr(
        common.subprocess, "run",
        lambda argv, **kw: calls.append(("taskkill", argv)))
    common.register_process(FakeProcess())
    assert common.terminate_all() == 1
    assert calls == [("taskkill", ["taskkill", "/F", "/T", "/PID", "4321"])]
    assert "kill" not in calls  # the tree kill replaces the plain kill


def test_terminate_all_falls_back_to_kill_without_pid(monkeypatch):
    """A process without a pid is killed directly (non-Windows path)."""
    import src.common as common

    calls = []

    class FakeProcess:
        pid = None

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(common.sys, "platform", "linux")
    common.register_process(FakeProcess())
    assert common.terminate_all() == 1
    assert calls == ["kill"]
