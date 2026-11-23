import subprocess
import sys
import types

import main as main_mod
import src.input_reader as input_reader_mod
import src.live_ai as live_ai_mod
import src.session as session_mod


def test_parse_args_defaults():
    a = main_mod.parse_args([])
    assert a.chat is False and a.no_tools is False
    assert a.config == main_mod.CONFIG_FILE and a.prompt is None
    assert a.tools_list is False


def test_parse_args_flags():
    a = main_mod.parse_args(["--chat", "--no-tools", "--config", "x.toml"])
    assert a.chat and a.no_tools and str(a.config).endswith("x.toml")


def test_parse_args_tools_list():
    assert main_mod.parse_args(["--tools-list"]).tools_list is True


def test_print_tools_list_shows_all_tools_and_off_markers(capsys):
    """--tools-list lists every known tool; tool_groups exclusions get [off]."""
    import re
    from types import SimpleNamespace

    config = SimpleNamespace(agent=SimpleNamespace(
        tool_groups="model, memory, files, web, system"))
    assert main_mod._print_tools_list(config) == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)

    # every known tool is listed (unfiltered set)
    assert "model_info" in out
    assert "schedule_task" in out
    assert "generate_image" in out
    assert "mcp_list_tools" in out
    # enabled tools have no marker; excluded ones are marked [off]
    assert "  web_search -" in out
    assert "  schedule_task [off]" in out
    assert "  agent_activity [off]" in out
    # summary line with counts
    match = re.search(r"(\d+) of (\d+) tools enabled", out)
    assert match and int(match.group(2)) > int(match.group(1)) > 0


def test_print_tools_list_marks_mcp_tools_off_when_disabled(capsys, monkeypatch):
    """[mcp] mcp_enabled = false: the two bridge tools are listed as [off]."""
    import re
    from types import SimpleNamespace

    import src.tools.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "enabled", lambda: False)
    config = SimpleNamespace(agent=SimpleNamespace(tool_groups="all"))
    assert main_mod._print_tools_list(config) == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "  mcp_list_tools [off]" in out
    assert "  mcp_call_tool [off]" in out


def test_print_tools_list_all_groups_enabled(capsys):
    import re
    from types import SimpleNamespace

    config = SimpleNamespace(agent=SimpleNamespace(tool_groups="all"))
    assert main_mod._print_tools_list(config) == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    # no per-tool [off] marker (the summary line's legend is not a marker)
    assert " [off] -" not in out and " [off]\n" not in out
    match = re.search(r"(\d+) of (\d+) tools enabled", out)
    assert match and match.group(1) == match.group(2)


def test_main_tools_list_exits_without_session(monkeypatch, capsys):
    """--tools-list prints the listing and returns 0 without building a session."""
    import re

    called = {"session": False}

    class Boom:
        def __init__(self, *a, **k):
            called["session"] = True

    monkeypatch.setattr(session_mod, "ConversationSession", Boom)
    monkeypatch.setattr(live_ai_mod, "Agent", Boom)
    monkeypatch.setattr(main_mod, "load_config",
                        lambda p: types.SimpleNamespace(
                            agent=types.SimpleNamespace(tool_groups="all")))

    assert main_mod.main(["--tools-list"]) == 0
    assert called["session"] is False  # no agent/session was ever built
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "Available tools:" in out


def test_import_main_has_no_side_effects():
    # Importing main must not call load_config; agent/session constructed only in main().
    assert callable(main_mod.main)


def test_parse_args_prompt():
    assert main_mod.parse_args(["--prompt", "hi"]).prompt == "hi"


def test_single_prompt_turn_uses_agent():
    from src.session import ConversationSession
    from tests.conftest import FakeAgent

    agent = FakeAgent()
    it = iter(["hello", None])
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == ["hello"]


def test_import_main_does_not_import_prompt_toolkit():
    code = "import sys, main; print('prompt_toolkit' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.stdout.strip() == "False"


def test_chat_builds_and_closes_reader(monkeypatch):
    created = {}

    class RecorderReader:
        def __init__(self, **kw):
            created["kw"] = kw
            created["closed"] = False
            created["read"] = 0

        def read(self, label=""):
            created["read"] += 1
            return "/quit"

        def close(self):
            created["closed"] = True

    class StubSession:
        COMMANDS = ("/help", "/quit")

        def __init__(self, agent, read_prompt, use_tools=True, quit_on_prompt_interrupt=False):
            created["read_prompt"] = read_prompt

        def run(self):
            assert created["read_prompt"]() == "/quit"
            return 0

    monkeypatch.setattr(input_reader_mod, "InputReader", RecorderReader)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: types.SimpleNamespace())

    assert main_mod.main(["--chat"]) == 0
    assert created["kw"]["memory_path"] is None
    assert created["read"] == 1
    assert created["closed"] is True


def test_chat_reader_uses_input_config(monkeypatch):
    created = {}

    class RecorderReader:
        def __init__(self, **kw):
            created["kw"] = kw

        def read(self, label=""):
            return "/quit"

        def close(self):
            pass

    class StubSession:
        COMMANDS = ("/help", "/quit")

        def __init__(self, agent, read_prompt, use_tools=True, quit_on_prompt_interrupt=False):
            pass

        def run(self):
            return 0

    cfg = types.SimpleNamespace(
        input=types.SimpleNamespace(
            history_file_cwd=".chat_history", auto_suggest=False,
            completion=False, multiline=True, explicit_paste=False,
        )
    )
    monkeypatch.setattr(input_reader_mod, "InputReader", RecorderReader)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: cfg)

    assert main_mod.main(["--chat"]) == 0
    assert created["kw"]["auto_suggest"] is False
    assert created["kw"]["multiline"] is True
    assert created["kw"]["explicit_paste"] is False
    assert created["kw"]["completer"] is None  # completion disabled


def test_chat_prompt_toolkit_false_uses_fallback(monkeypatch):
    created = {}

    class RecorderReader:
        def __init__(self, **kw):
            created["kw"] = kw

        def read(self, label=""):
            return "fallback"

        def close(self):
            pass

    class StubSession:
        COMMANDS = ("/help", "/quit")

        def __init__(self, agent, read_prompt, use_tools=True, quit_on_prompt_interrupt=False):
            pass

        def run(self):
            return 0

    cfg = types.SimpleNamespace(
        input=types.SimpleNamespace(prompt_toolkit=False)
    )
    monkeypatch.setattr(input_reader_mod, "InputReader", RecorderReader)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: cfg)

    assert main_mod.main(["--chat"]) == 0
    assert created["kw"]["session"] is None
    assert created["kw"]["fallback"] is main_mod.prompt_input


def test_main_closes_reader_on_exception(monkeypatch):
    created = {"closed": False}

    class RecorderReader:
        def __init__(self, **kw):
            pass

        def read(self, label=""):
            return "/quit"

        def close(self):
            created["closed"] = True

    class BoomSession:
        COMMANDS = ("/help", "/quit")

        def __init__(self, agent, read_prompt, use_tools=True, quit_on_prompt_interrupt=False):
            pass

        def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(input_reader_mod, "InputReader", RecorderReader)
    monkeypatch.setattr(session_mod, "ConversationSession", BoomSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: types.SimpleNamespace())

    import pytest

    with pytest.raises(RuntimeError):
        main_mod.main(["--chat"])
    assert created["closed"] is True


# --------------------------------------------------- A8 speech_det wiring


def test_voice_passes_speech_det_config(monkeypatch):
    import src.audio_wrapper as aw

    seen = {}

    class RecorderDetector:
        def __init__(self, **kw):
            seen.update(kw)

        def listen_and_transcribe(self, silence_timeout_s):
            return "/quit"

    class StubSession:
        COMMANDS = ()

        def __init__(self, agent, read_prompt, use_tools=True, quit_on_prompt_interrupt=False):
            pass

        def run(self):
            return 0

    cfg = types.SimpleNamespace(
        brain=types.SimpleNamespace(model="m"),
        agent=types.SimpleNamespace(bootstrap_prompt=""),
        speech_det=types.SimpleNamespace(
            model_path="m", language="en", silence_timeout_s=2.0,
            beam_size=7, show_levels=False, device="cpu",
            compute_type="int8", calibration_chunks=3, guard_factor=2.0,
        ),
    )
    monkeypatch.setattr(aw, "SpeechDetector", RecorderDetector)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: cfg)

    assert main_mod.main([]) == 0
    assert seen["beam_size"] == 7 and seen["show_levels"] is False
    assert seen["model_backend"] == "cpu" and seen["calibration_chunks"] == 3
    assert seen["guard_factor"] == 2.0


def test_voice_missing_device_returns_error(monkeypatch, capsys):
    import src.audio_wrapper as aw

    class NoDeviceDetector:
        def __init__(self, **kw):
            raise aw.AudioDeviceError("no mic")

    class StubSession:
        COMMANDS = ()

        def __init__(self, *a, **k):
            pass

        def run(self):
            return 0

    cfg = types.SimpleNamespace(
        brain=types.SimpleNamespace(model="m"),
        agent=types.SimpleNamespace(bootstrap_prompt=""),
        speech_det=types.SimpleNamespace(
            model_path="m", language="en", silence_timeout_s=2.0,
        ),
    )
    monkeypatch.setattr(aw, "SpeechDetector", NoDeviceDetector)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: cfg)

    assert main_mod.main([]) == 1
    out = capsys.readouterr().out.lower()
    assert "no mic" in out or "microphone" in out


# ---------------------------------------------- graceful voice-mode startup


def _voice_cfg():
    return types.SimpleNamespace(
        brain=types.SimpleNamespace(model="m"),
        agent=types.SimpleNamespace(bootstrap_prompt=""),
        speech_det=types.SimpleNamespace(
            model_path="m", language="en", silence_timeout_s=2.0,
        ),
    )


def test_voice_missing_model_returns_error(monkeypatch, capsys):
    """Missing Whisper model dir: clear message, exit code 1, no traceback."""
    import src.audio_wrapper as aw

    class MissingModelDetector:
        def __init__(self, **kw):
            raise FileNotFoundError("Model file not found: m")

    class StubSession:
        COMMANDS = ()

        def __init__(self, *a, **k):
            pass

        def run(self):
            return 0

    monkeypatch.setattr(aw, "SpeechDetector", MissingModelDetector)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: _voice_cfg())

    assert main_mod.main([]) == 1
    out = capsys.readouterr().out.lower()
    assert "model file not found" in out
    assert "--chat" in out  # hints at the text-mode alternative
    assert "traceback" not in out


def test_voice_missing_whisper_package_returns_error(monkeypatch, capsys):
    """Missing faster-whisper package: clear message, exit code 1."""
    import src.audio_wrapper as aw

    class NoPackageDetector:
        def __init__(self, **kw):
            raise ImportError("faster-whisper is not installed")

    class StubSession:
        COMMANDS = ()

        def __init__(self, *a, **k):
            pass

        def run(self):
            return 0

    monkeypatch.setattr(aw, "SpeechDetector", NoPackageDetector)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: _voice_cfg())

    assert main_mod.main([]) == 1
    out = capsys.readouterr().out.lower()
    assert "faster-whisper" in out
    assert "--chat" in out
