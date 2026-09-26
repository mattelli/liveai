# PLAN.md — Task Guide for an Automated Coding Agent

This plan restructures the user-interaction layer of `live_assistant` into **small,
independent tasks**. Each task is self-contained and must be completable without reading
the other tasks, and each is validated by unit tests you write in the same task.

Target files: `main.py`, `main-chat.py`, `src/common.py`, and a new `src/session.py`.
Shared runtime: `src/live_ai.py`, `src/audio_wrapper.py`.

---

## 0. Rules for the executing agent

1. Do **one task at a time**, in the order listed in §4. Each task is one commit.
2. For every task: write/modify tests **first or alongside** the code, then run the task's
   validation command. A task is not done until its tests pass.
3. Never run hardware or network code in tests. Do not instantiate real models, microphones,
   TTS, cameras, or Selenium. Use the fakes in §2.
4. Do not import heavy modules (`lmstudio`, `sounddevice`, `faster_whisper`, `piper`, `cv2`,
   `selenium`, `yfinance`) at the top level of any module that tests import. Import them
   lazily inside functions.
5. Keep public function signatures exactly as specified. A basic agent following this plan
   must be able to rely on them.
6. If a test cannot be written without hardware, mark the task's test as skipped with
   `@pytest.mark.skip(reason="requires <hardware>")` and state why in the commit message.
7. Do not delete `src/live_ai.py` classes unless a task explicitly says to.

---

## 1. Current behaviour (baseline, for reference)

| Concern | `main.py` (voice) | `main-chat.py` (text) |
|---|---|---|
| Entry guard | `if __name__ == "__main__"` (main.py:19) | **none** (runs on import) |
| Entry function | `main()` (main.py:10) | top-level loop (main-chat.py:22) |
| Input | `SpeechDetector.listen_and_transcribe` | `input()` (main-chat.py:26) |
| Exit | `KeyboardInterrupt` (live_ai.py:284) | string `\x` (main-chat.py:31) |
| Interrupt handling | yes | **no** |
| Empty input | skipped (live_ai.py:280) | not handled (passed to model) |
| Tool toggle | always tools via `Agent.act` | module constant `USE_TOOLS` (main-chat.py:6) |
| Error recovery | **none** around `agent.act` | **none** |
| Commands | **none** | **none** |
| Resource cleanup | **none** | **none** |

Interaction defects:
- `main-chat.py:25` prints prompt via `cprint(..., end="")` with no `flush=True`, then `input()`
  (main-chat.py:26) — prompt can be overwritten by streamed output.
- `src/live_ai.py:239` `Agent.act` has no `try/except`; any model/tool error kills the loop.
- `src/audio_wrapper.py:105` writes live RMS with bare `print(..., end="")` to stdout.

---

## 2. Test strategy and conventions

- Test runner: **pytest**. Create `tests/` at repo root, run with
  `python -m pytest tests/ -q` from the repo root.
- Add `requirements-dev.txt` containing `pytest>=8`.
- `src` is already a package (`src/__init__.py` exists), so tests import
  `from src.session import ConversationSession`.
- No real agent/model is used. A `FakeAgent` (defined in the plan as
  `tests/conftest.py`) stands in for `Agent`. `ConversationSession` must accept its
  collaborators by **dependency injection** so no heavy import is required.
- Standard library only inside `src/session.py` and `src/common.py` (plus `rich` already
  used by `common.py`).
- Use pytest `capsys` to assert console output and `monkeypatch` to inject failures.

### 2.1 FakeAgent contract (create in `tests/conftest.py`)

```python
class Frag:
    def __init__(self, content): self.content = content

class FakeAgent:
    def __init__(self):
        self.acted, self.printed, self.speech = [], [], []
        self.cleared = 0
        self.flush_count = 0
    def act(self, prompt): self.acted.append(prompt)
    def model_fragments(self, prompt): return [Frag("stream")]   # tools-off path
    def print_fragment(self, frag): self.printed.append(frag.content)
    def flush_stream(self): self.flush_count += 1
    def clear_chat(self): self.cleared += 1
    def allow_speech(self, enable): self.speech.append(enable)
    def model_info(self): return "fake-model-info"
```

### 2.2 Session contract to build (used by all session tests)

```python
class ConversationSession:
    def __init__(self, agent, read_prompt, use_tools=True):
        '''
        agent: object exposing act/model_fragments/print_fragment/flush_stream/
               clear_chat/allow_speech/model_info (FakeAgent in tests)
        read_prompt: Callable[[], str | None]; None means EOF
        use_tools: bool
        '''
    should_quit: bool
    use_tools: bool
    def run(self) -> int: ...            # returns process exit code (0)
    def handle_command(self, text: str) -> bool: ...  # True if consumed as a command
    def banner(self) -> None: ...
    def shutdown(self) -> int: ...
```

`run()` algorithm (must match):
```
banner()
while not should_quit:
    prompt = read_prompt()            # may raise KeyboardInterrupt
    if prompt is None: break          # EOF
    text = prompt.strip()
    if text == "": continue           # skip empty
    if handle_command(text): continue
    try:
        if use_tools: agent.act(text)
        else:
            for f in agent.model_fragments(text): agent.print_fragment(f)
    except KeyboardInterrupt: should_quit = True
    except Exception as exc: cprint(f"❌ {exc}", "red")
    finally: agent.flush_stream()
return shutdown()
```

---

## 3. Design constraints (apply to all tasks)

- All user-facing output goes through `cprint` (`src/common.py:14`) or `rich.Console`.
  The only exception is `input()` reads.
- Text mode and voice mode share this one loop. `main.py` is the only executable entrypoint.
- `main.py` must not import `src.live_ai` or `src.audio_wrapper` at module top level; import
  them inside `main()` after argument parsing so `import main` stays side-effect free.

---

## 4. Task index

| ID | Title | Depends on | Phase |
|----|-------|-----------|-------|
| T0 | Bootstrap pytest + fakes | — | Foundations |
| T1 | Add flush-safe `prompt_input` | T0 | Foundations |
| T2 | Harden `cprint`/`StreamPrinter` flush semantics | T0 | Foundations |
| T3 | Create `ConversationSession` core loop (quit/EOF/empty) | T1 | Session |
| T4 | Add `/help` command | T3 | Session |
| T5 | Add `/clear` command | T3 | Session |
| T6 | Add `/voice on\|off` command | T3 | Session |
| T7 | Add `/tools on\|off` command | T3 | Session |
| T8 | Add `/info` command | T3 | Session |
| T9 | Unknown `/` command handling | T3 | Session |
| T10 | Per-turn exception recovery | T3 | Session |
| T11 | KeyboardInterrupt handling in the loop | T3 | Session |
| T12 | `\x`/`exit` aliases for `/quit` | T3 | Session |
| T13 | `shutdown()` releases resources | T3 | Session |
| T14 | Rewrite `main.py` with argparse + DI | T3 | Entrypoints |
| T15 | Convert `main-chat.py` to a guarded shim | T14 | Entrypoints |
| T16 | Non-interactive `--prompt "..."` mode | T14 | Entrypoints |
| T17 | End-to-end session integration test | T14 | Verification |
| T18 | Remove raw `print()` from `SpeechDetector` | T3 | Output polish |
| T19 | Move bootstrap greeting out of `Agent.__init__` | T3 | Output polish |

Phases should be done in order. Tasks within a phase may be done in any order unless a
dependency is listed.

---

## T0 — Bootstrap pytest and test fakes

**Goal.** Give the repo a working test setup and a shared `FakeAgent`.

**Current behaviour.** No `tests/` directory, no pytest configuration, no fakes.

**Expected behaviour.**
- `tests/__init__.py` exists (empty).
- `tests/conftest.py` defines `Frag` and `FakeAgent` exactly as in §2.1 and exposes them as
  fixtures:
  ```python
  import pytest
  from tests.conftest import FakeAgent, Frag   # or define locally

  @pytest.fixture
  def fake_agent(): return FakeAgent()

  @pytest.fixture
  def prompts():
      """Returns a helper: prompts(["a", None]) -> a read_prompt callable."""
      def _make(values):
          it = iter(values)
          return lambda: next(it)
      return _make
  ```
- `requirements-dev.txt` contains `pytest>=8`.
- `tests/test_smoke.py` contains `def test_fake_agent_contract(fake_agent)` asserting
  `fake_agent.act("x")` appends and `fake_agent.model_info() == "fake-model-info"`.

**Files.** `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`,
`requirements-dev.txt`.

**Validation.**
```
python -m pytest tests/test_smoke.py -q
```
Expected: 1 passed.

**Done when.** The validation command passes and no heavy module is imported by the tests.

---

## T1 — Add flush-safe `prompt_input`

**Goal.** Provide one helper that renders a prompt and reads a line reliably.

**Current behaviour.** `main-chat.py:25-26` does `cprint(..., end="")` then `input()`
without flushing; `EOFError` is caught at the loop level (main-chat.py:27).

**Expected behaviour.** In `src/common.py` add:
```python
def prompt_input(label: str, color: Colors = "yellow") -> str | None:
    """Write label (flushed, no newline) and read a line.
    Returns the raw line (without trailing newline), or None on EOFError."""
```
- Uses `cprint(label, color, end="", flush=True)`.
- Calls builtin `input()`.
- Returns `None` on `EOFError` (does **not** print anything on EOF).

**Files.** `src/common.py`; new `tests/test_common.py`.

**Tests.** In `tests/test_common.py`:
```python
from src.common import prompt_input

def test_prompt_input_returns_line(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "hello")
    assert prompt_input("You: ") == "hello"

def test_prompt_input_none_on_eof(monkeypatch):
    def boom(): raise EOFError
    monkeypatch.setattr("builtins.input", boom)
    assert prompt_input("You: ") is None

def test_prompt_input_flushes_and_keeps_label(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "x")
    prompt_input("You: ")
    out = capsys.readouterr().out
    assert "You: " in out
```

**Validation.** `python -m pytest tests/test_common.py -q` → 3 passed.

**Done when.** Tests pass; `prompt_input` is the only place `input()` is called in new code.

---

## T2 — Harden `cprint` / `StreamPrinter` flush semantics

**Goal.** Guarantee buffered stream text is always emitted and never bleeds into the next
prompt.

**Current behaviour.** `StreamPrinter.flush` (`src/common.py:88`) exists but has no test.
A partial line is printed by `_process_line` and `_buffer` reset. Unterminated code fences
are closed. There is no test proving this.

**Expected behaviour.**
- Keep the existing implementation. Add tests that lock the contract:
  - feeding `"partial"` (no newline) then `flush()` prints `partial` once.
  - feeding `` "```py\nx=1" `` then `flush()` does not raise and clears the code buffer.
  - calling `flush()` twice is a no-op the second time.
- `cprint(..., flush=True)` must forward `flush=True` to `print`. (Already true; add a test.)

**Files.** `tests/test_common.py` (extend); `src/common.py` only if a test exposes a bug.

**Tests.** Add:
```python
import io
from rich.console import Console
from src.common import StreamPrinter, cprint

def test_stream_printer_flush_partial(capsys):
    sp = StreamPrinter(); sp.feed("partial"); sp.flush()
    assert "partial" in capsys.readouterr().out

def test_stream_printer_flush_idempotent(capsys):
    sp = StreamPrinter(); sp.feed("a"); sp.flush(); sp.flush()
    assert capsys.readouterr().out.count("a") == 1

def test_stream_printer_unterminated_code_block(capsys):
    sp = StreamPrinter(); sp.feed("```py\nx=1"); sp.flush()
    # must not raise and must emit the code text
    assert "x=1" in capsys.readouterr().out
```
Note: `StreamPrinter` uses `rich.Console()`; if `capsys` does not capture rich output,
instead monkeypatch `sp.console.print` to append to a list and assert on that list. Prefer
the capture-list approach to keep the test deterministic.

**Validation.** `python -m pytest tests/test_common.py -q` → 6 passed.

**Done when.** All tests pass and `StreamPrinter.flush` is safe to call on any error path.

---

## T3 — Create `ConversationSession` core loop

**Goal.** One loop that owns prompt acquisition, empty-input skipping, quit, and shutdown.

**Current behaviour.** Two divergent loops: `src/live_ai.py:270` (`LiveAi.run`) and
`main-chat.py:22`. No shared session object, no dependency injection.

**Expected behaviour.** New file `src/session.py` implementing the contract in §2.2.
For this task implement only:
- `__init__(self, agent, read_prompt, use_tools=True)` storing all three, plus
  `self.should_quit = False`, `self._iteration = 0`.
- `banner()` printing one line via `cprint` that contains the model-independent text
  `"Type /help for commands"`.
- `handle_command(text)` returning `False` for everything except `/quit` and `\x`
  (both set `should_quit = True` and return `True`). Unknown `/...` returns `False` for now.
- `run()` exactly per the algorithm in §2.2.
- `shutdown()` returning `0` and printing `"👋 Bye!"` via `cprint`. Calls
  `agent.flush_stream()` once if the agent has it.

**Files.** `src/session.py` (new); `tests/test_session.py` (new).

**Tests.**
```python
from src.session import ConversationSession
from tests.conftest import FakeAgent

def make(values, use_tools=True):
    it = iter(values)
    return ConversationSession(FakeAgent(), lambda: next(it), use_tools=use_tools)

def test_run_dispatches_to_agent(capsys):
    it = iter(["hello", "/quit"])
    agent = FakeAgent()
    s = ConversationSession(agent, lambda: next(it))
    assert s.run() == 0
    assert agent.acted == ["hello"]
    assert agent.flush_count == 1

def test_run_skips_empty_input():
    it = iter(["   ", "hi", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == ["hi"]

def test_run_stops_on_eof():
    it = iter([None])
    agent = FakeAgent()
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == []

def test_tools_off_uses_stream_path():
    it = iter(["hi", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it), use_tools=False).run()
    assert agent.acted == [] and agent.printed == ["stream"]

def test_banner_mentions_help(capsys):
    it = iter(["/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).banner()
    assert "/help" in capsys.readouterr().out
```

**Validation.** `python -m pytest tests/test_session.py -q` → 5 passed.

**Done when.** Tests pass; `src/session.py` imports only `src.common` and stdlib.

---

## T4 — `/help` command

**Goal.** Make commands discoverable.

**Current behaviour.** No command surface anywhere.

**Expected behaviour.** `handle_command("/help")` prints one line per available command,
each prefixed with the command name, and returns `True`. Must list at least:
`/help`, `/quit`, `/clear`, `/voice`, `/tools`, `/info`. Does not call `agent.act`.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_help_lists_commands(capsys):
    s = make(["/help", "/quit"])
    s.run()
    out = capsys.readouterr().out
    for cmd in ["/help", "/quit", "/clear", "/voice", "/tools", "/info"]:
        assert cmd in out
```
Also assert `agent.acted == []`.

**Validation.** `python -m pytest tests/test_session.py -q` → all pass.

**Done when.** `/help` is handled locally and never forwarded to the model.

---

## T5 — `/clear` command

**Goal.** Let the user clear the conversation without the LLM deciding to.

**Current behaviour.** `Brain.clear_chat` (`src/live_ai.py:100`) is only reachable as a tool.

**Expected behaviour.** `handle_command("/clear")` calls `agent.clear_chat()` if the agent
has it, prints `"Chat cleared"` via `cprint`, returns `True`. If the agent lacks the method,
print a grey notice and still return `True`.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_clear_calls_agent():
    it = iter(["/clear", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.cleared == 1
    assert agent.acted == []

class NoClear:
    def flush_stream(self): pass
def test_clear_without_support(capsys):
    it = iter(["/clear", "/quit"])
    ConversationSession(NoClear(), lambda: next(it)).run()
    assert "not supported" in capsys.readouterr().out.lower()
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass.

---

## T6 — `/voice on|off` command

**Goal.** Toggle speech at runtime from the prompt.

**Current behaviour.** Speech is controlled only by the LLM tool `allow_speech`
(`src/live_ai.py:140`).

**Expected behaviour.**
- `/voice on` → `agent.allow_speech(True)`, print `"Voice enabled"` (green), return `True`.
- `/voice off` → `agent.allow_speech(False)`, print `"Voice disabled"`, return `True`.
- `/voice` or other argument → print usage `"Usage: /voice on|off"` (yellow), return `True`,
  do **not** call `allow_speech`.
- Case-insensitive argument.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_voice_on_off():
    it = iter(["/voice on", "/voice OFF", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == [True, False]

def test_voice_usage(capsys):
    it = iter(["/voice maybe", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == []
    assert "usage" in capsys.readouterr().out.lower()
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass.

---

## T7 — `/tools on|off` command

**Goal.** Replace the module-level `USE_TOOLS` constant with a runtime toggle.

**Current behaviour.** `main-chat.py:6` sets `USE_TOOLS` once; no way to change it while
running.

**Expected behaviour.**
- `/tools on` → `self.use_tools = True`; `/tools off` → `False`; print the new state.
- `/tools` with no/unknown argument → print `"Usage: /tools on|off"`, no change.
- After `/tools off`, the next normal prompt uses `agent.model_fragments` instead of
  `agent.act`.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_tools_toggle_runtime():
    it = iter(["/tools off", "hello", "/tools on", "world", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    # first prompt streamed, second acted
    assert agent.printed == ["stream"]
    assert agent.acted == ["world"]

def test_tools_usage(capsys):
    it = iter(["/tools", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "usage" in capsys.readouterr().out.lower()
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass and `USE_TOOLS` is no longer referenced by new code.

---

## T8 — `/info` command

**Goal.** Show model/context info on demand.

**Current behaviour.** `Agent.model_info` (`src/live_ai.py:119`) is tool-only.

**Expected behaviour.** `handle_command("/info")` prints the return of `agent.model_info()`
via `cprint`, returns `True`. If the agent lacks `model_info`, print `"Info unavailable"`.
Never call `agent.act`.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_info_prints_model_info(capsys):
    it = iter(["/info", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "fake-model-info" in capsys.readouterr().out
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass.

---

## T9 — Unknown `/` command handling

**Goal.** Stop typos from being sent to the model.

**Current behaviour.** Any text, including `/foo`, is passed to the model.

**Expected behaviour.** If `text.startswith("/")` and no command matches,
`handle_command` prints `"Unknown command: <text>. Type /help."` (red) and returns `True`.
`agent.act` must not be called.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_unknown_command_not_sent():
    it = iter(["/frobnicate", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == []

def test_unknown_command_message(capsys):
    it = iter(["/frobnicate", "/quit"])
    ConversationSession(FakeAgent(), lambda: next(it)).run()
    assert "unknown command" in capsys.readouterr().out.lower()
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass.

---

## T10 — Per-turn exception recovery

**Goal.** One failing turn must not end the session.

**Current behaviour.** `Agent.act` (`src/live_ai.py:239`) and both entrypoint loops have no
`try/except`; any exception propagates and crashes.

**Expected behaviour.** `run()` (already sketched in §2.2) catches `Exception` from the turn,
prints `❌ <message>` in red via `cprint`, calls `agent.flush_stream()` in `finally`, and
continues to the next prompt. The session returns `0` at the end.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_turn_error_is_recovered(capsys):
    class Boom(FakeAgent):
        def __init__(self): super().__init__(); self.calls = 0
        def act(self, prompt):
            self.calls += 1
            if self.calls == 1: raise RuntimeError("kaboom")
            super().act(prompt)
    agent = Boom()
    it = iter(["bad", "good", "/quit"])
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == ["good"]
    assert "kaboom" in capsys.readouterr().out
    assert agent.flush_count == 3   # flushed every turn, including the failed one
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass.

---

## T11 — KeyboardInterrupt handling in the loop

**Goal.** `Ctrl+C` anywhere in the loop exits cleanly with the same farewell as `/quit`.

**Current behaviour.** Only `LiveAi.run` (`src/live_ai.py:284`) catches `KeyboardInterrupt`;
text mode (`main-chat.py`) does not.

**Expected behaviour.**
- `KeyboardInterrupt` raised by `read_prompt()` → `run()` stops and returns `0`.
- `KeyboardInterrupt` raised inside a turn → caught by the turn handler, `should_quit` set,
  loop exits, `shutdown()` runs.
- No traceback is printed.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_ctrl_c_on_prompt_exits():
    def reader(): raise KeyboardInterrupt
    agent = FakeAgent()
    assert ConversationSession(agent, reader).run() == 0

def test_ctrl_c_during_turn_exits(capsys):
    class Interrupting(FakeAgent):
        def act(self, prompt): raise KeyboardInterrupt
    it = iter(["hi"])
    counter = {"n": 0}
    def reader():
        counter["n"] += 1
        return "hi" if counter["n"] == 1 else "/quit"
    agent = Interrupting()
    assert ConversationSession(agent, reader).run() == 0
    assert "Bye" in capsys.readouterr().out
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass and `run()` never lets `KeyboardInterrupt` escape.

---

## T12 — `\x` and `exit` aliases for `/quit`

**Goal.** Preserve the old `main-chat.py` exit string while unifying the command surface.

**Current behaviour.** `main-chat.py:31` compares `user_input.strip() == "\\x"`.

**Expected behaviour.** `handle_command` treats, case-insensitively and stripped:
`"/quit"`, `"exit"`, and `"\\x"` as quit. Each sets `self.should_quit = True` and returns
`True`. (Note: in source this is a backslash + `x`, two characters.)

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
import pytest
@pytest.mark.parametrize("token", ["/quit", "exit", "EXIT", r"\x"])
def test_quit_aliases(token):
    it = iter([token])
    agent = FakeAgent()
    assert ConversationSession(agent, lambda: next(it)).run() == 0
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** All aliases exit with code `0`.

---

## T13 — `shutdown()` releases resources

**Goal.** Always flush/release before exit.

**Current behaviour.** Neither entrypoint releases the mic, TTS, or model on exit.

**Expected behaviour.** `shutdown()`:
1. calls `agent.flush_stream()` if present,
2. calls `agent.allow_speech(False)` if present **and** speech was enabled earlier via
   `/voice on` (track `self._speech_enabled`),
3. prints `"👋 Bye!"` via `cprint`,
4. returns `0`.
It must be safe to call twice.

**Files.** `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_shutdown_disables_voice_if_enabled():
    it = iter(["/voice on", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == [True, False]

def test_shutdown_no_voice_call_when_never_enabled():
    it = iter(["/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.speech == []

def test_shutdown_idempotent():
    s = ConversationSession(FakeAgent(), lambda: "/quit")
    assert s.shutdown() == 0 and s.shutdown() == 0
```

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass.

---

## T14 — Rewrite `main.py` with argparse and dependency injection

**Goal.** Single entrypoint; `import main` is side-effect free.

**Current behaviour.** `main.py` imports `src.live_ai` at top (main.py:3), defines `main()`
(main.py:10), and only builds `LiveAi`. `main-chat.py` duplicates config + loop.

**Expected behaviour.**
```python
WORK_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = WORK_DIR / "config.toml"

def parse_args(argv=None) -> argparse.Namespace:
    # --chat (store_true)
    # --no-tools (store_true)
    # --config PATH (default CONFIG_FILE)
    # --prompt TEXT (default None)

def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(...)
    config = load_config(args.config)
    from src.live_ai import Agent          # lazy import
    agent = Agent(config=config)
    if args.chat:
        read_prompt = lambda: prompt_input(f"[{...}] You (type /quit to exit): ")
    else:
        from src.audio_wrapper import SpeechDetector   # lazy import
        detector = SpeechDetector(
            model_path=config.speech_det.model_path,
            language=config.speech_det.language,
        )
        read_prompt = lambda: detector.listen_and_transcribe(
            silence_timeout_s=config.speech_det.silence_timeout_s
        )
    from src.session import ConversationSession
    return ConversationSession(agent, read_prompt, use_tools=not args.no_tools).run()

if __name__ == "__main__":
    raise SystemExit(main())
```
Requirements:
- Top-level imports of `main.py`: `logging`, `argparse`, `pathlib`, `src.common`
  (`load_config`, `prompt_input`). **No** `src.live_ai`, `src.audio_wrapper`, `src.session`
  at module top level.
- `parse_args([])` defaults: `chat is False`, `no_tools is False`,
  `config == CONFIG_FILE`, `prompt is None`.

**Files.** `main.py`; new `tests/test_main.py`.

**Tests.**
```python
import importlib
import sys
import main as main_mod
from pathlib import Path

def test_parse_args_defaults():
    a = main_mod.parse_args([])
    assert a.chat is False and a.no_tools is False
    assert a.config == main_mod.CONFIG_FILE and a.prompt is None

def test_parse_args_flags():
    a = main_mod.parse_args(["--chat", "--no-tools", "--config", "x.toml"])
    assert a.chat and a.no_tools and str(a.config).endswith("x.toml")

def test_import_main_has_no_side_effects(monkeypatch):
    monkeypatch.setattr(main_mod, "load_config",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("imported")))
    # importing must not call load_config; agent/session constructed only in main()
    assert callable(main_mod.main)
```
Note: `test_parse_args_flags` asserts on the namespace only; the actual file need not exist
because `parse_args` must not read it.

**Validation.** `python -m pytest tests/test_main.py -q`.

**Done when.** `python -c "import main"` runs with no output and no model load, and tests pass.

---

## T15 — Convert `main-chat.py` to a guarded shim

**Goal.** Preserve `python main-chat.py` as text mode without duplicating logic.

**Current behaviour.** Top-level code runs on import (main-chat.py:10-43); defines
`USE_TOOLS` and its own loop.

**Expected behaviour.** `main-chat.py` becomes:
```python
"""Deprecated shim. Use: python main.py --chat"""
from main import main

if __name__ == "__main__":
    raise SystemExit(main(["--chat"]))
```
No top-level agent construction, no `load_config` at import, no loop.

**Files.** `main-chat.py`; `tests/test_main.py`.

**Tests.** Add to `tests/test_main.py`:
```python
def test_main_chat_import_is_side_effect_free(monkeypatch):
    for name in list(sys.modules):
        if name == "main-chat" or name.startswith("main-chat"):
            del sys.modules[name]
    import importlib.util, pathlib
    p = pathlib.Path("main-chat.py")
    spec = importlib.util.spec_from_file_location("main_chat_shim", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # must not load config/agent
    assert callable(mod.main)
```
(The test uses a distinct module name because module names cannot contain `-`.)

**Validation.** `python -m pytest tests/test_main.py -q`.

**Done when.** `python -c "import importlib.util,..."`-style import of the shim does not
trigger config loading and tests pass.

---

## T16 — Non-interactive `--prompt "..."` mode

**Goal.** Allow scripting and smoke tests without stdin.

**Current behaviour.** Not supported by either script.

**Expected behaviour.** When `args.prompt` is set, `main()` runs exactly one turn and exits:
- if `args.prompt.strip()` starts with `/`, dispatch it as a command via
  `session.handle_command(...)`;
- otherwise dispatch via the normal turn logic (respecting `use_tools`);
- return the session's exit code; never call `read_prompt`.
Implement by giving `ConversationSession` an optional `predetermined: list[str] | None`
in the constructor **or** by constructing `read_prompt` from the list:
`read_prompt = iter([args.prompt, None]).__next__`. Prefer the latter (no new session API).

**Files.** `main.py`; `tests/test_main.py`.

**Tests.** Use `ConversationSession` with a list-backed reader (DI, no real agent):
```python
def test_single_prompt_turn_uses_agent():
    from src.session import ConversationSession
    from tests.conftest import FakeAgent
    agent = FakeAgent()
    it = iter(["hello", None])
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == ["hello"]
```
Plus a `parse_args` test: `main_mod.parse_args(["--prompt", "hi"]).prompt == "hi"`.

**Validation.** `python -m pytest tests/test_main.py -q`.

**Done when.** Tests pass. (A real end-to-end CLI run is covered in T17 with mocked imports.)

---

## T17 — End-to-end session integration test

**Goal.** Prove the whole interaction loop works with a fake agent and a scripted transcript.

**Current behaviour.** No integration coverage.

**Expected behaviour.** One test drives a realistic transcript through `ConversationSession`
and asserts every user-visible effect:

**Files.** new `tests/test_integration.py`.

**Tests.**
```python
from src.session import ConversationSession
from tests.conftest import FakeAgent

TRANSCRIPT = [
    "/help",
    "/voice on",
    "summarise the plan",
    "/tools off",
    "just stream this",
    "/clear",
    "/info",
    "/tools on",
    "final answer",
    "/quit",
]

def test_full_session_transcript(capsys):
    it = iter(TRANSCRIPT)
    agent = FakeAgent()
    session = ConversationSession(agent, lambda: next(it))
    assert session.run() == 0

    assert agent.acted == ["summarise the plan", "final answer"]
    assert agent.printed == ["stream"]
    assert agent.speech == [True, False]     # enabled then disabled on shutdown
    assert agent.cleared == 1
    assert agent.flush_count == len([t for t in TRANSCRIPT if not t.startswith("/")]) + \
           len([t for t in TRANSCRIPT if t.startswith("/") and t.split()[0] in
                {"/help", "/voice", "/tools", "/clear", "/info"}])
    out = capsys.readouterr().out
    assert "fake-model-info" in out
```
If the exact `flush_count` assertion proves brittle, assert `agent.flush_count >= 9` and add
a comment. Do not weaken the `acted`/`printed`/`speech`/`cleared` assertions.

**Validation.** `python -m pytest tests/test_integration.py -q`.

**Done when.** Test passes.

---

## T18 — Remove raw `print()` from `SpeechDetector`

**Goal.** Keep the terminal layout intact in voice mode.

**Current behaviour.** `src/audio_wrapper.py:105` prints
`\rIN-Audio RMS: ...` directly to stdout inside `listen()`.

**Expected behaviour.**
- Add a constructor parameter `show_levels: bool = False` to `SpeechDetector`
  (`src/audio_wrapper.py`, `__init__`).
- When `show_levels` is `False` (default) the RMS line is **not** printed.
- When `True`, the RMS line must go through `cprint(..., end="", flush=True)` and be cleared
  (print `"\r" + " "*40 + "\r"`) before returning.
- Do not import `common.cprint` at module top if it causes a cycle; import inside the method
  is acceptable.
- `main.py` (T14) may pass `show_levels=True` for interactive voice mode.

**Files.** `src/audio_wrapper.py`; new `tests/test_audio_wrapper.py`.

**Tests.** Do not open a microphone. Test the level-formatting helper in isolation:
```python
from src.audio_wrapper import format_level

def test_format_level_hidden():
    assert format_level(0.5, 0.1, show=False) == ""

def test_format_level_shown():
    s = format_level(0.5, 0.1, show=True)
    assert "0.500" in s and "0.100" in s
```
Refactor the RMS string building out of `listen()` into a pure
`format_level(rms: float, threshold: float, show: bool) -> str` so it is testable.

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`.

**Done when.** Tests pass and `listen()` contains no bare `print(`.

> Note: `audio_wrapper.py` imports `sounddevice` etc. at module top, which may not be
> installed. If import fails, guard the test module with
> `pytest.importorskip("sounddevice")` and document the skip.

---

## T19 — Move the bootstrap greeting out of `Agent.__init__`

**Goal.** The greeting should be emitted by the session loop, after the banner, not during
agent construction.

**Current behaviour.** `Agent.__init__` calls `self.act(...)` at `src/live_ai.py:216-218`,
so the model greets before the caller's loop starts (and races mic calibration).

**Expected behaviour.**
- Remove the `if self.config.agent.bootstrap_prompt: self.act(...)` block from
  `Agent.__init__`.
- Add `def bootstrap(self) -> None` to `Agent` that calls `self.act(bootstrap_prompt)` when
  the prompt is configured.
- `ConversationSession.run()` calls `agent.bootstrap()` **once, after `banner()`**, if the
  agent has the method (use `getattr(agent, "bootstrap", None)`).

**Files.** `src/live_ai.py`; `src/session.py`; `tests/test_session.py`.

**Tests.**
```python
def test_session_calls_bootstrap_once(capsys):
    class B(FakeAgent):
        def __init__(self): super().__init__(); self.n = 0
        def bootstrap(self): self.n += 1
    it = iter(["hi", "/quit"])
    agent = B()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.n == 1

def test_session_without_bootstrap_ok():
    it = iter(["hi", "/quit"])          # FakeAgent has no bootstrap()
    assert ConversationSession(FakeAgent(), lambda: next(it)).run() == 0
```
`test_session_calls_bootstrap_once` must also assert the bootstrap turn is **not** counted as
`agent.acted` if `bootstrap` is implemented to call a distinct callback. If `Agent.bootstrap`
reuses `act`, assert `agent.acted == ["greeting", "hi"]` instead. Pick one convention and
state it in the test docstring.

**Validation.** `python -m pytest tests/test_session.py -q`.

**Done when.** Tests pass and constructing an `Agent` no longer produces output.

---

## 5. Global acceptance criteria

Run from the repo root:

```
python -m pytest tests/ -q
```

Must pass, and additionally verify manually:

1. `python -c "import main"` prints nothing and loads no model.
2. `python -c "import importlib.util,pathlib; ..."` of `main-chat.py` (T15) loads no config.
3. `python main.py --chat` behaves like the old `main-chat.py`; `Ctrl+C` and `\x` both exit
   with code `0` and no traceback.
4. `python main.py` still runs the voice loop, now via `ConversationSession`.
5. `/help`, `/quit`, `/clear`, `/voice`, `/tools`, `/info`, and unknown-command handling work
   in both modes.
6. A turn that raises prints one red line and returns to the prompt.
7. No bare `print(` remains in `src/session.py`, `main.py`, `main-chat.py`, or the
   interaction path of `src/audio_wrapper.py` (T18).
8. Every task above has at least one test that fails before the change and passes after.

---

## 6. Out of scope (do not implement under this plan)

- Session transcript persistence / `--resume`.
- Readline / pyreadline3 integration or multi-line paste.
- Removing `LiveAi` entirely (T14 leaves it untouched; `main.py` stops using it).
- Any change to model selection, tools, or `config.toml` semantics.
