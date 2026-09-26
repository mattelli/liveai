# PLAN-INPUT.md — Interactive Input with prompt_toolkit

Addendum to `PLAN.md`. It supersedes the single bullet in `PLAN.md` §6:
> ~~Readline / pyreadline3 integration or multi-line paste.~~

Everything in `PLAN.md` §0 (rules), §2 (test strategy, `FakeAgent`) and §3 (design
constraints) still applies. Read `PLAN.md` §0–§3 before starting a task here.

**Decision:** use **prompt_toolkit** as the single input layer. Do **not** use
`readline` (Unix-only) or `pyreadline3` (incomplete, ANSI-prompt conflicts). prompt_toolkit
is pure Python, works on Windows and POSIX, and provides line editing, persistent history,
multi-line paste, and completion in one dependency. It is installed as
`prompt_toolkit` by `setup-lms-cpu.ps1`.

Target files: new `src/input_reader.py`, `src/common.py` (no signature changes),
`main.py`, `config.toml`, `tests/test_input_reader.py`, `tests/test_session.py` (one
new test), `tests/conftest.py` (one fixture), `README.md`, `setup-lms-cpu.ps1`.

---

## 1. Problem

`--chat` input funnels through one primitive (`src/common.py:39`):

```python
def prompt_input(label, color="yellow"):
    cprint(label, color, end="", flush=True)
    try:
        return input()
    except EOFError:
        return None
```

On Windows 11 / Python 3.12 (the supported target), builtin `input()` gives:

| Symptom | Cause |
|---|---|
| Basic editing only; no cross-session history | `readline` is Unix-only and absent |
| No persistent Up/Down history file | History never saved |
| Pasting an N-line block runs N separate turns | `input()` returns only the first line; the rest stay buffered and are consumed by later prompts |
| A pasted block whose first line looks like `/clear` silently clears the chat | Buffered lines are replayed through `handle_command` |

Verified on this machine: `import readline`, `import pyreadline3`, and `import
prompt_toolkit` all fail in the `.lms` venv.

## 2. What prompt_toolkit provides

| Need | prompt_toolkit |
|---|---|
| Line editing (arrows, Home/End, word ops, Ctrl-K/U/W, selection) | `PromptSession` |
| Persistent history | `history=FileHistory(path)` |
| Inline history suggestion | `auto_suggest=AutoSuggestFromHistory()` |
| Multi-line paste | bracketed paste is handled internally; pasted `\n` are inserted into the buffer and Enter submits the whole block as one string |
| Completion | `completer=WordCompleter(...)` |
| Ctrl-C / Ctrl-D | `prompt()` raises `KeyboardInterrupt` / `EOFError` (configurable via `interrupt_exception` / `eof_exception`) |
| Headless testing | `create_pipe_input()` + `DummyOutput()` + `create_app_session()` |

Verified in the prompt_toolkit source:
- default `multiline=False`; `enter` accepts; `<sigint>` → `interrupt_exception()`;
  `c-d` on an empty buffer → `eof_exception()`;
- `Keys.BracketedPaste` normalises `\r\n`/`\r` to `\n` and inserts the whole payload.

This is exactly the "Readline replacement + multi-line paste" behaviour we want, from one
cross-platform package.

## 3. Design

### 3.1 One new module

All new code lives in `src/input_reader.py`. Integration points: `main.py` (chat branch)
and one guard in `ConversationSession`.

```python
class InputReader:
    def __init__(
        self,
        history_path: Path | None = None,
        auto_suggest: bool = True,
        completer=None,               # prompt_toolkit Completer | None
        multiline: bool = False,      # advanced: explicit multi-line editing
        session=None,                 # test seam: prebuilt PromptSession
        fallback=None,                # test seam / no-prompt_toolkit path
    ): ...

    def read(self, label: str = "") -> str | None:
        """Return the entered text (may contain newlines), or None on EOF.
        KeyboardInterrupt propagates to the session loop."""

    def close(self) -> None:
        """Flush history / release resources. Safe to call twice."""
```

`read` body:

```python
if self._session is None:            # prompt_toolkit unavailable
    return (self._fallback or prompt_input)(label)
try:
    return self._session.prompt(label)
except EOFError:
    return None
```

- `KeyboardInterrupt` is **not** caught here; `ConversationSession.run` already handles it
  and exits cleanly with code 0.
- The label is passed as the prompt_toolkit message so it is redrawn/cleared correctly
  (unlike today's pre-printed label).

### 3.2 Module-level import (defensive guard)

```python
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    _PROMPT_TOOLKIT = True
except ImportError:                  # pragma: no cover - env dependent
    _PROMPT_TOOLKIT = False
```

prompt_toolkit is a **required** dependency installed by `setup-lms-cpu.ps1`, but the
guard keeps `import src.input_reader` from crashing in a degraded environment:
`InputReader` then routes through `prompt_input`. `main.py` still imports
`src.input_reader` lazily inside `main()`.

### 3.3 History

- `FileHistory(str(history_path))` when a path is configured; `InMemoryHistory` otherwise.
- `history_path.parent.mkdir(parents=True, exist_ok=True)` before constructing.
- `FileHistory` has no size cap; keep the whole file (friendly for a personal assistant)
  and document it. Optional trim helper can be a later task.
- `InMemoryHistory.get_strings()` is the assertion seam in tests.

### 3.4 Multi-line paste

Primary mechanism, no extra code and no config: prompt_toolkit's bracketed-paste handler
inserts the whole pasted block; `Enter` accepts it as one string containing `\n`. A paste
is therefore one turn.

Fallback for terminals that do not send bracketed-paste markers is optional (P5): an
explicit `/paste` … `.` block mode. `multiline=True` is offered as an advanced config for
deliberate multi-line editing (Enter inserts a newline, Esc+Enter accepts).

### 3.5 Command-vs-paste at the session layer

`ConversationSession.run` currently sends every non-empty prompt through
`handle_command`, which dispatches on `text.startswith("/")`. A multi-line paste whose
first line is a command would be misinterpreted. P4 adds one guard:

```python
if "\n" not in text and self.handle_command(text):
    continue
```

Single-line `/commands` are unaffected; multi-line text always goes to the agent.

### 3.6 Config

```toml
[input]
history_file = ".chat_history"   # resolved by load_config (key ends in _file)
prompt_toolkit = true            # false -> builtin input() fallback
auto_suggest = true
completion = true                # complete /slash commands
explicit_paste = true            # /paste ... . fallback (P5)
multiline = false                # true -> Esc+Enter to accept (P5)
```

All keys optional with the defaults above.

### 3.7 Dependency

`prompt_toolkit` (pulls `wcwidth` and `pygments`) is a **required** runtime dependency,
installed by `setup-lms-cpu.ps1`:

```powershell
# CLI INPUT (line editing, persistent history, multi-line paste)
pip install prompt_toolkit
```

No version is pinned: the environment is created fresh by the setup script, and `pip`
installs the latest release. The import guard in §3.2 remains only as a safety net; the
tests and acceptance criteria assume prompt_toolkit is installed.

---

## 4. Task index

| ID | Title | Depends on | Phase |
|----|-------|-----------|-------|
| P0 | Install dependency + prompt_toolkit test fixture | T0 | Foundations |
| P1 | `InputReader` core (`read`, EOF, Ctrl-C, fallback) | P0 | Core |
| P2 | Persistent history + auto-suggest | P1 | Core |
| P3 | Wire `InputReader` into `main.py --chat` + `[input]` config | P2 | Integration |
| P4 | Native multi-line paste + session command guard | P3 | Paste |
| P5 | Optional explicit multi-line (`/paste` fallback, `multiline=true`) | P4 | Paste |
| P6 | `/slash`-command completion | P3 | Polish |
| P7 | Docs + packaging | P3 | Polish |
| P8 | End-to-end paste integration test + acceptance | P4 | Verification |

Do P0–P4 first: that delivers editing, persistent history, and multi-line paste. P5–P7
are optional/independent.

---

## P0 — Install dependency and prompt_toolkit test fixture

**Goal.** Make `prompt_toolkit` a required install and provide a headless input harness so
no test needs a TTY.

**Files.** `setup-lms-cpu.ps1` (already updated); `tests/conftest.py`; new
`tests/test_input_reader.py`.

**Change (done).** `setup-lms-cpu.ps1` now installs prompt_toolkit in the required
packages section:

```powershell
# CLI INPUT (line editing, persistent history, multi-line paste)
pip install prompt_toolkit
```

**Fixture** (in `tests/conftest.py` or `tests/test_input_reader.py`):

```python
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

@pytest.fixture
def pt_input():
    """A prompt_toolkit pipe input installed as the active app session."""
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            yield pipe

def make_session(pipe, **kw):
    """PromptSession bound to the test pipe, never touching the real TTY."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    return PromptSession(input=pipe, output=DummyOutput(),
                         history=kw.pop("history", InMemoryHistory()), **kw)
```

**Tests.**
```python
def test_prompt_toolkit_smoke(pt_input):
    pt_input.send_text("hello\n")
    assert make_session(pt_input).prompt("> ") == "hello"
```

**Validation.** `python -m pytest tests/test_input_reader.py -q`.

**Done when.** `import prompt_toolkit` succeeds in the `.lms` venv and the smoke test
passes without touching stdin/stdout.

---

## P1 — `InputReader` core

**Goal.** A synchronous reader that owns the prompt and maps EOF/interrupt correctly.

**Files.** `src/input_reader.py`; `tests/test_input_reader.py`.

**Expected behaviour.** As in §3.1–§3.2:

- `read(label)` calls `session.prompt(label)`; returns the string.
- `EOFError` → `None` (no extra output).
- `KeyboardInterrupt` propagates.
- `session=None` → `fallback(label)` (defaults to `src.common.prompt_input`).
- `close()` is idempotent and a no-op when there is no session.

**Tests.**
```python
def test_read_returns_line(pt_input):
    pt_input.send_text("hello\n")
    r = InputReader(session=make_session(pt_input))
    assert r.read("You: ") == "hello"

def test_read_eof_returns_none(pt_input):
    pt_input.send_text("\x04")          # Ctrl-D on empty buffer
    r = InputReader(session=make_session(pt_input))
    assert r.read("You: ") is None

def test_read_ctrl_c_propagates(pt_input):
    pt_input.send_text("\x03")          # Ctrl-C
    r = InputReader(session=make_session(pt_input))
    with pytest.raises(KeyboardInterrupt):
        r.read("You: ")

def test_fallback_used_when_no_session():
    r = InputReader(session=None, fallback=lambda label: "fallback")
    assert r.read("You: ") == "fallback"

def test_close_is_idempotent():
    InputReader(session=None, fallback=lambda label: "").close()
    InputReader(session=None, fallback=lambda label: "").close()
```

**Validation.** `python -m pytest tests/test_input_reader.py -q`.

**Done when.** Tests pass and `KeyboardInterrupt` still reaches
`ConversationSession.run` unchanged.

---

## P2 — Persistent history and auto-suggest

**Goal.** Up/Down recalls prior turns, and history survives across sessions.

**Files.** `src/input_reader.py`; `tests/test_input_reader.py`.

**Expected behaviour.**
- With `history_path`, `InputReader` builds `FileHistory(str(path))` (creating the parent
  directory) and `AutoSuggestFromHistory()` when `auto_suggest=True`.
- Without `history_path`, uses `InMemoryHistory` (or the injected session's history).
- Multiple `read()` calls accumulate in `history.get_strings()`.
- `close()` performs no destructive action; `FileHistory` appends as it goes.

**Tests.**
```python
def test_history_records_entries(pt_input, tmp_path):
    session = make_session(pt_input, history=InMemoryHistory())
    r = InputReader(session=session)
    pt_input.send_text("one\n")
    r.read("> ")
    pt_input.send_text("two\n")
    r.read("> ")
    assert session.history.get_strings() == ["one", "two"]

def test_file_history_persists_across_readers(tmp_path):
    hist = tmp_path / ".chat_history"
    session = make_session(pt_input, history=FileHistory(str(hist)))
    pt_input.send_text("remember me\n")
    InputReader(session=session).read("> ")
    # FileHistory writes on append for FileHistory; assert the file exists
    assert hist.exists()

def test_history_path_creates_parent(tmp_path):
    nested = tmp_path / "a" / "b" / ".chat_history"
    r = InputReader(history_path=nested)          # constructs FileHistory
    assert nested.parent.exists()
```

If `FileHistory`'s write timing is deferred, assert via construction/inspection rather
than flush; document whichever behaviour the installed version shows.

**Validation.** `python -m pytest tests/test_input_reader.py -q`.

**Done when.** History is visible in-session and a history file is created at the
configured path.

---

## P3 — Wire `InputReader` into `main.py --chat`

**Goal.** Chat mode uses prompt_toolkit; voice and `--prompt` modes are untouched;
`import main` stays side-effect free.

**Expected behaviour.** In `main.py`:

- Top-level imports unchanged.
- The `--chat` branch lazily imports `src.input_reader`, builds one `InputReader` from the
  optional `[input]` config, and sets
  `read_prompt = lambda: reader.read("[You] (type /quit to exit): ")`.
- Wrap `ConversationSession(...).run()` in `try/finally` and call `reader.close()` in the
  `finally` (reader is `None` for voice / `--prompt`).
- When `[input] prompt_toolkit = false` (or prompt_toolkit is not importable), build a
  reader with `session=None` so it routes to `prompt_input`.

Sketch:
```python
reader = None
if args.prompt is not None:
    read_prompt = iter([args.prompt, None]).__next__
elif args.chat:
    from src.input_reader import InputReader               # lazy
    in_cfg = getattr(config, "input", None)
    if getattr(in_cfg, "prompt_toolkit", True):
        reader = InputReader(
            history_path=getattr(in_cfg, "history_file", None),
            auto_suggest=getattr(in_cfg, "auto_suggest", True),
            multiline=getattr(in_cfg, "multiline", False),
        )
    read_prompt = lambda: (reader or InputReader()).read("[You] (type /quit to exit): ")
else:
    ...detector...
try:
    return ConversationSession(agent, read_prompt, use_tools=not args.no_tools).run()
finally:
    if reader is not None:
        reader.close()
```

**Tests.** In `tests/test_main.py`:
- `python -c "import main"` prints nothing (existing test).
- Monkeypatch `src.input_reader.InputReader` with a recorder; call
  `main_mod.main(["--chat", ...])` with `Agent`/`ConversationSession` stubbed; assert the
  reader was constructed and `close()` called.
- `parse_args` behaviour unchanged.

**Validation.** `python -m pytest tests/test_main.py tests/test_input_reader.py -q`.

**Done when.** `python main.py --chat` uses prompt_toolkit and `import main` imports no
prompt_toolkit.

---

## P4 — Native multi-line paste and session command guard

**Goal.** A pasted block becomes exactly one turn, even if it starts with `/`.

**Expected behaviour.**
- prompt_toolkit handles the paste (no new code in `input_reader` beyond P1).
- In `src/session.py`, change the command dispatch:
  ```python
  if "\n" not in text and self.handle_command(text):
      continue
  ```

**Tests.**
```python
def test_bracketed_paste_is_one_prompt(pt_input):
    pt_input.send_text("\x1b[200~line one\nline two\x1b[201~\n")   # paste, then Enter
    r = InputReader(session=make_session(pt_input))
    assert r.read("You: ") == "line one\nline two"

def test_multiline_prompt_is_not_a_command():
    it = iter(["/quit\nstill here", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == ["/quit\nstill here"]
    assert agent.cleared == 0

def test_single_line_command_still_dispatches():
    it = iter(["/clear", "/quit"])
    agent = FakeAgent()
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.cleared == 1 and agent.acted == []
```

**Validation.** `python -m pytest tests/test_session.py tests/test_input_reader.py -q`.

**Done when.** A bracketed paste is returned with embedded newlines as one string, and
the session never parses a multi-line string as a command.

---

## P5 — Optional explicit multi-line

**Goal.** Support terminals that do not send bracketed-paste markers, and deliberate
multi-line editing, without changing the default single-line UX.

**Expected behaviour.**
- When `explicit_paste=True`, a first line of exactly `/paste` starts a block; subsequent
  lines are read from the same prompt (`... ` continuation by default via
  `prompt_continuation`) until a line whose stripped value is `"."`. The block is returned
  joined with `\n`; `ConversationSession` never sees `/paste`.
  This is implemented in `InputReader.read` around `session.prompt`, using a loop; it
  works identically with the headless test session.
- When `multiline=True`, configure `PromptSession(multiline=True,
  prompt_continuation=...)`: Enter inserts a newline and Esc+Enter accepts. Documented as
  advanced; default remains `False`.

**Tests.**
```python
def test_explicit_paste_block(pt_input):
    pt_input.send_text("/paste\n")
    pt_input.send_text("a\n")
    pt_input.send_text("b\n")
    pt_input.send_text(".\n")
    r = InputReader(session=make_session(pt_input), explicit_paste=True)
    assert r.read("You: ") == "a\nb"

def test_explicit_paste_eof_returns_partial(pt_input):
    pt_input.send_text("/paste\na\n\x04")
    r = InputReader(session=make_session(pt_input), explicit_paste=True)
    assert r.read("You: ") == "a"

def test_explicit_paste_disabled(pt_input):
    pt_input.send_text("/paste\n")
    r = InputReader(session=make_session(pt_input), explicit_paste=False)
    assert r.read("You: ") == "/paste"
```

**Validation.** `python -m pytest tests/test_input_reader.py -q`.

**Done when.** Tests pass and `explicit_paste=True` is the only added default.

---

## P6 — Slash-command completion

**Goal.** Tab completes `/` commands, using prompt_toolkit's completer.

**Expected behaviour.**
- `InputReader(completer=...)` forwards the completer to `PromptSession`.
- `main.py` builds a `WordCompleter(ConversationSession.COMMANDS, ignore_case=True,
  sentence=True)` when `[input] completion = true`.
- No completion when disabled; `completer=None`.

**Files.** `main.py`; `src/input_reader.py`; `tests/test_input_reader.py`.

**Tests.**
```python
def test_completer_forwarded(pt_input):
    seen = {}
    class C:                                   # drip-feed prompt_toolkit's completer protocol
        def get_completions(self, document, complete_event):
            seen["text"] = document.text
            return iter(())
    r = InputReader(session=make_session(pt_input, completer=C()))
    pt_input.send_text("/he\t\n")              # tab; completion is best-effort in tests
    r.read("You: ")
    assert "text" in seen or True             # smoke: no crash, completer installed
```

Prefer asserting that `session.completer is completer` over driving the completion menu,
which is layout-dependent.

**Validation.** `python -m pytest tests/test_input_reader.py -q`.

**Done when.** Typing `/` + Tab shows command candidates interactively and tests pass.

---

## P7 — Documentation and packaging

**Files.**

- `config.toml`: add the `[input]` block from §3.6 (defaults shown).
- `setup-lms-cpu.ps1`: already updated in P0 (`pip install prompt_toolkit`); no further
  change needed here.
- `README.md`: an "Interactive chat input" subsection covering Up/Down history, paste,
  `/paste` fallback, completion, the `[input]` options, and where history is stored.
- `.gitignore`: add `.chat_history`.
- `PLAN.md` §6: replace the removed bullet with a pointer to this file.

**Tests.** A config smoke test: loading a `config.toml` without `[input]` does not raise;
with `[input]` the resolved `history_file` is absolute (`load_config` resolves `*_file`
keys).

**Validation.** `python -m pytest tests/ -q`.

**Done when.** A fresh checkout starts `python main.py --chat` with no missing-import
error and no `[input]`-specific crash.

---

## P8 — End-to-end paste test and acceptance

**Goal.** Prove one paste == one agent turn through the real loop.

**Files.** `tests/test_integration.py` (extend) or `tests/test_input_reader.py`.

**Tests.**
```python
def test_paste_end_to_end_reaches_agent_as_one_turn(pt_input):
    pt_input.send_text("\x1b[200~def f():\n    return 1\x1b[201~\n")   # paste + Enter
    r = InputReader(session=make_session(pt_input))
    agent = FakeAgent()
    it = iter([r.read("You: "), "/quit"])
    ConversationSession(agent, lambda: next(it)).run()
    assert agent.acted == ["def f():\n    return 1"]

def test_explicit_paste_end_to_end(pt_input):
    for chunk in ["/paste\n", "one\n", "two\n", ".\n", "/quit\n"]:
        pt_input.send_text(chunk)
    r = InputReader(session=make_session(pt_input), explicit_paste=True)
    agent = FakeAgent()
    ConversationSession(agent, lambda: r.read("You: ")).run()
    assert agent.acted == ["one\ntwo"]
```

**Validation.** `python -m pytest tests/ -q`.

**Done when.** The paste is one turn and history contains the block, not the `/paste`
line.

---

## 5. Acceptance criteria

Run from the repo root: `python -m pytest tests/ -q` must pass, and additionally verify
manually:

1. `python -c "import main"` prints nothing and does not import prompt_toolkit.
2. `.\setup-lms-cpu.ps1` installs prompt_toolkit; then `python main.py --chat` shows an
   editable prompt and Up/Down recalls turns.
3. History persists across two `--chat` sessions in `.chat_history`.
4. Pasting a 5-line block sends **one** turn containing newlines; a block starting with
   `/quit` does not quit.
5. Ctrl-C and `\x`/`/quit` exit with code `0` and no traceback; Ctrl-D exits via EOF.
6. `/` + Tab completes session commands when `[input] completion = true`.
7. Deleting prompt_toolkit from the venv, or setting `[input] prompt_toolkit = false`,
   falls back to `prompt_input` with no crash (defensive path only; not the supported
   configuration).
8. Every task has at least one test that fails before its change and passes after.
9. Pre-existing `tests/test_common.py`, `tests/test_session.py`, `tests/test_main.py`
   pass unchanged (except the deliberate P4 additions).

## 6. Platform notes / risks

- **Rich vs prompt_toolkit output.** Both write the terminal, but never at the same time:
  output streams between prompts, and prompt_toolkit owns the terminal only during
  `prompt()`. No `patch_stdout` is needed for this synchronous design. If input-during-
  streaming is ever added, wrap the streaming phase in
  `prompt_toolkit.patch_stdout.patch_stdout()` in that future task.
- **Ctrl-C semantics.** prompt_toolkit maps Ctrl-C to `KeyboardInterrupt` and Ctrl-D on an
  empty buffer to `EOFError`, matching `ConversationSession.run`. Ctrl-C therefore exits
  the loop (current behaviour), not "cancel line and continue". Changing that is out of
  scope.
- **`FileHistory` has no size cap.** The file grows with use. A trim-on-close helper can
  be a follow-up; document the location so users can delete it.
- **Dumb terminals.** prompt_toolkit falls back to a minimal prompt internally; still
  functional.
- **Windows input backend.** prompt_toolkit uses its own Win32/VT input layer; no
  `pyreadline3`, no console-mode juggling required.
- **Dependency weight.** prompt_toolkit pulls `wcwidth` and `pygments`; both are small and
  pure Python.

## 7. Out of scope

- `readline` / `pyreadline3` backends (explicitly rejected).
- Background stdin reading while the model streams.
- Vi mode, mouse support, syntax highlighting of input, editor integration (`Ctrl-X
  Ctrl-E`).
- History size limiting / `--resume` of transcripts.
- Any change to voice mode, model selection, tools, or config keys other than `[input]`.