# PLAN.md — Task Guide for an Automated Coding Agent

This plan restructures the user-interaction layer of `live_assistant` into **small,
independent tasks**. Each task is self-contained and must be completable without reading
the other tasks, and each is validated by unit tests you write in the same task.

---

## 0. Rules for the executing agent

1. Do **one task at a time**, in the order listed in §4. Each task is one commit.
2. For every task: write/modify tests **first or alongside** the code, then run the task's
   validation command. A task is not done until its tests pass.
3. Never run hardware or network code in tests. Do not instantiate real models, microphones,
   TTS, cameras, or Selenium. Use the fakes in §1.
4. Do not import heavy modules (`lmstudio`, `sounddevice`, `faster_whisper`, `piper`, `cv2`,
   `selenium`, `yfinance`) at the top level of any module that tests import. Import them
   lazily inside functions.
5. Keep public function signatures exactly as specified. A basic agent following this plan
   must be able to rely on them.
6. If a test cannot be written without hardware, mark the task's test as skipped with
   `@pytest.mark.skip(reason="requires <hardware>")` and state why in the commit message.
7. Do not delete `src/live_ai.py` classes unless a task explicitly says to.

---

## 1. Test strategy and conventions

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

### 1.1 FakeAgent contract (create in `tests/conftest.py`)

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

## 2. Design constraints (apply to all tasks)

- All user-facing output goes through `cprint` (`src/common.py:14`) or `rich.Console`.
  The only exception is `input()` reads.
- Text mode and voice mode share this one loop. `main.py` is the only executable entrypoint.
- `main.py` must not import `src.live_ai` or `src.audio_wrapper` at module top level; import
  them inside `main()` after argument parsing so `import main` stays side-effect free.

