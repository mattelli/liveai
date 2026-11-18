# PLAN-CANCEL.md — Reliable generation interruption (chat mode, Ctrl+C)

Status: proposed (not implemented)
Scope: chat mode (`--chat`), `Ctrl+C` handling. Voice mode / barge-in is out of scope.

## Goals

1. `Ctrl+C` during a chat turn stops the response **immediately** — including
   during server prefill, when no token has arrived yet.
2. A cancel is reported as a cancel (`⏹ generation cancelled`), never as an
   error (`❌ ...`).
3. `Ctrl+C` while the answer is being **spoken** silences the audio only; the
   **next** `Ctrl+C` cancels the generation.
4. On cancel the **partial answer is discarded** (not kept in history).
5. A cancel **kills an already-running tool** (e.g. a long `run_command`), not
   just queued ones.
6. The chat session always survives a cancel and stays usable.

## Current behavior

| # | Location | Behavior today |
|---|---|---|
| 1 | `src/session.py:47-78` `_generation_sigint_guard` | Swaps `SIGINT` -> `agent.cancel_generation()` for the duration of a turn. |
| 2 | `src/live_ai.py:1851` `cancel_generation` | Sets `_cancel_requested = True`, closes `self._active_stream` (best effort). |
| 3 | `src/live_ai.py:2380-2381`, `1811-1812`, `2574-2575` | `_active_stream` is assigned **after** `_stream_completion()` returns -> during prefill there is nothing to close; the flag is only read **between chunks** (`2389`, `1816`, `2579`). |
| 4 | `src/session.py:160-165` | A stream aborted by `close()` raises an httpx/OpenAI exception -> caught by `except Exception` -> prints `❌ ...`; the `finally` *also* prints "generation cancelled". |
| 5 | `src/tools/speech.py:113-139` `_speech_sigint_guard` | Nested **over** the generation guard -> while audio plays, `Ctrl+C` only silences speech; generation keeps running. |
| 6 | `src/live_ai.py:2743-2748`, `1832-1836` | Partial answer is **kept** in history (plus `CANCELLED_PLACEHOLDER` when empty). |
| 7 | `src/live_ai.py:2526-2529` | Only *queued* tools are skipped; a running `run_command` (`src/tools/system.py:78-98`) keeps running until its own timeout. |
| 8 | `src/live_ai.py:712` | `OpenAI(base_url=..., api_key=...)` — no client timeout. |
| 9 | `src/live_ai.py:2708` | `act()` resets `_cancel_requested = False` at turn start (a cancel between turns is silently dropped). |

### Root causes (summary)

1. **`_active_stream` is registered too late** — during prefill the loop is
   blocked in the socket read; `close()` is what aborts it, and that window has
   no stream to close.
2. **The abort surfaces as an error, not a cancel** — closing the response
   mid-read raises an httpx/OpenAI exception, which the session reports as a
   failure.
3. **Speech steals `Ctrl+C`** — the speech guard is nested over the generation
   guard, so the most visible symptom is "the generation keeps running".
4. **No client timeout** — a hung stream can block indefinitely with nothing to
   interrupt.
5. **Running tools are not stopped** — only queued ones are skipped.
6. **Minor** — `act()` clears the cancel flag at turn start, so a cancel
   arriving between turns is dropped; the "cancelled" banner can also print for
   a turn that actually completed.

## Expected behavior

1. `Ctrl+C` during prefill **or** streaming aborts within ~a few hundred ms;
   the spinner stops; the prompt returns.
2. Output on cancel: `⏹ generation cancelled` (yellow) — no `❌`, no traceback.
3. Speech playing: 1st `Ctrl+C` -> `🔇 Speech stopped`, generation continues;
   2nd `Ctrl+C` -> generation cancelled.
4. History after a cancel: the partial answer is **removed**; the turn leaves no
   dangling `user`/`tool` messages (handled by the existing history repair).
5. A running `run_command` / `python_eval` subprocess is terminated on cancel;
   its tool result is reported as cancelled.
6. The session continues to the next prompt; `/quit`, `/clear` etc. still work.

## Design (shared by all tasks)

- **Cancel token**: a `threading.Event` on the agent (`self._cancel_event`)
  replaces the plain `_cancel_requested` bool as the source of truth;
  `cancel_requested` (property) keeps working for existing callers/tests.
  Set/clear is idempotent and thread-safe (signal handler, listener thread,
  tool thread).
- **`GenerationCancelled` exception** (subclass of `Exception`, exported from
  `src/live_ai.py`) raised by the streaming loops when the token is set, so the
  session can tell a cancel from a real error.
- **Early stream registration**: the completion request is issued from a helper
  thread while the main thread waits on a cancel-aware wait, so the in-flight
  HTTP response is reachable (and closable) from the very first moment —
  including prefill. Fallback path keeps the current behavior if the
  SDK/transport cannot be closed.
- **Process registry**: a small module-level registry of running subprocesses
  (`src/common.py`) with `register_process` / `terminate_all()`;
  `cancel_generation` calls `terminate_all()`.

## Tasks (one commit per task — one task at a time, confirmation before the next)

### Task 1 — Cancel primitives on the agent

- Add `GenerationCancelled` and a thread-safe `_cancel_event`
  (`threading.Event`) to `src/live_ai.py`; `cancel_requested` property reads the
  event; `_cancel_requested` stays as a compatibility alias.
- `cancel_generation()`: set the event, close the active stream, idempotent,
  never raises; `interrupt()` keeps its graceful semantics but also uses the
  event.
- Streaming loops (`_stream_round`, `model_fragments`, `_force_final_answer`)
  check the event and raise `GenerationCancelled` instead of `break`.

**Commit**: `feat(cancel): add GenerationCancelled and a thread-safe cancel event to Agent`

### Task 2 — Make the in-flight request cancellable from the first byte

- Register the stream/response *before* it can block (helper thread +
  cancel-aware wait, or a `with`-managed stream), so `cancel_generation()`
  aborts during prefill too.
- Add an httpx timeout to the client (`src/live_ai.py:712`) so a stalled stream
  cannot hang forever.

**Commit**: `fix(cancel): register the in-flight completion early and add a client timeout`

### Task 3 — Report a cancel as a cancel

- `src/session.py:_run_generation_step`: catch `GenerationCancelled` (and detect
  the event in the generic `except`) -> print `⏹ generation cancelled`, no `❌`.

**Commit**: `fix(session): report Ctrl+C cancellation as a cancel, not an error`

### Task 4 — Discard the partial answer on cancel

- On cancel, do not append the partial assistant text; drop the partial answer
  and any trailing incomplete tool-call group (`_repair_history` extension in
  `src/live_ai.py:1383`).
- Update the README paragraph (`README.md:558-564`) that documents "the partial
  answer stays in the chat history".

**Commit**: `feat(cancel): discard partial output and repair history after a cancelled turn`

### Task 5 — Chained speech guard (audio first, then generation)

- `src/tools/speech.py:_speech_sigint_guard`: on the 1st `Ctrl+C` stop the audio
  and re-install the previous (generation) handler; the 2nd `Ctrl+C` reaches
  `cancel_generation()`.
- Update README (`README.md:545-549`).

**Commit**: `feat(speech): chain Ctrl+C so the next press cancels the generation`

### Task 6 — Kill an already-running tool

- Add the process registry to `src/common.py`; register foreground subprocesses
  in `src/tools/system.py` (`run_command`, `_spawn_posix`/`_spawn_windows`
  foreground) and `src/tools/pyeval.py`; `cancel_generation()` calls
  `terminate_all()`.
- `_execute_tool_calls` stops waiting / reports cancelled results for the killed
  calls.

**Commit**: `feat(cancel): terminate running subprocesses when a generation is cancelled`

### Task 7 — Docs and end-to-end verification

- README "Interrupting" section rewritten to the new semantics; run the full
  suite.

**Commit**: `docs: document the reliable Ctrl+C cancellation semantics`

## Verification (unit tests, per task)

- **Task 1** (`tests/test_live_ai.py`): `cancel_generation()` sets the event and
  is idempotent/thread-safe; a stream that yields chunks forever raises
  `GenerationCancelled` promptly after the event is set from another thread;
  `interrupt()` does not close the stream.
- **Task 2**: a fake `_stream_completion` that blocks in `create()` (never
  returns) — `cancel_generation()` from another thread returns and the turn
  unwinds within a bounded time; a fake stream whose `close()` raises must not
  raise.
- **Task 3** (`tests/test_session.py`): a turn raising `GenerationCancelled`
  prints `⏹ generation cancelled`, **no** `❌`, and the session continues to
  `/quit`; the existing `test_ctrl_c_during_turn_cancels_but_keeps_session` and
  `test_sigint_during_generation_hits_cancel_handler` still pass.
- **Task 4**: after a cancelled turn the partial text is absent from
  `_messages`; `_repair_history` drops a trailing unanswered `user` message and
  a trailing incomplete `assistant(tool_calls)`/`tool` group; role alternation
  holds for the next request.
- **Task 5** (`tests/test_tools.py`, T16 speech): 1st simulated `SIGINT` during
  playback -> audio stopped, generation **not** cancelled, generation handler
  restored; 2nd `SIGINT` -> `cancel_generation()` called; existing
  `test_ctrl_c_during_playback_stops_audio_only` updated.
- **Task 6** (`tests/test_tools.py`): a fake `Popen` registered by `run_command`
  is killed when `terminate_all()` runs; `run_command` returns a
  cancelled/timeout-style result; `python_eval` subprocess terminated; no
  process left registered afterwards.
- **Task 7**: `pytest tests/ -q` fully green.

## Open points (assumed unless you object)

- `ask_user` (`src/tools/user_input.py`) and in-flight MCP calls are **not**
  force-killed in Task 6 (they block on I/O, not a child process); they are
  unblocked by the cancel check at the next boundary.
- Voice mode / barge-in is **out of scope** (chat-only).
