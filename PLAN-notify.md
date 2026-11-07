# PLAN-notify.md — Immediate terminal notification for background task results

## Goals

1. When a **scheduled task** or a **delegated sub-agent** completes, the user
   sees an immediate one-line hint in the terminal — without typing anything.
2. The hint shows **only** the task kind and id (e.g.
   `🔔 Scheduled task t-1 completed — the full result will be delivered on
   your next message.`). **No result preview is printed.**
3. The hint is opt-in via a `[notify]` config section; setups without the
   section keep today's behavior exactly.
4. The hint must not corrupt the prompt line in chat mode
   (prompt_toolkit) nor the voice-mode loop.
5. The full result keeps being delivered into the agent history on the next
   turn (existing mechanism, unchanged).

## Current behavior

- Scheduled tasks (`src/tools/scheduler.py` → `_agent_executor`) and
  delegated sub-agents (`src/delegation.py` → `_run_child`) finish by calling
  `queue_agent_result(kind, result_id, text)` (`src/live_ai.py:567`).
- The result is appended to the shared `_BACKGROUND_RESULTS` queue and stays
  there until the **user types something**: `_run_turn` calls
  `_inject_scheduled_results` (`src/session.py:176`) which drains the queue
  and appends the full result to `agent.messages`.
- Consequence: while the user is idle at the prompt, a completed task gives
  **no feedback at all** — the terminal sits silent.
- In chat mode a raw `print()` from a background thread would garble the
  prompt_toolkit input line (text rendered mid-line, cursor jumps).

## Expected behavior

- Right after `queue_agent_result` stores a result, the background thread
  prints one hint line (ANSI color, spinner-aware via `cprint`):
  - kind `scheduled` → `🔔 Scheduled task t-1 completed — the full result
    will be delivered on your next message.`
  - kind `delegation` → `🔔 Sub-agent task researcher (agent-1) completed —
    the full result will be delivered on your next message.`
  - unknown kind → `🔔 Background task <id> completed — the full result will
    be delivered on your next message.`
- **No part of the result text is printed** (no preview).
- Empty result text (scheduler "produced no answer" case) → no hint.
- Config (all optional, missing section → disabled):

  ```toml
  [notify]
  terminal = true   # print a hint when a background task completes
  ```

- Chat mode: the session runs inside prompt_toolkit's `patch_stdout()`, so
  hints print safely **above** the prompt line and the prompt is redrawn
  below. Voice mode: `cprint` is already safe (no prompt_toolkit).
- The queue, its locking, `_inject_scheduled_results` and the agent state are
  untouched; the hint is purely cosmetic.

## Tasks (one commit per task)

### Task 1 — Notification config + hook in `src/live_ai.py`

- Add module state `_NOTIFY` (dict with `terminal: bool`) and
  `configure_notify(config)` reading the `[notify]` section
  (missing → `terminal = False`).
- In `queue_agent_result`, after appending to the queue, call the new
  `_notify_background_result(kind, result_id)` when `terminal` is on:
  builds the label from `kind` (`scheduled` / `delegation` / generic), prints
  via `cprint(..., "magenta")` — **no result text, no preview**.
- Export `configure_notify` from the module.
- Commit 1: `notify: add [notify] config and terminal hint hook`

### Task 2 — Wire config injection in `main.py`

- Call `configure_notify(config)` next to `configure_tools(config)` /
  `configure_ui(config)`.
- Chat branch: wrap the `ConversationSession(...).run()` call in
  prompt_toolkit's `patch_stdout()` context manager (lazy import with
  graceful fallback when prompt_toolkit is unavailable — degraded path
  already uses `prompt_input`, where raw printing is safe).
- Voice branch: unchanged.
- Commit 2: `notify: wire config injection and chat-mode patch_stdout`

### Task 3 — Unit tests

- `tests/test_background_results.py` (or a new
  `tests/test_notify.py`):
  - hint printed for `scheduled` and `delegation` kinds, exact wording,
    **asserts the result text does not appear** in the output;
  - no hint when `[notify]` is missing / `terminal = false`;
  - no hint for an empty result text;
  - generic label for an unknown kind;
  - hint still queues the result (drain returns it — delivery unchanged).
- `tests/test_main.py`: `configure_notify` is called at startup with the
  loaded config.
- Commit 3: `notify: unit tests for terminal hint`

### Task 4 — Documentation

- `README.md`: document the `[notify]` section next to `[scheduler]`
  (one short paragraph + example).
- Commit 4: `notify: document [notify] config`

## Verification of the implementation with unit tests

- Run `pytest tests/test_notify.py tests/test_background_results.py
  tests/test_main.py -q` — all green.
- Manual check (chat mode): run `--chat`, schedule a one-shot task
  ~15 s ahead via `schedule_task`, keep typing at the prompt; the hint
  renders above the prompt line without garbling it; the full result is
  injected into the next turn's history as before.
- Manual check (voice mode): same scenario, hint prints between prompts.
- Regression: with no `[notify]` section, behavior is byte-for-byte
  identical to today.
