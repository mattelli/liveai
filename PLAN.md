# PLAN.md — Task Scheduling Tool for the Live Assistant (Proposal A)

Implement an in-process task scheduler so the agent can run prompts at a
future time (one-shot or recurring), following the existing `src.tools`
architecture. Work through the tasks **in order**; make **one git commit per
task** (message given in each task). Do not start a task before the previous
one is committed and its tests pass.

---

## 1. Goals

- The agent gains three new tools:
  - `schedule_task(prompt, at, repeat="")` — schedule an agent prompt for a
    future time; `repeat` optionally makes it recurring.
  - `list_scheduled_tasks()` — list pending/recurring tasks with ids.
  - `cancel_scheduled_task(task_id)` — cancel a pending/recurring task.
- Scheduled prompts are executed by a **dedicated, isolated agent** (same
  pattern as pooled agents: `isolated_tools=True, quiet=True`), never by the
  main session agent — `Agent.act()` is not thread-safe (shared history,
  stream printer, spinner).
- Tasks **survive restarts**: persisted as JSON; on startup, missed one-shot
  jobs are handled per config (`run` or `skip`).
- Recurrence support: `"daily"`, `"hourly"`, `"every <N><m|h|s>"` (e.g.
  `"every 30m"`), and 5-field cron-like `"min hour dom month dow"`.
- No new third-party dependencies (stdlib `threading`, `heapq`, `json` only).
- Config-driven via a new `[scheduler]` section in `config.toml`; missing
  section keeps the scheduler disabled (backward compatible).

## 2. Current behavior

- `src/live_ai.py` defines the tool-bearing `Agent` (round-based `act()` loop)
  and `AgentPool` (named quiet, isolated sub-agents for delegation).
- Tools live in `src/tools/<topic>.py`, are re-exported by
  `src/tools/__init__.py`, receive app config via `configure(config)` (called
  once by `main.py` after `load_config`), and session-bound state via
  `bind_*` functions (see `memory.py`, `vision.py` as reference).
- `main.py` builds the main `Agent`, optionally the `AgentPool`
  (`_setup_agent_pool`), then runs `ConversationSession`.
- There is **no** way to run anything in the future: all tool execution is
  synchronous inside the user's turn. No scheduler module, no `[scheduler]`
  config, no persistence of deferred work.

## 3. Expected behavior

- User (or the model) calls `schedule_task(prompt="...", at="2026-01-01 09:00")`;
  the tool returns a confirmation with a task id, e.g.
  `Scheduled task t-1 at 2026-01-01 09:00 (one-shot): <prompt>`.
- At (or shortly after) the scheduled time, a background thread wakes up and
  runs the prompt on the dedicated scheduler agent (`act(prompt)`); the
  answer is logged (and printed via the agent's logger; no console spinner).
- `repeat` values reschedule the task after each run (next occurrence
  computed from the previous scheduled time); one-shot tasks are removed
  after running.
- `list_scheduled_tasks()` returns one line per task:
  `<id> | <next_run ISO> | <repeat or "one-shot"> | <prompt truncated to 80 chars>`,
  or `No scheduled tasks.` when empty.
- `cancel_scheduled_task("t-1")` removes the task and returns
  `Cancelled task t-1`; unknown ids return an `Error: ...` string (tools
  never raise; they return `Error: ...` strings, matching existing tools).
- After a restart, persisted tasks are reloaded; a one-shot whose time already
  passed is either run immediately or dropped, per `[scheduler] on_missed`.
- When the scheduler is disabled (no `[scheduler]` section or
  `enabled = false`), the three tools return
  `Error: scheduler is disabled (set [scheduler] enabled = true)`.

## 4. Design notes (read before implementing)

- **Scheduler core** (`Scheduler` class): daemon thread + `heapq` min-heap of
  jobs ordered by `next_run`; a `threading.Event` for wake-up on
  add/cancel/quit (no busy polling; wait with timeout = time of next job).
- **Execution agent**: lazily built on first run via the same recipe as
  `AgentPool` (`Agent(config, isolated_tools=True, quiet=True)`); the
  scheduler owns it exclusively — only one `act()` at a time (serialize with
  a lock, like `AgentPool.ask` does per agent).
- **Persistence**: JSON file (default `data/schedule.json`, resolved relative
  to the config file) holding the task list; rewritten atomically
  (write temp file + `os.replace`) on every mutation.
- **Time parsing**: accept ISO 8601 (`datetime.fromisoformat`) and
  `YYYY-MM-DD HH:MM`; naive datetimes are local time. Reject past times for
  one-shot tasks with an `Error: ...` string.
- **Module-default instance pattern**: copy `memory.py` — logic in a
  `SchedulerTools` class with `for_owner(agent)`, module-level functions
  delegating to a module-default instance. This keeps the historical
  monkeypatch-able seams for tests and per-agent isolation for pooled agents.
- **Test seams**: expose a module-level `_SCHEDULER_FACTORY` (like
  `_VIDEO_CAPTURE_FACTORY` in `camera.py`) so tests can inject a fake
  scheduler; tests must never need a real LLM.

## 5. Tasks

### Task 1 — Scheduler core engine (`src/tools/scheduler.py`)

- Create `src/tools/scheduler.py` with:
  - `parse_when(at: str) -> datetime` (ISO/`YYYY-MM-DD HH:MM`; raises
    `ValueError` on bad input).
  - `next_occurrence(repeat: str, after: datetime) -> datetime` (supports
    `daily`, `hourly`, `every <N><s|m|h|d>`, cron-like 5-field
    `min hour dom month dow`; raise `ValueError` on unknown formats).
  - `ScheduledTask` dataclass: `id`, `prompt`, `next_run` (ISO string),
    `repeat` (`""` = one-shot).
  - `Scheduler` class: `add(prompt, at, repeat) -> ScheduledTask`,
    `cancel(task_id) -> bool`, `list() -> list[ScheduledTask]`,
    `start()`, `stop()`, and a `_run_loop()` daemon thread that pops due
    jobs, calls a `_execute(task)` callback (injected; default logs), and
    reschedules recurring tasks.
  - Persistence: `_load()` on init, `_save()` (atomic) on every mutation;
    `on_missed` handling for one-shot tasks in the past at load time
    (`"run"` → execute immediately, `"skip"` → drop).
- Config: read `[scheduler]` (`enabled`, `persist_file`, `on_missed`) via a
  module `configure(config)`; missing section → `enabled=False`.
- **Unit tests** (`tests/test_scheduler.py`, no LLM, no real threads where
  avoidable — drive `_run_loop` logic via the injected `_execute` callback or
  very short waits):
  - `parse_when`: valid ISO, `YYYY-MM-DD HH:MM`, invalid strings raise.
  - `next_occurrence`: daily/hourly/every 30m/cron cases; unknown format raises.
  - `Scheduler.add/cancel/list` round-trip; cancel unknown id returns False.
  - Persistence: add → new `Scheduler` on same file sees the task.
  - `on_missed`: `run` executes a past one-shot once; `skip` drops it.
  - Disabled scheduler (no config): tools return the disabled-error string.
- **Commit**: `feat(scheduler): core engine with parsing, recurrence, persistence`

### Task 2 — Agent tools on top of the engine

- In `src/tools/scheduler.py` add the `SchedulerTools` class +
  module-default instance (copy the `memory.py` pattern) and the three
  module-level tools:
  - `schedule_task(prompt: str, at: str, repeat: str = "") -> str`
  - `list_scheduled_tasks() -> str`
  - `cancel_scheduled_task(task_id: str) -> str`
- All tools return strings, never raise; errors as `Error: ...`.
- Add a module-level `_SCHEDULER_FACTORY` seam (default builds the real
  `Scheduler` from config) and `bind_scheduler(...)` if needed for the
  execution-agent injection (Task 3).
- Re-export the three tools from `src/tools/__init__.py` (imports + `__all__`)
  and add `scheduler` to the `configure()` module list.
- **Unit tests** (extend `tests/test_scheduler.py` or `tests/test_tools.py`):
  - `schedule_task` returns a confirmation containing the task id; bad `at`
    returns `Error: ...`; past one-shot returns `Error: ...`.
  - `list_scheduled_tasks` shows the added task; empty state message.
  - `cancel_scheduled_task` cancels an existing id; unknown id → `Error: ...`.
  - Tools honor the `_SCHEDULER_FACTORY` seam (inject a fake scheduler).
- **Commit**: `feat(scheduler): schedule_task/list/cancel agent tools`

### Task 3 — Execution agent wiring (scheduler runs prompts)

- In `src/tools/scheduler.py`: the scheduler's `_execute(task)` builds (once,
  lazily) a dedicated isolated agent and runs `act(task.prompt)` under a
  lock. Agent construction needs the app `config`: extend `configure(config)`
  to keep the full config, and build the agent the same way `AgentPool` does
  (`Agent(config, isolated_tools=True, quiet=True)`); if the models file is
  missing/unavailable, fall back to logging the prompt only (never crash).
- Start the scheduler thread from `main.py` after `configure_tools(config)`
  (add a small `_setup_scheduler(config)` helper mirroring
  `_setup_agent_pool`), and `stop()` it in the `finally` block.
- **Unit tests**:
  - Inject a fake executor via the seam; assert a due one-shot task triggers
    exactly one `act(prompt)` call with the right prompt.
  - Assert a recurring task triggers repeated calls with advancing
    `next_run` values.
  - Assert execution errors (fake agent raising) are caught and logged, and
    the scheduler thread stays alive.
  - `main.py` startup test: with a fake factory, `_setup_scheduler` starts
    the scheduler; missing `[scheduler]` section keeps it off.
- **Commit**: `feat(scheduler): run scheduled prompts on a dedicated agent`

### Task 4 — Register tools with the agent + config documentation

- Add `schedule_task, list_scheduled_tasks, cancel_scheduled_task` to
  `_default_tools()` in `src/live_ai.py` (import from `src.tools`).
- Add the `[scheduler]` section to `config.toml` (commented example, like
  `[mcp]`): `enabled`, `persist_file`, `on_missed`.
- Mention the new tools in `README.md` (short section: what they do, config).
- **Unit tests**:
  - `tests/test_live_ai.py` (or a new test): a `Agent`-like tool list includes
    the three new tools and their OpenAI tool definitions are well-formed
    (name + description + parameters present).
  - Config load test: `config.toml` with `[scheduler]` parses into the
    expected attributes (enabled/persist_file/on_missed).
- **Commit**: `feat(scheduler): register tools with the agent + config/README`

### Task 5 — End-to-end smoke + cleanup

- Add an integration-style test (mark it like the existing
  `test_integration.py`/`test_smoke.py` markers): with a fake executor agent
  (no LLM), schedule a task ~1 second in the future, start the scheduler,
  wait for execution, assert the prompt was run and the task list is empty
  afterwards; then cancel-path: schedule 1 hour ahead, cancel, assert never
  executed.
- Run the **full** test suite (`python -m pytest tests/`) and fix any
  regressions.
- Verify manually once (optional, needs a working model): run the app, ask
  the agent to schedule a reminder 2 minutes ahead, confirm it fires.
- **Commit**: `test(scheduler): end-to-end smoke test + suite green`

## 6. Verification checklist (definition of done)

- [x] `python -m pytest tests/` passes with the new tests from all tasks.
- [x] `schedule_task` / `list_scheduled_tasks` / `cancel_scheduled_task`
      appear in the agent's tool list and are callable by the model.
- [x] A scheduled one-shot prompt runs at the requested time on the isolated
      agent; recurring tasks reschedule themselves.
- [x] Tasks persist across a restart; missed one-shots follow `on_missed`.
- [x] With no `[scheduler]` section, behavior is unchanged (tools return the
      disabled error; existing tests untouched and green).
- [x] One commit per task, in order, with the messages above.
