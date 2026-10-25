# PLAN_REMOVE_AGENT_POOL.md

Remove the multi-agent pool functionalities covered by `tests/test_agent_pool.py`
(the pool, its delegation tools and its async delegation mode) — while **keeping
the sub-agent activity monitor**, which remains in use by the Scheduler — and
amend the unit tests so the suite verifies the removal.

---

## 1. Goals

- Delete the multi-agent pool feature set end to end:
  - The `AgentPool` class (sync delegation, per-agent locks, duplicate-entry
    keys, receipt registry, executor).
  - The delegation tools `ask_agent`, `run_parallel_agents` and their seams
    (`_POOL`, `bind_agent_pool`, the nested-delegation refusal).
  - The async-delegation delivery path (`"delegation"` kind in the
    background-results queue and its session injection).
  - The `[agent]` pool config keys (`parallel_agents`, `agent_models`,
    `agent_system_prompts`, `async_delegation`) and the pool part of
    `agent_pool_settings`.
  - The pool startup wiring in `main.py` (`_setup_agent_pool`,
    `_stop_agent_pool`).
- **Keep the sub-agent activity monitor** (used by the Scheduler):
  - `AgentMonitor`, `bind_monitor`, `get_monitor`, `_monitor_flags`,
    `_monitor_buffer`, `MONITOR_BUFFER_DEFAULT`.
  - The `/monitor` session command.
  - The scheduler monitoring integration (`_monitor_or_none`,
    `_set_executor_name`, monitor records in `_agent_executor`).
  - The `agent_activity` tool (reports scheduled-run events).
  - The monitor reporting in `Agent._execute_tool` (via the `_TOOL_CONTEXT`
    thread-local, which the scheduler sets to `'scheduler'`).
  - The `[agent]` monitor config keys (`monitor`, `monitor_log_file`,
    `monitor_buffer`).
- Keep everything else the app depends on:
  - `_split_list` (also used by `[agent] tool_groups` selection).
  - The background-results queue for **scheduled tasks** (`"scheduled"` kind)
    and `drain_agent_results` (used by `session.py` and `tests/conftest.py`).
  - `Agent(isolated_tools=..., quiet=...)` as generic Agent capabilities — the
    scheduler's dedicated execution agent is built with them
    (`src/tools/scheduler.py:467`). Their tests move out of
    `test_agent_pool.py` into a dedicated file.
- Amend the unit tests: delete the tests of removed functionality, relocate the
  still-valid ones (monitor, isolation, quiet), and add negative tests
  asserting the pool features are gone.

## 2. Current behavior

- `main.py` builds an `AgentPool` from `[agent] parallel_agents` +
  `agent_models` (one quiet, tool-isolated `Agent` per models.json entry) and
  binds it via `bind_agent_pool`, enabling the tools `ask_agent` and
  `run_parallel_agents` for the main agent.
- `agent_pool_settings()` resolves the pool config, including monitor flags
  (`_monitor_flags`, `_monitor_buffer`) and the `async_delegation` flag.
- `AgentPool.ask` / `ask_parallel` delegate prompts to pooled agents (blocking,
  or returning `d-N` receipts when `async_delegation = true`; async answers are
  queued via `queue_agent_result("delegation", ...)` and injected into the main
  agent's history by `ConversationSession` at the next turn).
- The `AgentPool` constructor is what currently creates and binds the
  process-wide `AgentMonitor` when `[agent] monitor` includes `calls`/`log`
  (`bind_monitor`); the session exposes `/monitor`, the agent exposes the
  `agent_activity` tool, and the scheduler records its runs on the same monitor
  (via `_monitor_or_none` + `_set_executor_name('scheduler')`).
- `tests/test_agent_pool.py` (1032 lines) covers all of the above; related
  suites: `test_agent_monitor.py`, parts of `test_session.py`,
  `test_scheduler_agent.py`, `test_token_reduction.py`.

## 3. Expected behavior (after removal)

- The app is single-agent: `main.py` creates one `Agent` and starts the session;
  no pool is built, `bind_agent_pool` does not exist, `ask_agent` and
  `run_parallel_agents` are gone.
- **The monitor stays**: `main.py` (not the removed pool) reads the
  `[agent]` monitor keys and creates + binds the `AgentMonitor` when monitoring
  is enabled — a new small helper `monitor_settings(config)` replaces the
  monitor part of `agent_pool_settings`. With monitoring off nothing is
  recorded and no log file is written (unchanged).
- `/monitor` keeps working; `agent_activity` keeps working and reports
  scheduled-run events (the scheduler is now the only event source).
- Scheduled tasks keep running and keep delivering their results through the
  `"scheduled"` background-results path; `drain_agent_results()` still exists
  but never contains `"delegation"` entries.
- The scheduler execution agent keeps `Agent(config, isolated_tools=True,
  quiet=True)`; per-agent memory/vision isolation and quiet mode remain tested
  (relocated tests).
- `pytest tests/` passes with the amended suites; a new negative test asserts
  `from src.live_ai import AgentPool` raises `ImportError` and that
  `ask_agent`/`run_parallel_agents` are absent from `Agent._build_tools()`.

## 4. Out of scope

- The scheduler feature (`src/tools/scheduler.py`) itself — only its pool
  references (none) are touched; its monitor integration is kept.
- The background-results queue for scheduled tasks.
- `_split_list`, `_select_tools`, `_TOOL_TOPICS` (shared with `tool_groups`).

---

## 5. Implementation tasks (one commit per task)

Run the full suite after every task; a task is done only when the suite is
green. Commit message template: `Remove agent pool (N/M): <title>`.

### Task 1/6 — Remove async delegation
**Files:** `src/live_ai.py` (async branch of `AgentPool.ask`/`ask_parallel`,
`_run_delegation`, `_next_delegation_id`, `_DELEGATION_COUNTER`,
`_DELEGATION_LOCK`, `_delegations`, `_executor`, `stop()`, `async_delegation`
flag), `src/session.py` (the `kind == "delegation"` delivery branch),
`tests/test_agent_pool.py`.
**Amend tests:** delete `test_agent_pool_settings_async_delegation_*`,
`test_async_delegation_load_config_round_trip`, `test_pool_async_delegation_*`,
`test_pool_synchronous_by_default_and_stop_idempotent`, all `test_async_*`,
`test_synchronous_mode_unchanged`, `test_e2e_async_delegation_receipt_and_delivery`.
**Verify:** `pytest tests/test_agent_pool.py tests/test_session.py -x`
**Commit:** `Remove agent pool (1/6): drop async delegation mode`

### Task 2/6 — Re-wire the monitor creation from AgentPool to main.py
The monitor is **kept**; only its creation site moves (today it lives inside
`AgentPool.__init__`, which is about to be deleted).
**Files:** `src/live_ai.py` (new `monitor_settings(config)` helper returning
`{"monitor", "monitor_log_file", "monitor_buffer"}` using the kept
`_monitor_flags`/`_monitor_buffer`; remove monitor construction from
`AgentPool.__init__`), `main.py` (new `_setup_monitor(config)` called from
`main()`: build + `bind_monitor` an `AgentMonitor` when
`monitor_settings(config)["monitor"]` is non-empty, with `log_file` only when
`"log"` is among the flags; the monitor needs no shutdown hook),
`tests/test_agent_pool.py`, `tests/test_agent_monitor.py`.
**Amend tests:** relocate the monitor-settings tests
(`test_monitor_flags_parsing_variants`, `test_monitor_buffer_parsing`,
`test_agent_pool_settings_monitor_keys`, `test_monitor_settings_load_config_round_trip`)
to `tests/test_agent_monitor.py`, retargeted at `monitor_settings`;
`test_agent_monitor.py` pool-based tests (the `_make_pool`-driven ones) are
rewritten to bind a monitor directly + build a plain quiet/isolated agent —
no pool involved.
**Verify:** `pytest tests/test_agent_monitor.py tests/test_session.py tests/test_scheduler_agent.py -x`
**Commit:** `Remove agent pool (2/6): bind the activity monitor from main.py (kept for the scheduler)`

### Task 3/6 — Remove the delegation tools (keep agent_activity)
**Files:** `src/live_ai.py` (`ask_agent`, `run_parallel_agents`, `_POOL`,
`bind_agent_pool`, the nested-delegation refusal in `AgentPool.ask`,
`"delegation"` entry in `_TOOL_TOPICS`, the two tools from the `_default_tools`
list). **Kept:** `agent_activity` (scheduler events), `_TOOL_CONTEXT` +
`_current_agent_name` (the scheduler sets the thread-local name;
`Agent._execute_tool` monitor reporting depends on it).
**Amend tests:** in `tests/test_agent_pool.py` delete
`test_default_tools_include_delegation_tools`, `test_ask_agent_*`,
`test_run_parallel_agents_*`, `test_nested_delegation_is_refused`,
`test_pooled_agent_act_sets_thread_local_name`; in `tests/conftest.py` and
`tests/test_session.py` drop `_POOL`/`bind_agent_pool` usages.
**Add:** `test_delegation_tools_removed` asserting `ask_agent` and
`run_parallel_agents` are not importable from `src.live_ai` and not in
`Agent._build_tools()`, while `agent_activity` still is.
**Verify:** `pytest tests/ -x`
**Commit:** `Remove agent pool (3/6): drop ask_agent / run_parallel_agents tools`

### Task 4/6 — Remove the AgentPool class and main.py pool wiring
**Files:** `src/live_ai.py` (`class AgentPool`), `main.py` (`_AGENT_POOL`,
`_setup_agent_pool`, `_stop_agent_pool`, their call sites — the monitor setup
from Task 2 stays),
`tests/test_agent_pool.py` (delete all pool tests: `_make_pool`,
`_pool_config`, `test_agent_pool_builds_one_agent_per_model`,
`test_agent_pool_unknown_model_entry_fails_fast`,
`test_agent_pool_requires_models_file`, `test_agent_pool_empty_models_fails`,
`test_agent_pool_ask_*`, `test_agent_pool_per_agent_system_prompt`,
`test_duplicate_model_entries_get_unique_keys`,
`test_ask_parallel_same_agent_runs_sequentially`,
`test_ask_parallel_distinct_agents_still_fan_out`).
**Amend tests:** add `test_agent_pool_class_removed` asserting
`from src.live_ai import AgentPool` raises `ImportError`.
**Verify:** `pytest tests/ -x` and `python main.py --chat` smoke (single agent
boots, `/monitor` still answers).
**Commit:** `Remove agent pool (4/6): delete AgentPool and startup wiring`

### Task 5/6 — Remove agent_pool_settings and the [agent] pool config keys
**Files:** `src/live_ai.py` (`agent_pool_settings` — replaced by the Task-2
`monitor_settings`; pool keys `parallel_agents`, `agent_models`,
`agent_system_prompts`, `async_delegation` disappear; stale comments
referencing pooled workers, e.g. `_MODEL_META_CACHE`),
`config.toml` + examples (drop `parallel_agents`, `agent_models`,
`agent_system_prompts`, `async_delegation`; **keep** `monitor`,
`monitor_log_file`, `monitor_buffer`), `README.md`, `src/common.py` comment
mentioning `agent_models`.
**Amend tests:** delete the pool-settings tests in `tests/test_agent_pool.py`
(`test_agent_pool_settings_*`, `test_load_config_round_trips_agent_pool_keys`);
move the still-valid isolation/quiet tests (`test_pooled_agent_*`,
`test_main_agent_keeps_module_default_tools`, `test_tool_names_unchanged_for_pooled_agent`,
`test_quiet_agent_*`, `test_non_quiet_agent_still_speaks`, `test_max_act_rounds_*`)
to a new `tests/test_agent_isolation.py` (they test generic Agent capabilities
still used by the scheduler execution agent).
**Verify:** `pytest tests/ -x`; `grep -ri "agent_pool\|parallel_agents\|agent_models\|async_delegation" src/ main.py config.toml` returns nothing.
**Commit:** `Remove agent pool (5/6): drop agent_pool_settings and [agent] pool keys`

### Task 6/6 — Final cleanup and documentation
**Files:** delete `tests/test_agent_pool.py` (now empty), sweep remaining
pool references (`grep -ri "AgentPool\|agent pool\|bind_agent_pool\|ask_agent\|run_parallel_agents" src/ tests/ main.py README.md`),
update `README.md` (remove pool/async-delegation sections; document that the
activity monitor now tracks scheduled-task runs), remove the `"delegation"`
kind mention from the background-results comments (keep `"scheduled"`).
**Amend tests:** run the whole suite; confirm `tests/test_agent_isolation.py`
covers the retained Agent capabilities and `tests/test_agent_monitor.py`
covers the monitor end to end (parsing, ring buffer, JSONL sink, `/monitor`,
scheduler events via `agent_activity`).
**Verify:** `pytest tests/ -x` (full suite green), `python main.py --prompt "hi" --chat` smoke.
**Commit:** `Remove agent pool (6/6): cleanup, docs, final test sweep`

---

## 6. Risks and mitigations

- **Monitor regression:** the monitor is kept and re-wired in Task 2 *before*
  `AgentPool` is deleted (Task 4), so `/monitor`, `agent_activity` and the
  scheduler's event recording never lose their binding; covered by
  `tests/test_agent_monitor.py`, `tests/test_session.py` and
  `tests/test_scheduler_agent.py` after every task.
- **Scheduler regression:** the scheduler execution agent and the
  `"scheduled"` background-results path must keep working — covered by
  `tests/test_scheduler_agent.py` and `tests/conftest.py` after each task.
- **Shared helpers:** `_split_list`, `_TOOL_CONTEXT`/`_current_agent_name` and
  the results queue are shared; only pool-specific usage is removed (verified
  by the suite).
- **Config files in the wild:** leftover `[agent]` pool keys are inert after
  Task 5 (nothing reads them); document this in the README.
