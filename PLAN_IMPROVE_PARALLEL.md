# PLAN_IMPROVE_PARALLEL.md — Non-blocking sub-agent delegation (Option 3)

Make `ask_agent` / `run_parallel_agents` **non-blocking** for the main agent:
the tool call returns immediately with delegation receipts, workers keep
running in the background, and their answers are delivered to the main agent
as injected history messages on later turns — the same pattern the task
scheduler already uses (`_PENDING_RESULTS` / `drain_scheduled_results` /
`_inject_scheduled_results`).

Work through the tasks **in order**; make **one git commit per task**
(message given in each task). Do not start a task before the previous one is
committed and its tests pass.

---

## 1. Goals

- The main agent never blocks on worker completion: `ask_agent` /
  `run_parallel_agents` return **immediately** with a receipt the model can
  read and relay to the user ("dispatched 2 tasks, results will arrive").
- Worker answers are delivered asynchronously as **assistant-role history
  messages** ("Worker `<name>` finished task `<id>`. Result: ..."), injected
  by the session at the start of each turn — reusing the scheduler's
  pending-results machinery (generalized to any background result).
- The user can keep chatting while workers run; the main agent can mention
  pending work ("2 delegations still running") because the receipts are in
  its history and a `agent_activity()` call shows live status.
- **Backward compatible config flag**: `[agent] async_delegation = true`
  enables the new behavior; unset/false keeps today's synchronous semantics
  byte-for-byte (existing tests stay green).
- Cancellation and shutdown stay safe: Ctrl+C cancels the main agent's turn
  only; running workers finish their current round (unchanged rule), and
  undelivered results are drained at shutdown without crashing.
- The monitor keeps recording worker events exactly as today (start /
  tool_call / tool_result / answer) — async delegation changes *when* the
  answer is consumed, not *whether* it is monitored.

## 2. Current behavior

- `AgentPool.ask(agent, prompt)` runs `agent.act(prompt)` **synchronously**
  and returns the worker's final answer; `AgentPool.ask_parallel(tasks)`
  fans tasks out on a `ThreadPoolExecutor` (concurrency between workers
  only) and blocks on `future.result()` until every task finishes.
- The delegation tools (`ask_agent`, `run_parallel_agents` in
  `src/live_ai.py`) call the pool synchronously from `Agent._execute_tool`,
  inside the main agent's `act()` round loop: the main agent's spinner spins
  ("thinking") while workers run; the user waits with no visible progress
  and cannot interact until the tool result returns.
- The scheduler already solves the same problem for scheduled prompts:
  `_agent_executor` (scheduler thread) queues `(task_id, answer)` into
  `_PENDING_RESULTS` (module-level list + lock in `src/tools/scheduler.py`),
  and `ConversationSession._inject_scheduled_results` drains it at the start
  of every turn, appending assistant messages to the main agent's history.
- `ConversationSession.run()` loops: read prompt → `handle_command` →
  `_run_generation_step(lambda: self._run_turn(text))`; `_run_turn` first
  injects scheduled results, then calls `agent.act(text)`.
- There is no notion of a pending *delegation*: no receipt format, no
  delegation registry, no delivery of worker answers outside the synchronous
  return path.

## 3. Expected behavior

- With `[agent] async_delegation = true`:
  - `ask_agent(agent="glm", prompt="research X")` returns immediately:
    `Delegated task d-1 to glm (async): results will be delivered when the
    worker finishes.` The worker starts running in a background thread at
    delegation time.
  - `run_parallel_agents(tasks=[{...}, {...}])` returns immediately:
    `Delegated 2 tasks (d-1, d-2) to glm, qwen (async): results will be
    delivered as the workers finish.` Distinct agents still fan out
    concurrently; same-agent tasks still serialize on that agent.
  - When a worker finishes, its answer is queued for delivery; at the start
    of the main agent's **next turn** the session injects:
    `Worker glm finished delegated task d-1. Result:\n<answer>` as an
    assistant-role message (same injection point as scheduled results).
  - The main agent can answer questions about pending work: receipts in its
    history + `agent_activity()` events (`start` recorded, no `answer` yet)
    let it say "task d-2 is still running".
  - A worker failure (exception in `act`) is delivered as a result too:
    `Worker glm failed delegated task d-1: <error>` — the model learns the
    task did not complete instead of waiting forever.
- With `async_delegation` unset/false (default): **today's behavior
  unchanged** — synchronous blocking delegation, immediate answers in the
  tool result.
- Nested delegation is still refused in both modes (pooled agents cannot
  delegate); the refusal happens at submission time in async mode.
- `/monitor` and the JSONL log keep working identically; async worker runs
  produce the same event sequence as synchronous ones.
- Session shutdown: pending (undelivered) delegation results are drained and
  logged, never lost silently nor crashing the shutdown path.

## 4. Design notes (read before implementing)

- **Reuse the scheduler pattern, don't fork it**: the cleanest seam is to
  generalize "background results waiting for the main agent" into one
  mechanism. Two options, pick A:
  - **A (chosen)**: move the pending-results queue into `src/live_ai.py`
    (next to the monitor): `queue_agent_result(kind, id, text)` /
    `drain_agent_results() -> list[(kind, id, text)]` with `kind` in
    `{"scheduled", "delegation"}`. `src/tools/scheduler.py` keeps thin
    wrappers (`drain_scheduled_results` delegates to the shared queue with
    `kind="scheduled"`) so its public API and tests stay untouched. The
    session's `_inject_scheduled_results` becomes `_inject_background_results`,
    formatting each kind appropriately.
  - B (rejected): a second queue in live_ai — two injection points, two
    drains, ordering bugs.
- **Delegation registry**: `AgentPool` tracks in-flight delegations:
  `self._delegations: dict[str, dict]` mapping receipt id →
  `{agent, prompt, thread}`. Ids are `d-1`, `d-2`, ... (monotonic counter,
  process-wide like the monitor to avoid collisions across pool rebuilds).
- **Async execution**: in async mode `ask()` does NOT run `act()` inline;
  it submits `self._run_delegation(id, name, prompt)` to a
  `ThreadPoolExecutor` owned by the pool (`max_workers = len(self.agents)`,
  created once in `__init__`, daemon threads, `shutdown(wait=False)` on
  stop). `_run_delegation` wraps the existing synchronous logic (thread-local
  name, `set_subagent_dump`, monitor events, answer extraction) and on
  completion/failure queues the result via the shared queue. The existing
  synchronous path stays for `async_delegation = false`.
- **Same-agent serialization**: the per-agent lock discipline must survive:
  async mode keeps grouping by agent — `ask_parallel` submits at most one
  task per agent immediately; additional tasks for the same agent are
  chained (submitted as a follow-up that runs after the previous completes)
  or, simpler: keep a per-agent `threading.Lock` acquired inside
  `_run_delegation` around `act()` (mirrors today's "one act() per agent"
  rule without executor juggling).
- **Receipt format** (stable, testable):
  - single: `Delegated task d-1 to glm (async): results will be delivered
    when the worker finishes.`
  - parallel: `Delegated 2 tasks (d-1, d-2) to glm, qwen (async): results
    will be delivered as the workers finish.`
  - unknown agent / nested refusal: unchanged `Error: ...` strings,
    returned immediately.
- **Config seam**: `agent_pool_settings(config)` gains
  `"async_delegation": bool` (default False, defensive parsing like the
  monitor keys). `config.toml` gets a commented example key.
- **Test seams**: the pool's executor must be injectable or the tests drive
  `_run_delegation` directly (deterministic, no sleeps); the shared queue is
  module-level and drained by an autouse conftest fixture (extend the
  existing `_drain_scheduler_pending_results` fixture to the generalized
  queue). No test sleeps waiting for threads except one explicit e2e test.
- **Ordering guarantee**: results are delivered in *completion* order, not
  delegation order; each injected message carries the receipt id so the
  model can correlate. `run_parallel_agents`' synchronous return already
  gave the model the id→task mapping via the receipt text.

## 5. Tasks

### Task 1 — Shared background-results queue in `src/live_ai.py`

- Add next to the monitor:
  - `_BACKGROUND_RESULTS: list[tuple[str, str, str]]` +
    `_BACKGROUND_RESULTS_LOCK` (kind, id, text).
  - `queue_agent_result(kind: str, id: str, text: str) -> None`
  - `drain_agent_results() -> list[tuple[str, str, str]]` (atomic swap).
- Refactor `src/tools/scheduler.py`: `_PENDING_RESULTS` appends become
  `queue_agent_result("scheduled", task.id, answer)`; `drain_scheduled_results()`
  drains and filters `kind == "scheduled"` (keeps its public signature and
  tests green). Remove the module-local list/lock.
- **Unit tests** (`tests/test_agent_monitor.py` or a new
  `tests/test_background_results.py`):
  - queue then drain returns the entries and empties the queue; drain on
    empty returns `[]`; thread-safety (8 threads × 25 queued → 200 drained).
  - `drain_scheduled_results` still works and only returns scheduled kind.
- **Commit**: `feat(async): shared background-results queue`

### Task 2 — Async delegation setting + pool plumbing

- `agent_pool_settings(config)`: add `"async_delegation": bool`
  (`[agent] async_delegation`, default False; invalid values → False).
- `config.toml`: commented `# async_delegation = true` example in `[agent]`.
- `AgentPool.__init__`: store the flag; create the long-lived executor
  (`ThreadPoolExecutor(max_workers=max(1, len(agents)),
  thread_name_prefix="delegate")`) and a per-agent `threading.Lock` dict;
  add `stop()` that shuts the executor down without waiting (called from
  `main.py`'s `finally` next to `_stop_scheduler`).
- **Unit tests** (`tests/test_agent_pool.py`):
  - settings default False; `async_delegation = true` parses True; bad
    values → False; `load_config` round-trip.
  - pool exposes `async_delegation` flag, executor and `stop()` (stop is
    idempotent).
- **Commit**: `feat(async): [agent] async_delegation setting + pool executor`

### Task 3 — Non-blocking `ask`/`ask_parallel` + delivery

- `AgentPool.ask(...)` in async mode:
  - validate name/refusals as today; allocate receipt id
    (`d-<n>` from a module-level counter), register in `_delegations`;
  - submit `_run_delegation(id, name, prompt)` to the executor; return the
    receipt string immediately (no `act()` on this thread);
  - synchronous mode: exactly today's code path.
- `_run_delegation(id, name, prompt)`: acquires the per-agent lock, runs the
  existing synchronous body (thread-local name, dump suppression, monitor
  `start`/`answer`, answer extraction), removes the delegation from the
  registry, and on success queues
  `queue_agent_result("delegation", id, f"Worker {name} finished delegated
  task {id}. Result:\n{answer}")`; on `act()` failure queues
  `... failed delegated task {id}: <error>` (and records a monitor
  `answer` event with the error so `/monitor` shows the outcome).
- `ask_parallel` in async mode: validate tasks, allocate one id per task,
  submit all (per-agent locks serialize same-agent tasks), return the
  parallel receipt; synchronous mode unchanged.
- Session: rename/extend `_inject_scheduled_results` →
  `_inject_background_results` formatting `scheduled` (existing text) and
  `delegation` (new text) kinds; called from the same place in `_run_turn`.
- `main.py`: call `pool.stop()` in the `finally` block (only when the pool
  exists — keep a module-level reference like the scheduler's).
- **Unit tests**:
  - async `ask` returns the receipt immediately and does NOT run `act()` on
    the calling thread (fake agent records the calling thread id);
    `_run_delegation` runs `act` once with the right prompt and queues the
    delivery text containing id + agent + answer.
  - failure path queues the failure text; monitor got `start` + `answer`.
  - `ask_parallel` async: N receipts, all tasks submitted, per-agent lock
    prevents concurrent `act()` on the same agent (two tasks, lock-held
    counter asserts max concurrency 1).
  - synchronous mode: behavior identical to today (existing tests already
    cover it — they must stay green untouched).
  - session `_inject_background_results` appends the right message text for
    both kinds; unknown kinds are skipped, not crashing.
- **Commit**: `feat(async): non-blocking delegation with queued delivery`

### Task 4 — Registration, config docs, README

- `README.md`: extend the multi-agent pool section (and the monitoring
  section cross-reference) with the async mode: receipt format, delivery on
  next turn, `agent_activity()` for pending status, the config flag, and the
  note that Ctrl+C still only cancels the main turn.
- `config.toml` example already added in Task 2 — verify ordering/comments.
- **Unit tests**:
  - e2e-style (offline pool, `async_delegation = true`): `ask_agent` tool
    returns a receipt; run the submitted delegation to completion; session
    turn injects the worker answer into the history; `agent_activity()`
    reports the run.
  - shutdown: pending results are drained by the conftest fixture (no leak
    into other tests) and `pool.stop()` is idempotent.
- **Commit**: `docs(async): async delegation usage + e2e test`

## 6. Verification checklist (definition of done)

- [ ] `python -m pytest tests/` passes with the new tests from all tasks.
- [ ] With `async_delegation` unset, all existing tests pass untouched and
      runtime behavior is byte-identical (synchronous delegation).
- [ ] With `async_delegation = true`: `ask_agent`/`run_parallel_agents`
      return receipts immediately; the main agent's turn completes without
      waiting; worker answers arrive as injected assistant messages on a
      later turn, carrying the receipt id.
- [ ] Worker failures are delivered (no silent hangs); same-agent tasks
      never run `act()` concurrently; nested delegation still refused.
- [ ] Monitor events (`start`/`tool_call`/`tool_result`/`answer`) are
      recorded for async runs exactly as for synchronous ones; `/monitor`
      and `agent_activity()` show pending vs finished delegations.
- [ ] Shutdown drains pending results; `pool.stop()` is idempotent; no
      thread leaks between tests (autouse fixture).
- [ ] One commit per task, in order, with the messages above.
