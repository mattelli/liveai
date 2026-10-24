# PLAN_MONITOR.md — Sub-agent activity monitoring (Option 1 + 3)

Implement user-visible monitoring of the pooled sub-agents' activity, via a
config-driven verbosity setting (Option 1: re-enable the `[CALL]` tool-call
dumps) and a JSONL activity log (Option 3: an event bus writing every
sub-agent event to a file, plus an in-memory ring buffer exposed to the main
agent through a new `agent_activity` tool).

Work through the tasks **in order**; make **one git commit per task**
(message given in each task). Do not start a task before the previous one is
committed and its tests pass.

---

## 1. Goals

- The user can see **what sub-agents are doing** while they run, instead of
  only receiving their final answers.
- Monitoring is **config-driven** via a new `[agent] monitor` setting, so the
  default behavior stays exactly as today (`monitor` unset/empty = fully
  quiet sub-agents, all existing tests green).
- Two independent, combinable monitoring channels (comma-separated flags):
  - `calls` — re-enable the `[CALL]` tool-call blocks for sub-agent tool
    executions (labeled with the agent name), printed live on the console.
  - `log` — append a JSON event per sub-agent lifecycle step to a JSONL file
    (`[agent] monitor_log_file`, default `data/agents.jsonl`): delegation
    start/end, per-tool-call start/end with duration and success/failure.
- A **ring buffer** of recent events is kept in memory and exposed to the
  main agent as a new tool `agent_activity()` (so the user can ask
  "what did the sub-agents do?" and the model can answer from the log).
- No new third-party dependencies (stdlib `json`, `threading`,
  `collections.deque`, `time` only).
- Thread-safe: `run_parallel_agents` fans out on worker threads; events from
  concurrent agents must never interleave/corrupt lines in the JSONL file.

## 2. Current behavior

- `src/live_ai.py` `AgentPool` builds one quiet, tool-isolated `Agent` per
  `[agent] agent_models` entry (`Agent(config, isolated_tools=True,
  quiet=True)`); its stream printer is a `NullPrinter` (buffers only, nothing
  on console) and `AgentPool.ask()` calls `set_subagent_dump(False)` before
  `agent.act(prompt)`, which suppresses the `[CALL]` blocks printed by
  `dump_func_call`/`_print_call_block` in `src/common.py`.
- The user sees **nothing** of a sub-agent's work: no tool calls, no
  intermediate text, no timing. Only the final answer string returned by
  `ask_agent` / `run_parallel_agents` reaches the main agent.
- `agent_pool_settings(config)` (in `src/live_ai.py`) resolves the `[agent]`
  pool settings (`count`, `models`, `system_prompts`, `quiet`); it is the
  single seam between config and pool construction.
- `Agent._execute_tool(call)` (in `src/live_ai.py`) runs one tool call for an
  agent; it is the natural interception point for per-tool events.
- There is no event bus, no log file, no ring buffer, no `agent_activity`
  tool; `src/common.py` has no notion of sub-agent events beyond the
  thread-local `show_dump` flag.

## 3. Expected behavior

- `[agent] monitor` in `config.toml` accepts a comma-separated list of flags:
  - unset/empty/`"none"` → current behavior (fully quiet sub-agents).
  - `"calls"` → while a sub-agent runs, its tool executions print the usual
    `[CALL]` blocks, each header labeled with the agent name, e.g.
    `[CALL] web_search (ternary-bonsai)(`. The main agent's own `[CALL]`
    blocks are unchanged.
  - `"log"` → every sub-agent event is appended as one JSON object per line
    to `[agent] monitor_log_file` (default `data/agents.jsonl`), with events:
    - `{"ts": <ISO-8601 UTC>, "agent": "ternary-bonsai", "event": "start",
      "prompt": "<prompt truncated to 200 chars>"}`
    - `{"ts": ..., "agent": ..., "event": "tool_start", "tool": "web_search"}`
    - `{"ts": ..., "agent": ..., "event": "tool_end", "tool": "web_search",
      "ok": true, "ms": 1234}` (on failure: `"ok": false, "error": "<msg
      truncated to 200 chars>"`)
    - `{"ts": ..., "agent": ..., "event": "done", "answer": "<final answer
      truncated to 200 chars>", "ms": <total act() duration>}`
  - `"calls,log"` (or `"log,calls"`) → both channels active.
- The JSONL file is written **atomically per line** (single `write` of one
  complete line under a lock) so parallel agents never interleave half-lines;
  the file is created lazily on the first event and its directory is created
  if missing.
- A bounded in-memory ring buffer (default 100 events, `[agent]
  monitor_buffer` to override) mirrors the logged events (regardless of the
  `log` flag — the buffer is always populated when monitoring is enabled).
- New tool `agent_activity(last: int = 20) -> str`: returns the last `last`
  events as human-readable one-line strings
  (`<hh:mm:ss> <agent> <event> <details>`), or
  `No sub-agent activity recorded.` when the buffer is empty. It is
  registered in `_default_tools()` so the main agent (and only the main
  agent — pooled agents are isolated) can consult it.
- With `monitor` unset, `agent_activity` still exists but always reports the
  empty message (no events are recorded).
- Errors never crash the agent loop: any failure inside the monitoring
  machinery (log write, buffer append) is swallowed and logged via the
  module logger — monitoring must never break a delegation.

## 4. Design notes (read before implementing)

- **Settings seam**: extend `agent_pool_settings(config)` with a `"monitor"`
  key (normalized set of flags, e.g. `{"calls", "log"}`) and
  `"monitor_log_file"` / `"monitor_buffer"`. Follow the existing defensive
  style: missing `[agent]` section or invalid values fall back to defaults
  (`monitor = set()`, buffer = 100). Update the existing
  `tests/test_agent_pool.py` settings tests to include the new keys.
- **Event bus**: a small module-level class `AgentMonitor` in
  `src/live_ai.py` (same file as the pool — it is pool behavior, not a tool
  topic) owning:
  - a `threading.Lock` guarding both the file append and the deque append;
  - `log_event(agent, event, **fields)` — builds the record (UTC ISO `ts`),
    appends to the deque and (if the `log` flag is on) writes one JSON line;
  - `snapshot(last)` — returns the last N events from the deque;
  - an injectable `_writer` seam (callable receiving the JSON line) so tests
    never touch the real filesystem unless they want to.
  The pool holds one `AgentMonitor` built from the settings; `ask()` uses it.
- **Interception points** (all in `AgentPool.ask()`, no changes to `Agent`
  itself):
  - wrap the `agent.act(prompt)` call with `start`/`done` events (measure
    with `time.monotonic()`);
  - for per-tool events, temporarily wrap `agent._execute_tool` with a
    closure that emits `tool_start`/`tool_end` around the original method
    (restore in the `finally` block). This keeps `Agent` untouched and works
    for both `ask` and `ask_parallel` (which delegates to `ask`).
- **`calls` flag**: in `ask()`, call `set_subagent_dump(False)` only when the
  `calls` flag is **off** (current behavior); when on, skip the suppression
  so `_print_call_block` prints normally. To label the blocks with the agent
  name, extend `_print_call_block` in `src/common.py` to read an optional
  thread-local agent name (`_call_context.agent_label`, set by `ask()` —
  reuse the existing `_call_context` thread-local, adding a new attribute;
  keep `show_dump` semantics unchanged).
- **`agent_activity` tool**: implement as a module-level function in
  `src/live_ai.py` reading from the module-level monitor (the pool binds it
  via a small `bind_agent_monitor(monitor)` setter next to the existing
  `bind_agent_pool`). Add it to `_default_tools()`. It must be safe to call
  when no pool/monitor exists (return the empty message).
- **Truncation**: prompts/answers/errors in events are truncated to 200
  chars (single-line: replace newlines with spaces) to keep JSONL lines
  bounded.
- **Test seams**: tests inject a fake writer into `AgentMonitor` and build a
  pool-less monitor directly; no LLM, no real model server. For pool-level
  tests, reuse the `_offline_agent` pattern from `tests/test_agent_pool.py`
  (stubbed OpenAI client) if an `Agent` instance is needed.

## 5. Tasks

### Task 1 — Monitor settings resolution + config plumbing

- Extend `agent_pool_settings(config)` in `src/live_ai.py`:
  - `"monitor"`: parse `[agent] monitor` (string or list) into a `set` of
    known flags (`"calls"`, `"log"`); unknown tokens are ignored; empty/`none`
    → empty set.
  - `"monitor_log_file"`: path string (default `"data/agents.jsonl"`; not
    path-resolved by `load_config` since it arrives inside the `[agent]`
    namespace handling — resolve relative to the config dir if it ends with
    `_file`, or keep the plain default).
  - `"monitor_buffer"`: int (default 100; invalid/<=0 → default).
- Add a commented `[agent]` example block for the three new keys in
  `config.toml` (`monitor = ""`, `monitor_log_file = "data/agents.jsonl"`,
  `monitor_buffer = 100`).
- **Unit tests** (extend `tests/test_agent_pool.py` settings section):
  - default config (no `[agent]` section) → `monitor == set()`,
    `monitor_log_file == "data/agents.jsonl"`, `monitor_buffer == 100`.
  - `monitor = "calls"` / `"log"` / `"calls,log"` / `"log, calls"` → correct
    sets; unknown token `"bogus"` ignored; `"none"` → empty set.
  - `monitor_buffer = "50"` (string) → 50; `monitor_buffer = "bad"` → 100.
  - `load_config` round-trip: a TOML file with the three keys parses into the
    settings dict as expected.
- **Commit**: `feat(monitor): [agent] monitor settings resolution + config`

### Task 2 — `AgentMonitor` event bus (JSONL + ring buffer)

- Add the `AgentMonitor` class to `src/live_ai.py` (see Design notes):
  constructor takes `(log_file: str | None, buffer_size: int)`; a `None`
  log file disables file writing (buffer still works).
  - `log_event(agent, event, **fields)`; `snapshot(last=20)`; `__len__`.
  - Thread-safe (one lock for deque + file); lazy file creation; missing
    parent directories created on first write; failures swallowed + logged.
  - Injectable `_writer` seam (module-level or instance attribute) for tests.
- **Unit tests** (new `tests/test_agent_monitor.py`):
  - `log_event` appends to the ring buffer; `snapshot` returns the last N in
    chronological order; buffer respects `buffer_size` (oldest dropped).
  - With a temp `log_file`: events are written as one JSON object per line,
    each with a `ts` (ISO-8601, parseable), `agent`, `event` keys; file and
    parent directory are created lazily on first event.
  - Concurrency: 8 threads × 25 events each → file has exactly 200 complete,
    JSON-parseable lines (no interleaving), buffer has the last 100.
  - `log_file=None` → no file is created, buffer still populated.
  - A `_writer` that raises does not propagate (event swallowed, buffer may
    or may not have it, but `log_event` returns normally).
- **Commit**: `feat(monitor): AgentMonitor event bus with JSONL log + ring buffer`

### Task 3 — Wire monitoring into `AgentPool.ask()` (+ `calls` flag in common.py)

- In `src/common.py`:
  - extend `_print_call_block` to prefix the header with a thread-local
    agent label when set: read `_call_context.agent_label` (default `""`);
    header becomes `[CALL] <name> (<label>)(` when non-empty. No other
    changes; `show_dump` semantics untouched.
- In `src/live_ai.py`:
  - `AgentPool.__init__` builds `self.monitor = AgentMonitor(log_file,
    buffer_size)` from the settings; expose it module-level via
    `bind_agent_monitor(pool.monitor)` called from `main.py` next to
    `bind_agent_pool` (update `main.py` accordingly).
  - `ask()`: 
    - set `_call_context.agent_label = name` for the duration of `act()`;
    - emit `start` (with prompt) before `act()` and `done` (with answer,
      total ms) after, via `self.monitor.log_event`;
    - wrap `agent._execute_tool` with a tool-event closure (`tool_start` /
      `tool_end` with ok/ms/error), restored in `finally`;
    - `set_subagent_dump(False)` only when `"calls"` not in the monitor
      flags (when on, sub-agent `[CALL]` blocks print, labeled).
  - `ask_parallel` needs no changes (it funnels through `ask`).
- **Unit tests** (extend `tests/test_agent_pool.py` / new tests in
  `tests/test_agent_monitor.py`):
  - `_print_call_block` label: with `_call_context.agent_label = "x"`, the
    block header contains `(x)`; without it, unchanged; `show_dump=False`
    still suppresses everything.
  - Pool-level (offline agent via the `_offline_agent` pattern, stubbed
    client that returns one tool call then a final answer): after `ask()`,
    the monitor buffer contains exactly `start` → `tool_start` → `tool_end`
    → `done` for the right agent, in order; `done` carries the answer and a
    positive `ms`; `tool_end` carries `ok`/`ms`.
  - `monitor` flags empty (default) → `set_subagent_dump` was called with
    `False` during `act()` (assert via monkeypatched spy) and no events are
    recorded; with `"calls"` → dump not suppressed.
  - The original `agent._execute_tool` is restored after `ask()` (identity
    check) even when the fake tool raises.
- **Commit**: `feat(monitor): wire AgentMonitor into AgentPool.ask + labeled CALL blocks`

### Task 4 — `agent_activity` tool + registration

- Add module-level `agent_activity(last: int = 20) -> str` in
  `src/live_ai.py` (docstring in the usual tool style; returns formatted
  one-line events or `No sub-agent activity recorded.`); reads from the
  module-level monitor bound by `bind_agent_monitor`; safe when unbound.
- Format each event as `<HH:MM:SS> <agent> <event> <key=val ...>` (details
  joined from the record fields, truncated to ~120 chars per line).
- Add `agent_activity` to `_default_tools()` (import from the module itself —
  it lives in `src/live_ai.py` like `ask_agent`).
- Update `main.py` to call `live_ai_mod.bind_agent_monitor(pool.monitor)`
  after building the pool (only when the pool exists).
- Mention the feature in `README.md` (short section: `[agent] monitor` flags,
  log file location, `agent_activity` tool).
- **Unit tests**:
  - `agent_activity()` unbound → empty message; after `bind_agent_monitor`
    with a monitor holding 3 events → 3 formatted lines, newest last;
    `last=2` → only the 2 most recent.
  - `tests/test_live_ai.py`-style check: the tool list of a fresh (offline)
    `Agent` includes `agent_activity` and its OpenAI tool definition is
    well-formed (name/description/parameters present).
- **Commit**: `feat(monitor): agent_activity tool + registration + README`

### Task 5 — End-to-end smoke + full suite green

- Add an integration-style test (mark like the existing
  `test_integration.py` markers): with an offline pool (stubbed client,
  `monitor = "calls,log"`), run one `ask()` and one `ask_parallel()` with two
  agents; assert the JSONL file contains the expected event sequence per
  agent and `agent_activity()` returns a non-empty report mentioning both
  agent names.
- Run the **full** test suite (`python -m pytest tests/`) and fix any
  regressions (in particular the existing `tests/test_agent_pool.py` settings
  assertions updated in Task 1).
- Verify manually once (optional, needs a working model): set
  `monitor = "calls,log"` in `config.toml`, run the app, delegate a task via
  `ask_agent`, confirm labeled `[CALL]` blocks appear live and
  `data/agents.jsonl` fills up; then ask the main agent "what did the
  sub-agents do?".
- **Commit**: `test(monitor): end-to-end monitoring smoke + suite green`

## 6. Verification checklist (definition of done)

- [ ] `python -m pytest tests/` passes with the new tests from all tasks.
- [ ] With `[agent] monitor` unset, behavior is byte-identical to today
      (quiet sub-agents, no events recorded, existing tests untouched except
      the settings-dict assertions extended in Task 1).
- [ ] `monitor = "calls"` prints labeled `[CALL]` blocks for sub-agent tool
      calls, live, without breaking the main agent's output or spinner.
- [ ] `monitor = "log"` writes valid JSONL (one object per line, UTC `ts`)
      to the configured file; parallel agents never interleave lines.
- [ ] `agent_activity()` lets the main agent report recent sub-agent
      activity on user request.
- [ ] Monitoring failures never break a delegation (best-effort only).
- [ ] One commit per task, in order, with the messages above.
