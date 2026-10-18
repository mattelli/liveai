# PLAN.md — Multi-Agent Interaction for `live_assistant`

*Derived from `multi-agent-analysis.md` (§2 agent-as-tool, §3 parallelism,
§4 `[agent]` config, §5 accepted solution: per-agent tool instances).*

This plan is written to be executed by an AI agent, task by task.
**One git commit per task** — never mix tasks in a single commit.

---

## 1. Goals

1. Let the running agent (agent-1, the one the user talks to) **delegate
   tasks to named sub-agents** (agent-2, …) that run their own tool loop and
   return their final answer as a tool result (agent-as-tool pattern).
2. Support **parallel execution** of independent subtasks
   (fan-out / map-reduce via a thread pool; LLM calls are I/O-bound).
3. Declare the pool in **`config.toml` → `[agent]`**: number of parallel
   agents and the model each one uses (models.json entry names).
4. Eliminate the **shared tool-module state** hazard: every pooled agent gets
   **its own tool instances** (accepted solution — no locks, no thread-locals).
5. Keep workers **quiet**: no console interleaving, no speech.

## 2. Current behavior

- `main.py` builds **exactly one** `Agent` (`src/live_ai.py`) and hands it to
  `ConversationSession`. There is no pool, no delegation, no parallelism.
- `Agent.act(prompt)` is a self-contained round loop (max
  `MAX_ACT_ROUNDS = 20`): stream → run tools → feed results → repeat.
- Tools are plain callables from `_default_tools(agent)`; session-bound state
  lives in **module-level globals**: `src/tools/memory.py::_history_provider`
  (bound by `Brain.__init__` via `bind_history`) and
  `src/tools/vision.py::_session` (bound via `bind_session`). A second agent
  constructed later would **rebind these globals**, and parallel agents would
  fight over one binding.
- `Agent` always prints to the console (`StreamPrinter`, `cprint`) and
  `_handle_round_end` calls `speech_tools.speak_if_enabled(...)`.
- `[agent]` config holds only `bootstrap_prompt`.

## 3. Expected behavior

- The user asks agent-1 something; agent-1 may call
  `ask_agent(agent="agent-2", prompt="...")`. Agent-2 runs its own `act()`
  loop with its own tools (e.g. `web_search`) and its **final assistant text**
  is returned to agent-1 as a normal `role: "tool"` message.
- Agent-1 may call `run_parallel_agents(tasks=[{"agent": ..., "prompt": ...},
  ...])`; the sub-agents run **concurrently** (ThreadPoolExecutor) and a JSON
  list of `{"agent", "prompt", "answer"}` is returned for aggregation.
- Pool size and per-agent models come from `[agent] parallel_agents` and
  `[agent] agent_models` (models.json entry names; missing keys default to
  `0` / `[]` → single-agent behavior unchanged).
- Each pooled agent owns **its own tool instances**; `bind_history` /
  `bind_session`-style state is per-instance, never shared module state.
- Workers are quiet (buffered output, speech off); only agent-1 prints/speaks.
- With `parallel_agents = 0` (default) the app behaves exactly as today.

## 4. Ground rules for every task

- **One commit per task.** Commit message: `task N: <short summary>`
  (e.g. `task 3: quiet mode for pooled agents`). Commit only after the task's
  tests pass.
- Run the full suite before committing: `python -m pytest tests/ -x -q`.
- Follow existing test style: offline stubs only (see `tests/test_live_ai.py`:
  `_BareAgent`, `_fake_openai`, scripted `_stream_round`); never hit the
  network or load real models in unit tests.
- Backward compatibility is a hard requirement: all existing tests must keep
  passing unchanged.

---

## Task 1 — Config schema + settings helper  → *commit 1*

**Files:** `config.toml`, `src/live_ai.py` (or `src/common.py`)

**Do:**

1. Extend `[agent]` in `config.toml`:

   ```toml
   [agent]
   bootstrap_prompt = "..."          # unchanged
   parallel_agents = 0               # extra agents beside the running one (0 = off)
   agent_models = []                 # models.json entry names, one per agent
   # agent_system_prompts = []       # optional per-agent system prompts
   # agent_quiet = true              # workers don't print/speak (default true)
   ```

2. Add a pure helper (no I/O) in `src/live_ai.py`:

   ```python
   def agent_pool_settings(config) -> dict:
       """[agent] multi-agent settings with backward-compatible defaults."""
       agent = getattr(config, "agent", None)
       n = int(getattr(agent, "parallel_agents", 0) or 0)
       models = list(getattr(agent, "agent_models", []) or [])[:n]
       prompts = list(getattr(agent, "agent_system_prompts", []) or [])
       return {
           "count": n,
           "models": models,
           "system_prompts": (prompts + [None] * n)[:n],
           "quiet": bool(getattr(agent, "agent_quiet", True)),
       }
   ```

**Unit tests** (new `tests/test_agent_pool.py`):

- `agent_pool_settings` with a full config returns count/models/prompts/quiet.
- Missing `[agent]` keys → `{"count": 0, "models": [], ...}` (defaults).
- `agent_models` longer than `parallel_agents` is truncated to `n`.
- `agent_system_prompts` shorter than `n` is padded with `None`.
- `load_config` on a temp `config.toml` containing the new keys round-trips
  them (note: `load_config` resolves `*_file/*_dir/*_path` suffixes — the new
  keys deliberately avoid those suffixes).

## Task 2 — Per-agent tool instances (accepted shared-state solution)  → *commit 2*

**Files:** `src/tools/memory.py`, `src/tools/vision.py`, `src/live_ai.py`

**Do:**

1. In `src/tools/memory.py`: refactor the module functions into a class
   `MemoryTools` that keeps `history_provider` and the config cache as
   **instance attributes** (`recall_personal`, `store_personal`, `clear_chat`
   become methods with the same signatures/docstrings). Keep the module-level
   functions as thin delegates to a module-default instance so
   `bind_history()` + existing callers/tests keep working.
2. Same for `src/tools/vision.py`: class `VisionTools` with instance state
   (`session`), module-level `bind_session`/`image_response` delegate to a
   module-default instance (backward compatible).
3. In `src/live_ai.py`: `Agent._build_tools()` returns **fresh instances
   bound to `self`** for the session-bound tools:

   ```python
   def _build_tools(self) -> list[Callable]:
       tools = _default_tools(self)
       # per-agent instances: no shared module state between parallel agents
       return [
           tool.for_owner(self) if hasattr(tool, "for_owner") else tool
           for tool in tools
       ]
   ```

   (`for_owner` is provided by `MemoryTools`/`VisionTools`; other tools stay
   stateless module functions.)

**Unit tests** (extend `tests/test_agent_pool.py`, plus keep
`tests/test_tools.py` green):

- Two agents built via `Agent.__new__` + `_build_tools()` get **different**
  `MemoryTools`/`VisionTools` instances (`a is not b`).
- Binding history on agent A does not change what agent B's
  `MemoryTools.clear_chat` sees (isolation).
- Module-level `recall_personal`/`image_response` still work after binding
  the module-default instance (backward compat).
- Existing `_default_tools` name-list tests in `tests/test_live_ai.py`
  still pass (tool names unchanged).

## Task 3 — Quiet mode for pooled agents  → *commit 3*

**Files:** `src/live_ai.py`

**Do:**

1. Add a `quiet: bool = False` flag to `Agent.__init__` (stored as
   `self.quiet`).
2. When `quiet`: use a no-op stream printer (buffer-only `StreamPrinter` or a
   `NullPrinter` with the same interface: `feed`, `flush`, `note`,
   `start_spinner`, `stop_spinner`, `mark_line_clean`), skip the
   `cprint(f"\n🤖 [{model}]")` header, and skip
   `speech_tools.speak_if_enabled(...)` in `_handle_round_end`.

**Unit tests** (extend `tests/test_agent_pool.py`):

- A quiet agent's `_handle_round_end` never invokes speech (monkeypatch
  `src.tools.speech._speech_generator` with a recorder, as
  `test_cancelled_round_does_not_speak` does).
- A quiet agent writes nothing to stdout during `act()` (capsys).
- A non-quiet agent still speaks/prints (regression guard).
- Fragments still accumulate in `_fragments_buffer` in quiet mode.

## Task 4 — `AgentPool`  → *commit 4*

**Files:** `src/live_ai.py`

**Do:**

1. Add `AgentPool`:

   ```python
   class AgentPool:
       """Named sub-agents; each has its own model, history and tools."""

       def __init__(self, config, settings: dict):
           # build one Agent per entry in settings["models"], each with a
           # deepcopy(config), its system prompt (when given), quiet=True,
           # and its model resolved from the models.json entries
           # (Brain._load_model_entries / _entry_connection pattern).

       def ask(self, agent_name: str, prompt: str) -> str:
           # agent.act(prompt); return the final assistant text
           # (agent.messages[-1]["content"]).

       def ask_parallel(self, tasks: list[tuple[str, str]]) -> list[str]:
           # ThreadPoolExecutor(max_workers=len(tasks)); order preserved.
   ```

2. Model resolution: entries come from the **models.json file**
   (`[brain] models_file`); unknown name → `ValueError` at pool build time
   (fail fast, before the session starts).

**Unit tests** (extend `tests/test_agent_pool.py`, reuse the offline stubs):

- Pool builds one agent per configured model (offline `_fake_openai` stub +
  temp models.json, mirroring `_switch_brain` in `tests/test_live_ai.py`).
- `ask()` returns the scripted final assistant text (script
  `_stream_round` like `_bare_act_agent` does).
- `ask_parallel()` with two tasks returns results **in task order**, and both
  target agents actually ran (record calls).
- Unknown agent name in `ask()` raises a clear `KeyError`.
- Unknown model name at build time raises `ValueError` mentioning the name.

## Task 5 — Delegation tools `ask_agent` + `run_parallel_agents`  → *commit 5*

**Files:** `src/live_ai.py`

**Do:**

1. Add two tool callables with a pool seam (same pattern as
   `src/tools/memory.py::_history_provider`):

   ```python
   _pool_provider = None  # seam: callable returning the live AgentPool

   def bind_agent_pool(provider) -> None: ...

   def ask_agent(agent: str, prompt: str) -> str:
       """Delegate a task to a named sub-agent and return its final answer.
       The sub-agent can use its own tools to accomplish the task."""
       # no pool bound -> "Error: no agent pool is bound."

   def run_parallel_agents(tasks: list[dict]) -> str:
       """Run several sub-agents concurrently. Each task:
       {"agent": name, "prompt": text}. Returns a JSON list of
       {"agent", "prompt", "answer"} for aggregation."""
   ```

2. Register both in `_default_tools(agent)` (after the pool tools exist, the
   function-calling machinery picks them up via `_openai_tool_definition`).

**Unit tests** (extend `tests/test_agent_pool.py`):

- `_build_tools()` now includes `ask_agent` and `run_parallel_agents`.
- `ask_agent` with no bound pool returns the error string (never raises).
- With a fake pool bound, `ask_agent` forwards and returns the answer.
- `run_parallel_agents` returns valid JSON with `agent`/`prompt`/`answer`
  keys, order matching the input tasks.
- Malformed task entries (missing `agent`/`prompt`) produce a per-task error
  entry, not a crash.

## Task 6 — Wire the pool in `main.py`  → *commit 6*

**Files:** `main.py`

**Do:**

1. After `configure_tools(config)`:

   ```python
   settings = agent_pool_settings(config)
   if settings["count"] > 0:
       pool = AgentPool(config, settings)
       bind_agent_pool(lambda: pool)
   ```

   (`parallel_agents = 0` → nothing is built; current behavior unchanged.)

**Unit tests** (extend `tests/test_main.py` conventions):

- With `parallel_agents = 0`, no pool is built and `bind_agent_pool` is never
  called (monkeypatch `AgentPool`/`bind_agent_pool` with recorders).
- With `parallel_agents = 2` + two models.json entries, `AgentPool` is built
  once with the resolved settings and the provider returns it.
- `main(["--prompt", "hi", "--no-tools"])` still runs end-to-end offline
  (existing integration-style stubs).

## Task 7 — End-to-end scripted verification  → *commit 7*

**Files:** `tests/test_agent_pool.py`

**Do:** write the two scenario tests from `multi-agent-analysis.md` §2/§3,
fully offline (scripted rounds, fake pool, fake stream):

1. **Delegation:** agent-1's scripted round emits
   `ask_agent(agent="agent-2", prompt="Get current weather in Torino")`;
   agent-2's scripted rounds emit a `web_search` tool call then a final
   answer. Assert: agent-2's tool ran, agent-1's history contains the
   `role: "tool"` result with agent-2's final answer, and agent-1's final
   text merges it.
2. **Fan-out:** `run_parallel_agents` with two tasks on two fake agents;
   assert both ran, results are in order, and the JSON aggregates correctly.

**Final gate before the last commit:**

```
python -m pytest tests/ -q          # full suite green
git log --oneline                   # 7 commits, one per task
```

---

## 5. Out of scope (from the analysis, not planned here)

- Agent-to-agent streaming of partial output (only final answers cross the
  boundary — keeps contexts small, per §5 "Context & cost").
- Cancellation propagation into running workers (§5 "Cancellation": Ctrl+C
  flags the orchestrator only; workers finish their round).
- Per-agent tool allow-lists / locks for side-effect tools (`run_command`,
  `write_text_file`, …) — noted as a future hardening step in §5.
- VRAM-aware scheduling for local servers (§5 "Local servers").

## 6. Risks & fallbacks

| Risk | Fallback |
|---|---|
| Refactoring `memory`/`vision` breaks existing tests | The module-level delegates keep old behavior; if a test still fails, fix the delegate, never the test's expectations |
| Local server cannot load 2 models at once (VRAM) | Set `parallel_agents = 1`, or point worker entries at cloud endpoints in models.json |
| Model lacks tool-calling | Pool build fails fast (Task 4 validation); document in README |
| Deadlock via nested delegation (agent-2 calls `ask_agent`) | `ask_agent` refuses when called from a pooled agent (guard flag on the Agent instance); add a unit test for it in Task 5 |