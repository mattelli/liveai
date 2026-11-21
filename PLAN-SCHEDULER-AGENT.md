# PLAN — Configurable scheduled agent (`[delegation.scheduler] agent`)

## 1. Goal

Let the scheduled-task executor run on a **named sub-agent** instead of always
running on an agent built from the main `[brain]` config, via a new key:

```toml
[delegation.scheduler]
agent = "researcher"   # name of a [delegation.agents.<name>] entry; unset = main config
```

Expected behavior:

- `agent` unset/missing → **exactly today's behavior**: the executor is
  `Agent(config, isolated_tools=True, quiet=True)` built from the main config.
- `agent = "researcher"` → the executor is built from the
  `[delegation.agents.researcher]` spec (its `model`, `tool_groups`,
  `system_prompt`, `max_act_rounds`), reusing the existing sub-agent builder.
- `agent` names an unknown entry → a clear warning at startup and a **fallback
  to the main-config agent** (scheduled tasks must keep running).
- The run is still monitored and its answer still delivered to the main agent
  exactly as today.

Out of scope: a per-task agent override (`schedule_task(..., agent=...)`); the
executor is one process-wide agent, built lazily once and reused.

## 2. Current behavior

`src/tools/scheduler.py`:

- `configure(config)` (line 72) keeps `[delegation.scheduler]` as
  `_CONFIG = {enabled, persist_file, on_missed, config}` — the **full** config
  is stashed under `"config"` (line 86) precisely so the executor can build an
  agent.
- `_default_scheduler_factory()` (line 63) wires `_execute=_agent_executor`.
- `_agent_executor(task)` (line 501) lazily builds `_EXECUTOR_AGENT` via
  `_AGENT_FACTORY or _default_agent_factory`, then `agent.act(task.prompt)`.
- `_default_agent_factory()` (line 457) → `Agent(config, isolated_tools=True,
  quiet=True)`. **No model/tool/prompt selection**: it always uses the main
  `[brain]` model and the main `[agent] tool_groups`.

By contrast `delegate_task` (`src/delegation.py`) *does* select by name: it
looks the name up in the `[delegation.agents]` registry
(`src/agent_registry.py`) and builds the child with
`src/sub_agents.py::build_sub_agent(config, spec)`, which applies the spec's
`model`/`tool_groups`/`system_prompt`/`max_act_rounds` on a deep-copied config
(`child_config`) and returns `Agent(child_config, isolated_tools=True,
quiet=True)`.

So the missing piece is small: the scheduler already builds an isolated+quiet
agent; it just needs to build it **from a spec** when one is configured.

## 3. Design

### 3.1 Config key and resolution (T1)

`src/tools/scheduler.py::configure()` gains one key:

```python
_CONFIG = {
    "enabled": ...,
    "persist_file": ...,
    "on_missed": ...,
    "config": config,
    "agent": getattr(section, "agent", "") or "",   # new: entry name
    "agent_spec": <resolved AgentSpec or None>,     # new: resolved once
}
```

Resolution helper:

```python
def _resolve_agent_spec(config, name: str):
    """The AgentSpec named by ``[delegation.scheduler] agent`` (None if unset).

    Logs a warning and returns None for an unknown name: a typo must not stop
    scheduled tasks from running (they fall back to the main-config agent).
    """
    name = (name or "").strip()
    if not name:
        return None
    from src.agent_registry import agent_registry
    specs = agent_registry()
    spec = specs.get(name) or specs.get(name.lower())
    if spec is None:
        logger.warning(
            "[delegation.scheduler] agent %r is not a configured "
            "[delegation.agents] entry (configured: %s); using the main agent",
            name, ", ".join(sorted(specs)) or "none")
    return spec
```

Note on ordering: the registry is populated by `load_agent_registry(config)`
inside the **main** `Agent.__init__` (`src/live_ai.py:2248-2250`), which runs
*after* `configure_tools(config)` (main.py:35) and therefore after
`scheduler.configure()`. So the registry is normally **empty** when
`configure()` runs. `_resolve_agent_spec()` therefore parses the registry
itself (`load_agent_registry(config)`) when it finds it empty — a pure,
idempotent function of the config that the main agent reloads anyway — so a
configured name resolves at startup instead of only on the first scheduled
run. `configure()` also re-resolves lazily on the first executor build (see
3.2) as a second safety net.

### 3.2 Build the executor from the spec (T2)

`_default_agent_factory()` keeps its **zero-argument signature** (so the
`_AGENT_FACTORY` test seam and all existing tests are untouched) and reads the
spec from `_CONFIG`:

```python
def _default_agent_factory():
    config = _CONFIG.get("config")
    if config is None:
        return None
    spec = _CONFIG.get("agent_spec")
    if spec is None and _CONFIG.get("agent"):
        spec = _resolve_agent_spec(config, _CONFIG["agent"])  # late resolution
        _CONFIG["agent_spec"] = spec
    if spec is not None:
        from src.sub_agents import SubAgentBuildError, build_sub_agent
        try:
            return build_sub_agent(config, spec)   # isolated + quiet + spec overrides
        except SubAgentBuildError as exc:
            logger.warning("Cannot build the scheduled agent %r: %s; "
                           "falling back to the main agent", spec.name, exc)
            # fall through to the main-config path below
    try:
        from src.live_ai import Agent
        return Agent(config, isolated_tools=True, quiet=True)
    except Exception as exc:
        logger.warning("Cannot build the scheduler execution agent: %s", exc)
        return None
```

`build_sub_agent` already returns `Agent(child_config, isolated_tools=True,
quiet=True)` — the same flags the scheduler needs — plus the spec's model,
tool_groups, system_prompt and max_act_rounds. No new agent-building code.

`src/sub_agents.py` imports only `AgentSpec` at module level and imports
`src.live_ai` lazily inside its factory, so `scheduler → sub_agents` adds no
import cycle (the `src.live_ai` import stays lazy).

### 3.3 Monitoring label (T2)

`_agent_executor` sets the thread-local name for the activity monitor:

```python
label = spec.name if spec is not None else "scheduler"
_set_executor_name(label)
monitor.record(label, "start", ...)
```

So a scheduled run on the configured agent shows up under its own name in
`agent_activity()`; unset keeps the historical `"scheduler"` label (existing
tests assert it).

### 3.4 Config files + docs (T3)

`config.toml` / `config-code.toml`, under `[delegation.scheduler]`:

```toml
agent = ""   # run scheduled prompts on this [delegation.agents.<name>] entry
             # (unset/empty = the main agent's model and tool groups)
```

README: extend the scheduler section with the new key.

## 4. Verification — unit tests

`tests/test_scheduler_agent.py` (existing seams: `_AGENT_FACTORY`,
`_CONFIG`, `reset_executor_agent`):

| Test | Asserts |
|------|---------|
| `test_configure_reads_agent_key` | `configure()` stores `agent` and resolves `agent_spec` from the registry |
| `test_configure_without_agent_key_has_no_spec` | missing key → `agent == ""`, `agent_spec is None` |
| `test_configure_unknown_agent_warns_and_falls_back` | unknown name → warning logged, `agent_spec is None` |
| `test_default_agent_factory_uses_spec` | with a spec set, the factory calls `build_sub_agent(config, spec)` (stubbed) and returns its agent |
| `test_default_agent_factory_spec_build_error_falls_back` | `build_sub_agent` raising `SubAgentBuildError` → warning + main-config `Agent` |
| `test_default_agent_factory_without_spec_unchanged` | no spec → `Agent(config, isolated_tools=True, quiet=True)` (existing test, kept) |
| `test_executor_uses_configured_agent_name_as_label` | monitor/thread-local name is the spec name when configured |
| `test_executor_label_is_scheduler_without_agent` | unset → `"scheduler"` (existing behavior) |

`tests/test_sub_agents.py` already covers `build_sub_agent` itself (model
routing, spec overrides), so the scheduler tests only need to prove the
**wiring**.

Regression: the full suite (`python -m pytest tests/ -q`) must stay green —
in particular `test_scheduler_agent.py::test_default_agent_factory_builds_isolated_agent`
and `tests/test_scheduler_e2e.py`, which rely on the zero-arg seam and the
no-spec path.

## 5. Tasks — one commit per task

Proceed **one task at a time**; each task ends with its tests green and a
commit whose message details what was done. **Ask for confirmation before
starting the next task.**

| Task | Content | Commit message (draft) |
|------|---------|------------------------|
| **T1** | `[delegation.scheduler] agent` key: `configure()` stores it, `_resolve_agent_spec()` resolves it against the registry with a warning on unknown names. Behavior unchanged (the spec is not used yet). Tests: resolution, missing key, unknown name. | `feat(scheduler): read the [delegation.scheduler] agent key` |
| **T2** | `_default_agent_factory()` builds from the spec via `build_sub_agent` (with fallback to the main-config agent), and `_agent_executor` labels the run with the agent name. Tests: spec used, build-error fallback, label, no-spec unchanged. | `feat(scheduler): run scheduled tasks on the configured sub-agent` |
| **T3** | `config.toml` + `config-code.toml` key and README documentation. | `docs: document the scheduled-agent option` |

## 6. Notes / risks

- **No new agent-building code**: the spec path reuses `build_sub_agent`, so
  the scheduled agent behaves exactly like a `delegate_task` child (same
  isolation, same model routing, same flat topology — no `delegate_task` tool).
- **Backward compatible**: no key → byte-for-byte the current behavior; the
  zero-arg `_AGENT_FACTORY` seam and every existing scheduler test are kept.
- **Fail-soft**: an unknown name or a build failure logs a warning and falls
  back to the main-config agent rather than stopping the scheduler thread.
- The executor is built **once** and cached in `_EXECUTOR_AGENT`;
  `reset_executor_agent()` (already present) drops it, so a model switch or a
  config reload can rebuild it.
- If you would rather **fail loudly** on an unknown name (raise at startup
  instead of falling back), say so and I will change 3.1/3.2 accordingly.
