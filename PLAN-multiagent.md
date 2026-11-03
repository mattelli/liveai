# Plan: Asynchronous Multi-Agent Support for live_ai (flat, main-agent-only)

## Goals

1. The main session agent can delegate tasks to named sub-agents that run
   **asynchronously** (fire-and-forget): the parent's `act()` loop is never
   blocked, and the parent can keep working while children run.
2. Sub-agents are declared in **`config.toml`** (a new `[agents]` section:
   name → tool_groups / model / system prompt overrides), built from the same
   config pipeline used by the scheduler executor.
3. **Flat topology only**: the main agent delegates to children; children get
   no delegation tool and cannot spawn further agents.
4. Child results are routed back into the main session via the existing
   `queue_agent_result` / `drain_agent_results` machinery, so the user (and the
   parent agent on its next turn) can see what the children produced.
5. Child results can be **summarized before being shown** (optional, config
   flag): long child outputs are condensed by an LLM summary pass before they
   reach the UI drain queue, keeping the displayed result (and any future
   injection into the parent's context) lean.

## Current behavior

- `Agent(config, isolated_tools=True, quiet=True)` already builds tool-isolated,
  quiet agents (used by the scheduler executor, `src/tools/scheduler.py:456-467`).
- Background agents already queue results for the UI via
  `queue_agent_result(kind, result_id, text)` (`live_ai.py:563`) and the session
  drains them (`drain_agent_results`, `drain_agent_results_of_kind`).
- Parallel tool execution with `contextvars.copy_context()` per worker exists in
  `Agent._execute_tool_calls` (`live_ai.py:2434-2467`), so thread-local monitor
  wiring and agent naming survive background execution.
- Tool selection by topic exists: `_default_tools` + `_select_tools`
  (`tool_groups`), and `Agent._build_tools` (line 1875) honors `isolated_tools`
  to skip module-default binding.
- There is **no** delegation tool, no `[agents]` config section, no registry of
  named sub-agents, and no way for the parent to discover or launch one.

## Expected behavior

- New `[agents]` TOML section, e.g.:

  ```toml
  [agents.researcher]
  tool_groups = "web, files"
  model = "local-fast"        # optional: models.json entry; default = main model
  system_prompt = "You are a research assistant..."   # optional
  max_act_rounds = 50         # optional

  [agents.writer]
  tool_groups = "files, memory"
  ```

  Missing/empty section → feature disabled, no delegation tool registered.

- Optional result summarization (`[agents] summarize_results`):

  ```toml
  [agents]
  summarize_results = true     # false (default) -> child results shown as-is
  # summarize_max_chars = 800  # results longer than this get summarized (0 = always)
  ```

  When enabled, a child's final text longer than `summarize_max_chars` is
  condensed by an LLM summary pass (reusing the `summarizer_system` prompt
  pattern in `_PROMPTS` / `src.tools.memory`) before
  `queue_agent_result("delegation", ...)` is called. Short results pass through
  unchanged. The summary uses the same OpenAI-compatible endpoint as the child's
  model entry; on summary failure the original text is queued unchanged (never
  dropped).

- New tool `delegate_task(agent_name: str, task: str) -> str`, available only to
  the main session agent:
  - Validates the name against the `[agents]` registry (unknown name →
    `AgentToolArgumentError`).
  - Spawns a **daemon thread** running an isolated `Agent` (own tool instances,
    own history, `quiet=True`) with the child's configured tool groups / model /
    system prompt.
  - Returns immediately with an acknowledgment string, e.g.
    `"Delegated to 'researcher' (task_id=agent-1). Results will arrive via the
    agent_results queue."`
  - On completion, the child queues its final text with
    `queue_agent_result("delegation", task_id, text)`; failures queue an error
    text instead of raising.
  - Depth guard: children are built with delegation unavailable (flat topology),
    enforced by simply not registering `delegate_task` in child tool sets.
  - Cancellation: a `/cancel`-style request is out of scope for v1 (documented
    limitation); children run to completion or until their `max_act_rounds`.

- The parent's system prompt gains a managed-agents section listing each
  configured child (name + description), mirroring smolagents'
  `initialize_system_prompt`, so the model knows delegation exists.

- UI/session: `drain_agent_results_of_kind("delegation")` results surface in the
  session poll exactly like scheduler results (no new UI machinery needed).

## Tasks (one commit per task)

1. **Config parsing** — add `[agents]` section parsing/validation
   (`_load_agent_registry`): names, tool_groups, optional model/system_prompt/
   max_act_rounds; unknown keys warned and ignored; unit-testable pure function.
2. **Sub-agent factory** — `build_sub_agent(name, entry, config)` returning an
   isolated, quiet `Agent` with the child's tool groups/model/prompt; unit test
   with a fake config asserts tool selection and isolation flags.
3. **`delegate_task` tool** — registry lookup, thread spawn, immediate ack
   return, result queuing on completion (success and error paths); optional
   LLM summarization of long child results before queuing (config
   `summarize_results` / `summarize_max_chars`, fail-open to the original
   text); unit tests with a stubbed child agent (no LLM), including
   unknown-name and disabled-feature cases.
4. **Wiring** — register `delegate_task` in the main agent's default tools only
   when the registry is non-empty; append the managed-agents section to the
   parent system prompt; keep children flat (no delegation tool).
5. **Docs** — README/config.toml comments for the `[agents]` section; note the
   v1 limitations (no cancellation, no hierarchical delegation).

## Verification (unit tests)

- `test_agents_config.py`: valid/invalid/missing `[agents]` sections; shorthand
  string entries; unknown keys ignored; duplicate names rejected.
- `test_delegate_task.py`:
  - delegation returns an ack containing the task id;
  - child runs in a background thread (parent returns before child completes —
    use an event to synchronize);
  - child result lands in `drain_agent_results_of_kind("delegation")`;
  - with `summarize_results = true`, a long child result is replaced by a
    (stubbed) summary; short results pass through unchanged; a summary
    failure queues the original text;
  - child failure queues an error text, does not raise into the parent;
  - unknown agent name → `AgentToolArgumentError`;
  - empty registry → tool not registered / no-op error;
  - child tool set contains no `delegate_task` (flat topology).
- Existing scheduler/agent tests must keep passing (no behavior change when
  `[agents]` is absent).
