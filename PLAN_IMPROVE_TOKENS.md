# PLAN_IMPROVE_TOKENS.md — Reducing token consumption of the agents

Cut the number of tokens the agents defined in `src/live_ai.py` consume per
turn, without changing what the agents can do. Measured on the current code:
**44 tools → ~21.9k chars of JSON tool schemas (~5.5k tokens) sent on every
request of every agent**, plus unbounded tool results and an ever-growing
history in multi-round `act()` turns.

Work through the tasks **in order**; make **one git commit per task**
(message given in each task). Do not start a task before the previous one is
committed and its tests pass. Every new config key **defaults to today's
behavior** (backward compatible): existing tests must stay green untouched.

---

## 1. Goals

- **Bounded tool results**: a tool result stored in the chat history is
  capped (configurable), so one `read_text_file` / `run_command` /
  `extract_webpage` call cannot inject tens of thousands of tokens into
  every later request of the turn.
- **Lean tool schemas**: the JSON tool definitions sent with every request
  carry no `default` values and (optionally) truncated docstrings — the
  fixed ~5.5k-token payload shrinks without losing tool discoverability.
- **Configurable tool set**: the user can exclude whole tool groups
  (finance, archives, speech, ...) via config; excluded tools cost zero
  tokens (no schema, no description).
- **Bounded history growth**: when the context usage passes a configurable
  threshold, the oldest *complete* tool-call rounds are pruned from the
  history (system prompt and the most recent messages always kept), so long
  multi-round turns do not grow without limit.
- **No behavior regression**: with all new keys unset/0 the runtime behavior
  is byte-identical to today; the monitor, delegation, MCP injection and
  cancellation paths are untouched.
- **Measurable**: unit tests assert the schema payload stays under a char
  budget and that truncation/pruning actually shrink the history.

## 2. Current behavior

- `Agent._build_tools()` (`src/live_ai.py`) always returns the full list of
  44 tools from `_default_tools()`; `Agent.__init__` builds
  `self.tool_definitions` via `OpenAICore._openai_tool_definition(tool)` and
  `act()` sends **all** definitions with **every** round of **every** agent
  (main agent and pooled workers alike).
- `_openai_tool_definition()` embeds each tool's **full `__doc__`** and,
  for every parameter with a default, a `schema.setdefault("default", ...)`
  entry. Docstrings are long (e.g. `ask_agent` ~450 chars; the biggest
  schemas are `generate_image` 1173, `zip_files` 968, `web_images` 883
  chars). Defaults are informational only — models never need them.
- In `Agent.act()`, each tool result is appended to `self._messages` **as
  returned by the tool, uncapped** (`content=result`); only the monitor
  event is capped (`result[:200]`). A single big tool output is then re-sent
  on every subsequent round of the turn.
- `Agent._context_usage()` computes an approximate usage percentage, but
  **nothing acts on it**: the history is never pruned or summarized; a
  500-round turn (`[agent] max_act_rounds = 500` in config.toml) can grow
  until the context window is exhausted.
- Tool selection is all-or-nothing: `_TOOL_TOPICS` already groups tools by
  topic (used only by the `/tools` launcher in `session.py`), but there is
  no config to exclude a group from the agents.
- Config today (`config.toml`): `max_response_tokens = 0.1` (good),
  `show_thinking = true` with `max_thinking_tokens = 0` (display-only, but
  reasoning tokens may be billed by providers), `max_act_rounds = 500`.

## 3. Expected behavior

- `[brain] max_tool_result_chars = 4000` (example): every tool result is
  truncated before being appended to the history, with an explicit marker:
  `<first 4000 chars>… [truncated, 12345 chars total]`. The full result is
  still returned to the monitor (`result[:200]` rule unchanged) and to the
  logger (`str(result)[:200]` unchanged). `0` (default) = uncapped, as today.
- `[agent] tool_doc_chars = 300` (example): each tool description in the
  JSON schema is the docstring truncated to N chars (hard cut, no ellipsis
  needed) — one-line descriptions still tell the model when to use a tool.
  `0` (default) = full docstring, as today. `default` keys disappear from
  all parameter schemas unconditionally (schemas stay valid JSON Schema;
  `required` and `properties` are unchanged).
- `[agent] tool_groups = "all"` (default, today's behavior) or a
  comma-separated list of topic names from `_TOOL_TOPICS` (e.g.
  `"model, delegation, memory, files, web, documents, images, system, mcp"`);
  only the tools of the listed groups are registered. Unknown group names
  are ignored with a log line. MCP **server** tools injection is unchanged;
  the `mcp` group only controls the `mcp_list_tools`/`mcp_call_tool` bridge
  tools. Pooled agents inherit the same setting (they deepcopy the config).
- `[brain] context_trim_ratio = 0.7` (example) + `[brain]
  context_keep_messages = 40`: at the start of each `act()` turn, if the
  approximate usage exceeds the ratio, complete oldest tool rounds
  (assistant message with `tool_calls` + its following `tool` messages) are
  dropped until usage is back under the ratio, always keeping the system
  prompt (index 0) and the last `context_keep_messages` messages untouched.
  Plain user/assistant turns older than the keep-window are dropped too.
  `0` (default) = never prune, as today. A one-line system note is appended
  after pruning (`[history pruned: N messages dropped]`) so the model knows
  earlier context was compacted.
- With all keys unset: identical behavior to today, all existing tests green.
- **Runtime toggling**: the user can enable/disable individual tools or whole
  topics during a running session via `/tools on|off <name|topic>`; a
  disabled tool's schema is dropped from every subsequent request
  (immediate token saving) and restored on re-enable. Toggles are
  session-only; `[agent] tool_groups` remains the boot-time default.
- Expected effect with the example values above: schema payload
  ~21.9k → ~12k chars (~5.5k → ~3k tokens per request), plus no unbounded
  tool-result growth and a bounded history in long turns.

## 4. Design notes (read before implementing)

- **Seams (all in `src/live_ai.py` unless noted)**:
  - `_openai_tool_definition(tool)` — static, pure; extend it with an
    optional `max_doc_chars: int = 0` parameter and remove the
    `schema.setdefault("default", ...)` line. `Agent.__init__` passes the
    configured value (`getattr(config.agent, "tool_doc_chars", 0)`).
  - `Agent.act()` — apply `_truncate_tool_result(result)` (new small
    helper honoring `[brain] max_tool_result_chars`) right after
    `result = self._execute_tool(call)`, before the `self._messages.append`
    of the `tool` message. Keep the monitor/log capping where it is.
  - `_TOOL_TOPICS` + `_default_tools()` — add a module-level
    `_select_tools(tools, groups: str | list | None) -> list` that filters
    by topic name (reuse the `name_to_topic` inversion pattern from
    `tool_topics()`); call it from `Agent._build_tools()` with the
    `[agent] tool_groups` value. `"all"`/empty/None → unfiltered.
  - `Agent.act()` start — add `_prune_history()` (new method) called after
    `self._add_user_prompt(prompt)`: compute usage via `_context_usage()`,
    drop oldest *complete* groups while over the ratio. A "complete group"
    = one assistant message containing `tool_calls` plus every subsequent
    `role == "tool"` message with matching `tool_call_id` (never split a
    group — servers reject orphan tool results). Never touch index 0
    (system) or the last `context_keep_messages` entries.
- **Why truncate instead of summarize**: an LLM-generated summary must first
  send the full oversized result to a model (~12.5k tokens for a 50k-char
  result), so it *adds* tokens and per-round latency and risks dropping
  verbatim details (error messages, ids, code lines). Truncation is free,
  deterministic and never distorts what it keeps. The model can still get a
  summary on demand: the truncation marker invites a follow-up
  `summarize_doc` (local extractive LexRank summarizer, no LLM cost) or a
  targeted `query_document` / `search_in_file`. Optional future flag (not in
  scope here): `tool_result_summarize` running the local summarizer on
  oversized *prose* results (webpage/doc text), falling back to truncation
  for code/log-like content or summarizer failure.
- **Config parsing**: reuse the defensive style of `_monitor_flags` /
  `_split_list` (invalid values → default, never raise). Add the keys to
  `config.toml` as commented examples next to their section peers.
- **Test style**: follow `tests/test_live_ai.py` (`_BareAgent` pattern —
  `Agent.__new__`-style agents without SDK connection). Put the new tests
  in a new `tests/test_token_reduction.py`; reuse `conftest.py` fixtures.
  No network, no sleeps: schema tests are pure functions; history tests
  drive `act()` helpers directly or fake `_stream_round`/`_execute_tool`.
- **Do not** touch: `AgentPool`, delegation receipts, `AgentMonitor`,
  MCP injection, `_repair_history`, `CANCELLED_PLACEHOLDER` semantics.

## 5. Tasks

### Task 1 — Cap tool results stored in the history

- Add `_truncate_tool_result(text: str, limit: int) -> str` (module-level,
  next to `_response_token_cap`): `limit <= 0` or `len(text) <= limit` →
  unchanged; else keep the **head and the tail** —
  `text[:head] + f"\n… [truncated, {len(text)} chars total] …\n" + text[-tail:]`
  with `head = limit * 3 // 4` and `tail = limit // 4`. Head+tail matters
  for append-ordered files read whole by tools (e.g. `recall_memory` returns
  the entire `memory.log`, oldest entries first): head-only truncation would
  silently drop the **most recent** memories, so the tail must be kept.
  The marker tells the model the result was cut so it can re-query with a
  targeted tool instead of re-reading everything.
- `Agent.__init__`: resolve `self._max_tool_result_chars` from
  `[brain] max_tool_result_chars` (int, default 0, invalid → 0).
- `Agent.act()`: truncate the result before appending the `tool` message.
- `config.toml`: commented `# max_tool_result_chars = 4000` in `[brain]`.
- **Unit tests** (`tests/test_token_reduction.py`):
  - helper: unchanged under limit; truncated over limit with the marker
    carrying the original length; head+tail split honored (tail = last
    `limit // 4` chars, e.g. a memory entry at the end of `memory.log`
    survives truncation); limit 0 / negative → unchanged.
  - `act()` path: fake `_stream_round` returning one tool call, fake
    `_execute_tool` returning a 50k-char string; assert the `tool` message
    content in `self._messages` is capped and the monitor event still got
    the first 200 chars; with limit 0 the full string is stored.
- **Commit**: `feat(tokens): cap tool results stored in history`

### Task 2 — Lean tool schemas (no defaults, docstring cap)

- `OpenAICore._openai_tool_definition(tool, max_doc_chars: int = 0)`:
  remove the `schema.setdefault("default", param.default)` line; truncate
  the description to `max_doc_chars` chars when `> 0`.
- `Agent.__init__`: pass `tool_doc_chars` from `[agent]` (int, default 0,
  invalid → 0) when building `self.tool_definitions`.
- `config.toml`: commented `# tool_doc_chars = 300` in `[agent]`.
- **Unit tests**:
  - no definition of any default tool contains a `"default"` key anymore;
    `properties`, `required`, `name` and `type` are preserved for a tool
    with defaults (e.g. `generate_image`) and optional params
    (e.g. `agent_activity`).
  - `max_doc_chars=0` keeps the full docstring; `max_doc_chars=300`
    truncates every description to ≤300 chars; definitions stay valid
    (`function.name` non-empty, `parameters.type == "object"`).
  - payload budget: with `tool_doc_chars=300` the total JSON size of all
    44 definitions is < 14_000 chars (guards regressions; today ~21.9k).
- **Commit**: `feat(tokens): lean tool schemas (no defaults, doc cap)`

### Task 3 — Configurable tool groups

- Add `_select_tools(tools, groups) -> list` next to `_default_tools()`
  (topic lookup via `_TOOL_TOPICS`; `"all"`, empty or None → all tools;
  unknown names logged once and ignored).
- `Agent._build_tools()`: wrap the result of `_default_tools(self)` with
  `_select_tools(..., getattr(config.agent, "tool_groups", "all"))`
  (string or list accepted, like `_split_list`). MCP injection afterwards
  is unchanged.
- `config.toml`: commented example `# tool_groups = "model, delegation,
  memory, files, web, documents, images, system, mcp"` in `[agent]`.
- **Unit tests**:
  - default/`"all"`/empty → exactly today's 44 names.
  - a groups list keeps every tool of the listed topics and drops the
    others (e.g. without `"finance"`: no `fetch_financial_data`,
    `save_financial_chart`; with `"files"`: `read_text_file` present).
  - unknown group name is ignored (no crash, remaining groups honored);
    list form and comma-string form are equivalent.
  - existing `tests/test_live_ai.py` tool-presence tests keep passing
    (they use the default path).
- **Commit**: `feat(tokens): configurable tool groups via [agent] tool_groups`

### Task 4 — Prune old tool rounds at high context usage

- `Agent.__init__`: resolve `self._context_trim_ratio` (float, default 0.0,
  invalid or ≥1 → 0.0 = off) and `self._context_keep_messages` (int,
  default 40, invalid/`< 4` → 40) from `[brain]`.
- Add `Agent._prune_history() -> int` (returns dropped count): no-op when
  ratio is 0; while usage > ratio and there are droppable messages beyond
  the keep-window, drop the oldest complete tool round (assistant with
  `tool_calls` + its `tool` messages), else drop the oldest plain
  user/assistant pair; never touch index 0 or the keep-window; after
  pruning append the one-line note
  `[history pruned: N messages dropped]` as a system message (only when
  N > 0). Called from `act()` right after `_add_user_prompt`.
- `config.toml`: commented `# context_trim_ratio = 0.7` and
  `# context_keep_messages = 40` in `[brain]`.
- **Unit tests**:
  - ratio 0 → history untouched (today's behavior).
  - synthetic history (system + 20 old tool rounds + recent messages) with
    a small fake context window: pruning removes only complete groups (no
    orphan `tool` message without its `tool_calls` assistant, and vice
    versa), keeps the system prompt and the last `context_keep_messages`
    messages, appends the note once.
  - pruning stops when usage is back under the ratio; a history already
    under the ratio is untouched; a history smaller than the keep-window
    is untouched.
  - `act()` integration: fake round loop calls `_prune_history` once per
    turn (monkeypatch a counter).
- **Commit**: `feat(tokens): prune old tool rounds at high context usage`

### Task 5 — Docs, config examples, e2e check

- `README.md`: add a short "Reducing token consumption" section listing the
  four new keys, the measured baseline (44 tools ≈ 5.5k schema tokens per
  request) and recommended values (`max_tool_result_chars = 4000`,
  `tool_doc_chars = 300`, curated `tool_groups`, `context_trim_ratio =
  0.7`); cross-link from the multi-agent section (pooled agents inherit the
  settings).
- `config.toml`: verify all four commented examples are in place with a
  one-line rationale each.
- **Unit tests**:
  - e2e-style offline turn (`_BareAgent` + fake stream/tools) with all four
    features enabled: the history after the turn contains the truncated
    tool result, the pruned note when triggered, and the turn completes
    with the final assistant answer (no role-alternation break —
    `_repair_history` still satisfied).
  - config round-trip: `load_config` on a temp config with the new keys
    yields the parsed values; missing keys keep the defaults.
- **Commit**: `docs(tokens): token-reduction guide + e2e check`

### Task 6 — Runtime tool toggling (`/tools on|off`)

- `Agent` gains a toggle API (mirroring `refresh_mcp_tools`):
  - `set_tool_enabled(name: str, enabled: bool) -> str`: adds/removes
    `name` from a `self._disabled_tools: set[str]`; returns a
    human-readable confirmation listing the enabled tool count.
  - `enabled_tool_definitions() -> list[dict]`: the filtered
    `tool_definitions` (used by `_stream_round` instead of the raw list —
    one-line change in `_stream_round(tools=...)` call sites).
  - `_execute_tool` keeps working for disabled tools (defensive: returns
    `Error: tool '<name>' is disabled` so a stale model reference fails
    loudly instead of silently running).
  - Filtering applies to MCP tools too (they live in the same lists);
    `refresh_mcp_tools` re-injection must skip currently disabled names.
- `session.py`: extend `_cmd_tools` — `/tools` keeps the launcher;
  `/tools on|off <name|topic>` toggles (topic names resolved via
  `_TOOL_TOPICS`/`tool_topics`, toggling every tool of the topic);
  `/tools` with no arg shows the grouped list with `[off]` markers.
  Update `_cmd_help` row for `/tools`.
- Pooled agents are untouched (toggles are main-agent/session-only; workers
  keep their boot-time tool set).
- **Unit tests**:
  - `set_tool_enabled("web_search", False)` removes it from
    `enabled_tool_definitions()` and from the impls lookup used by
    `_execute_tool` (disabled call returns the error string); re-enabling
    restores both; unknown name returns an error, not an exception.
  - topic toggle: disabling topic `finance` removes both finance tools;
    `/tools` list shows `[off]` markers (capture cprint output or factor
    the listing into a testable helper).
  - `_stream_round` uses the filtered definitions (fake stream records the
    `tools` kwarg passed to `_stream_completion`).
  - `refresh_mcp_tools` after disabling an MCP tool keeps it disabled.
- **Commit**: `feat(tokens): runtime tool toggling via /tools on|off`

## 6. Verification checklist (definition of done)

- [ ] `python -m pytest tests/` passes with the new tests from all tasks.
- [ ] With the new keys unset/0, all existing tests pass untouched and
      runtime behavior is byte-identical (full tool set, full docstrings,
      uncapped results, no pruning).
- [ ] With the recommended values: schema payload < 14k chars (~3.5k
      tokens), tool results capped at 4000 chars, long turns pruned before
      hitting the context window.
- [ ] No orphan tool messages / split tool-call groups after pruning
      (servers must never see an incomplete parallel tool-call group).
- [ ] Monitor events, delegation, MCP injection and Ctrl+C cancellation
      behave exactly as before.
- [ ] `/tools on|off` toggles take effect on the next request (disabled
      tool schemas are not sent) and survive `/mcp refresh`; pooled agents
      are unaffected.
- [ ] One commit per task, in order, with the messages above.
