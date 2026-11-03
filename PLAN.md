# PLAN.md — live_ai.py improvements inspired by smolagents v1.26.0

Reference: `_tmp_smol/x/smolagents/agents.py` (v1.26.0, committed for comparison).

## Goals

Improve `src/live_ai.py` by adopting seven patterns from smolagents v1.26.0:

1. **Forced final answer** when `MAX_ACT_ROUNDS` is exhausted (smolagents:
   `_handle_max_steps_reached` / `provide_final_answer`).
2. **Tool argument validation** against the generated JSON schema before
   invoking (smolagents: `validate_tool_arguments` in `execute_tool_call`).
3. **Parallel tool calls** when the model requests several at once
   (smolagents: `ThreadPoolExecutor` + `copy_context()` in `process_tool_calls`).
4. **Real token usage tracking** per turn (smolagents: `token_usage` on every
   step) instead of the ~4 chars/token estimate only.
5. **Lightweight step records** for introspection (smolagents: structured
   memory steps, `replay()`).
6. **Public `interrupt()` API** (smolagents: `interrupt()` setting
   `interrupt_switch`).
7. **Polish**: typed agent errors, early streaming-capability warning, prompt
   templates as overridable data, and a structured return value from `act()`.

## Current behavior

- `Agent.act()` (L2061) loops `MAX_ACT_ROUNDS` rounds; when the budget is
  exhausted the turn simply ends — no final answer is produced if the model
  was mid-work, and the last partial text (possibly none) is all the user gets.
- `_execute_tool()` (L2034) JSON-decodes arguments and calls the tool; a
  missing/misspelled parameter surfaces as a raw `TypeError` string. The JSON
  schemas built in `_openai_tool_definition()` (L1145) are used only for the
  API request, never for validation.
- Tool calls in one round are executed strictly sequentially in `act()`.
- `_handle_round_end()` (L1983) reports context usage from `_context_usage()`
  (char-count heuristic); stream `usage` data is discarded.
- `_messages` (flat list) is the only record of what happened; `AgentMonitor`
  records events only for isolated sub-agents.
- Cancellation is driven by the private `_cancel_requested` flag; there is no
  public `interrupt()` and `_execute_tool` does not re-check it.
- `_execute_tool` catches bare `Exception`; no error hierarchy. `act()`
  returns `None`.

## Expected behavior

- When the round budget is exhausted, one final tool-free completion is
  requested and its text is appended as the assistant message; the turn always
  ends with a coherent answer (or the cancellation placeholder).
- Before invoking a tool, arguments are validated against the tool's JSON
  schema; on failure the model receives a corrective error message listing the
  expected parameters, so it can self-correct next round.
- Multiple tool calls in one round run concurrently (bounded pool); results
  are appended to history in the original call order; thread-locals
  (`_current_agent_name`, monitor) keep working via `copy_context()`.
- Each round captures the stream's `usage` chunk (with
  `stream_options.include_usage`); per-turn prompt/completion token totals are
  shown in the round-end stats line and used to correct the context estimate.
- Every round appends a step record (round index, tool names, duration, error,
  tokens) to an in-memory list, queryable for replay/introspection.
- A public `Agent.interrupt()` sets the cancel flag; `_execute_tool` refuses
  to start a tool after cancellation was requested.
- Tool failures raise/return through a small typed error hierarchy
  (`AgentToolError`, `AgentToolArgumentError`, `AgentToolExecutionError`);
  `OpenAICore` warns at init when the selected model/endpoint does not support
  streaming; the summarizer/system prompt strings live in a `prompts` data
  structure overridable via config; `act()` returns an `ActResult` dataclass
  (text, rounds, tool calls, duration, token usage).

## Tasks (one commit per task)

### Task 1 — Public `interrupt()` API (point 6)
- Add `Agent.interrupt()`: sets `_cancel_requested = True` (delegates to the
  existing `cancel_generation()` flag path, without closing the stream).
- In `_execute_tool`, return an error string ("cancelled") instead of
  executing when `_cancel_requested` is set.
- Files: `src/live_ai.py`.
- Tests (`tests/test_live_ai.py`, `_BareAgent` pattern):
  - `test_interrupt_sets_flag` — `interrupt()` flips `cancel_requested`.
  - `test_execute_tool_skipped_after_interrupt` — `_execute_tool` returns the
    cancellation message and never calls the tool callable.
- Commit: `feat(agent): public interrupt() and cancellation check in _execute_tool`

### Task 2 — Forced final answer on round-budget exhaustion (point 1)
- In `Agent.act()`, after the loop ends without a final text round, run one
  extra `_stream_completion(None)` (no tools), append its text as the
  assistant message, and log a warning ("max rounds reached").
- Skip the extra call when the turn was cancelled.
- Files: `src/live_ai.py`.
- Tests:
  - `test_act_forces_final_answer_after_max_rounds` — fake stream returns
    tool calls every round; assert a final assistant message with text exists
    in `_messages` after `act()`.
  - `test_act_no_final_call_when_cancelled` — cancelled turn does not make
    the extra request.
- Commit: `feat(agent): force a final answer when MAX_ACT_ROUNDS is exhausted`

### Task 3 — Tool argument validation against the JSON schema (point 2)
- Extend `_openai_tool_definition` bookkeeping so `_tool_impls` also keeps the
  schema per tool (or add `_tool_schemas`).
- Add `_validate_tool_args(name, args) -> str | None`: check required keys,
  unknown keys, and primitive types; return a corrective message (listing
  expected parameters) or None when valid.
- Call it in `_execute_tool` before invoking; on failure return the message
  without executing.
- Files: `src/live_ai.py`.
- Tests:
  - `test_validate_tool_args_missing_required`
  - `test_validate_tool_args_unknown_key`
  - `test_validate_tool_args_wrong_type`
  - `test_execute_tool_returns_validation_error_without_calling_tool`
- Commit: `feat(agent): validate tool arguments against JSON schema before execution`

### Task 4 — Parallel tool calls (point 3)
- In `act()`, when `len(tool_calls) > 1`, execute them via
  `ThreadPoolExecutor(max_workers=min(len(calls), 4))`, submitting
  `contextvars.copy_context().run(self._execute_tool, call)`.
- Collect results and append `role: "tool"` messages in the original
  (enumerated) order; keep the sequential path for a single call.
- Config: `[agent] max_parallel_tools` (0/1 = sequential, default 1 —
  opt-in to preserve current behavior).
- Files: `src/live_ai.py`.
- Tests:
  - `test_parallel_tool_calls_preserve_order` — three slow fake tools finish
    out of order; history order matches call order.
  - `test_parallel_tool_calls_sequential_by_default`
  - `test_parallel_tool_calls_propagate_context` — monitor thread-local
    (`_current_agent_name`) visible inside tools.
- Commit: `feat(agent): optional parallel execution of multi-tool rounds`

### Task 5 — Real token usage tracking (point 4)
- In `_stream_completion`, pass `stream_options={"include_usage": True}`
  (guarded: omit for servers that reject it — retry once without on 4xx).
- In `_stream_round`, read the final chunk's `usage` (prompt/completion
  tokens) and return it alongside (text, calls).
- Accumulate per-turn totals in `act()`; show them in `_handle_round_end`
  stats line (`ctx … · in/out tokens`); feed the observed prompt-token count
  back into `_context_usage` as a correction when available.
- Files: `src/live_ai.py`.
- Tests:
  - `test_stream_round_captures_usage` — fake stream yields a usage chunk;
    returned usage matches.
  - `test_stream_completion_retries_without_stream_options` — server error on
    the option falls back gracefully.
  - `test_round_end_stats_include_tokens` — stats line contains token counts.
- Commit: `feat(brain): capture real token usage from streaming responses`

### Task 6 — Step records for replay/introspection (point 5)
- Add a `_steps: list[dict]` per agent; append one record per round:
  `{round, tools, durations_ms, errors, prompt_tokens, completion_tokens}`.
- Expose `Agent.replay() -> list[dict]` (read-only copy) and wire the main
  agent's records into `AgentMonitor` when a monitor is bound.
- Files: `src/live_ai.py`.
- Tests:
  - `test_act_records_steps` — after a two-round `act()` (tool round + final),
    `_steps` has two records with correct fields.
  - `test_replay_returns_copy` — mutating the returned list does not affect
    the agent.
- Commit: `feat(agent): per-round step records and replay()`

### Task 7 — Typed agent errors (point 7a)
- Add module-level exceptions: `AgentToolError(Exception)`,
  `AgentToolArgumentError(AgentToolError)`, `AgentToolExecutionError(AgentToolError)`.
- `_execute_tool` raises/returns through these types (argument problems →
  `AgentToolArgumentError`; tool failure → `AgentToolExecutionError`); the
  string returned to the model keeps the same text as today.
- Files: `src/live_ai.py`.
- Tests:
  - `test_execute_tool_error_types` — malformed args raise
    `AgentToolArgumentError`; a raising tool raises
    `AgentToolExecutionError` with the original message.
- Commit: `refactor(agent): typed error hierarchy for tool execution`

### Task 8 — Early streaming-capability warning (point 7b)
- In `OpenAICore.__init__`/`select_model`, when the resolved model metadata
  lacks streaming support (or the endpoint errors on a probe), log a warning
  once instead of failing mid-turn.
- Files: `src/live_ai.py`.
- Tests:
  - `test_select_model_warns_when_streaming_unsupported` — warning logged,
    selection still succeeds.
- Commit: `feat(core): warn at model selection when streaming is unsupported`

### Task 9 — Prompt templates as overridable data (point 7c)
- Move `_SUMMARIZER_SYSTEM_PROMPT` and the history-prune marker text into a
  module-level `_PROMPTS` dict; allow `[brain] prompts_file` (JSON) to
  override entries, validated like `_validated_style`.
- Files: `src/live_ai.py`.
- Tests:
  - `test_prompts_file_overrides_summarizer` — config file value is used.
  - `test_prompts_defaults_unchanged_without_config`.
- Commit: `refactor(brain): prompt templates as overridable data`

### Task 10 — `ActResult` return value (point 7d)
- Add `@dataclass ActResult`: `text`, `rounds`, `tool_calls`, `duration_s`,
  `prompt_tokens`, `completion_tokens`, `cancelled`.
- `act()` builds and returns it; `queue_agent_result`/monitor consumers updated.
- Files: `src/live_ai.py` (+ callers in the session/scheduler if signatures
  change — keep `act()` returning the value, no breaking change).
- Tests:
  - `test_act_returns_act_result` — fields populated after a fake run.
  - `test_act_result_cancelled_flag`.
- Commit: `feat(agent): return ActResult from act()`

## Verification

- Every task adds unit tests to `tests/test_live_ai.py` following the
  existing `_BareAgent` / fake-stream conventions; no network or model calls.
- Full suite after each task: `python -m pytest tests/test_live_ai.py -q`
  (and `python -m pytest tests -q` before the final commit).
- Manual smoke check after Tasks 2, 4, 5: run the live assistant, issue a
  tool-using prompt, verify final answer, stats line, and parallel calls.
- One commit per task, message template shown per task.


## Tool ports from smolagents (Tasks 11–14)

Ports of smolagents v1.26.0 default tools that have no equivalent in
`src/tools` (see the coverage comparison: `web_search`/`extract_webpage`/
`mcp_*`/`run_command` already cover the rest). Each follows the existing
tool-module conventions: per-topic submodule, re-export in
`src/tools/__init__.py` (`__all__`), `configure()` section, registration in
`Agent._build_tools` via `_default_tools`.

Token note (ref. previous discussion): each new tool adds its JSON schema to
every request of the agents that carry it (~50–100 tokens each). Tasks 13–14
are deliberately scoped to isolated/background agents only, so the main
session's per-request overhead stays unchanged.

### Task 11 — `wikipedia_search` tool (port of `WikipediaSearchTool`)
- New `src/tools/wiki.py`: `wikipedia_search(query, language="en", max_results=3)`
  — search via `w/api.php?action=query&list=search`, then fetch page summaries
  via `rest_v1/page/summary/{title}`; returns title, URL, and summary extract.
  Custom `User-Agent` header (required by Wikimedia API policy, as smolagents'
  `user_agent` attribute does). Uses the lazy `requests` import pattern of
  `src.tools.__init__`.
- Wire: re-export in `src/tools/__init__.py`; add to the web topic in
  `_default_tools` (`src/live_ai.py`).
- Files: `src/tools/wiki.py`, `src/tools/__init__.py`, `src/live_ai.py`.
- Tests (`tests/test_wikipedia_search.py`, mock `requests`):
  - `test_wikipedia_search_returns_summary_and_url`
  - `test_wikipedia_search_handles_missing_page` — empty result, no raise.
  - `test_wikipedia_search_sends_user_agent`
  - `test_wikipedia_search_in_build_tools`
- Commit: `feat(tools): wikipedia_search tool (smolagents WikipediaSearchTool port)`

### Task 12 — `transcribe_audio` tool (port of `SpeechToTextTool`)
- New `src/tools/stt.py`: `transcribe_audio(path)` — local transcription via
  `faster-whisper` (lazy import, optional dependency like the piper stack;
  friendly error when not installed). `configure()` reads `[stt]`
  (model_size, device). Returns recognized text; caps input file size.
- Wire: re-export in `src/tools/__init__.py`; add to `_default_tools`.
- Files: `src/tools/stt.py`, `src/tools/__init__.py`, `src/live_ai.py`.
- Tests (`tests/test_stt.py`, fake transcriber seam):
  - `test_transcribe_audio_returns_text`
  - `test_transcribe_audio_missing_file`
  - `test_transcribe_audio_lazy_import` — import of `src.tools` stays cheap.
  - `test_transcribe_audio_in_build_tools`
- Commit: `feat(tools): transcribe_audio tool (smolagents SpeechToTextTool port)`

### Task 13 — `python_eval` sandboxed tool for isolated agents
(port of `PythonInterpreterTool`)
- New `src/tools/pyeval.py`: `python_eval(code, timeout_s=10)` — runs the
  snippet in a subprocess with a restricted environment (no network env,
  temp cwd, stdout/stderr cap, hard timeout, output includes the exit status).
  Deliberately NOT in the main agent's tool set (`run_command` already covers
  it and the main session is user-supervised); registered only for
  isolated/background agents in `Agent._build_tools` when
  `isolated_tools=True` — the same path that gets fresh memory/vision
  instances.
- Files: `src/tools/pyeval.py`, `src/tools/__init__.py`, `src/live_ai.py`
  (`_build_tools` isolated branch).
- Tests (`tests/test_pyeval.py`):
  - `test_python_eval_returns_stdout`
  - `test_python_eval_timeout_kills_process`
  - `test_python_eval_captures_stderr_and_status`
  - `test_python_eval_isolated_agents_only` — absent from main agent tools,
    present for `isolated_tools=True`.
- Commit: `feat(tools): sandboxed python_eval for isolated agents (smolagents PythonInterpreterTool port)`

### Task 14 — `ask_user` tool for background agents (port of `UserInputTool`)
- New `src/tools/user_input.py`: `ask_user(question, timeout_s=300)` — for
  isolated/background agents: queues the question to the live session through
  the existing `queue_agent_result`/monitor plumbing and blocks on an answer
  queue (timeout returns a graceful "no answer" message). Excluded from the
  main agent (it would block the interactive UI thread).
- Files: `src/tools/user_input.py`, `src/tools/__init__.py`,
  `src/live_ai.py` (`_build_tools` isolated branch + session wiring for the
  answer queue).
- Tests (`tests/test_user_input.py`):
  - `test_ask_user_delivers_question_and_returns_answer`
  - `test_ask_user_timeout_returns_graceful_message`
  - `test_ask_user_not_in_main_agent_tools`
- Commit: `feat(tools): ask_user tool for background agents (smolagents UserInputTool port)`

### Verification (tool ports)
- New test modules follow the existing mocking conventions (no network, no
  model calls); `python -m pytest tests -q` green before each commit.
- Manual smoke checks: Task 11 — ask the assistant "search Wikipedia for X";
  Task 12 — transcribe a sample wav; Task 13 — background agent computes
  something with `python_eval`; Task 14 — background agent asks a question
  that surfaces in the live session.
- Skipped tools (documented decision): `FinalAnswerTool` (act() loop already
  terminates on a tool-free answer; PLAN Task 2 covers exhaustion) and
  Google/Brave API search variants (`web_search` covers the need).
