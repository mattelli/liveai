# PLAN_ADD_COMPACTIFICATION.md

## Goal

Add a **conversation-history compaction** capability to the Live Assistant:

1. A new tool `compact_chat` (in `src/tools/memory.py`) that shrinks the live
   chat history **in place** by replacing the oldest messages with a compact
   summary, keeping the most recent turns verbatim.
2. A new slash command `/compact [keep_last]` in `src/session.py` so the user
   can trigger compaction manually.
3. Registration of the tool in the agent tool set (`src/live_ai.py`) and in
   the `/tools` topic map so the model can also call it on its own when the
   context grows long.
4. An optional automatic trigger: the new `[brain] auto_compaction` flag in
   `config.toml` (default `false`). When `true`, compaction is invoked at the
   existing `context_trim_ratio` checkpoint, replacing the oldest-round
   pruning with a summarizing compaction.

Rationale: `clear_chat` destroys context entirely; long sessions eventually
exceed the model context window. Compaction keeps a lossy-but-useful summary
of the earlier conversation while preserving recent turns and the system
prompt / personal-memory seeding, all without changing the messages list
object identity (the session must keep seeing the *same* list, exactly like
`clear_chat` already guarantees).

## Current behavior

- `src/tools/memory.py`:
  - `MemoryTools.clear_chat()` resets the history **in place** to
    `[system, recall_memory(), last user message]`. Everything before is lost.
  - History access goes through `self._history_provider` (a callable returning
    the live messages list); the module-default instance is backed by the
    module global `_history_provider` (monkeypatch-able seam).
- `src/session.py`:
  - `ConversationSession.COMMANDS` = `("/help", "/quit", "/clear", "/voice",
    "/tools", "/info", "/model", "/session", "/mcp", "/monitor")`.
  - `/clear` maps to `_cmd_clear()` which calls `clear_chat()`.
- `src/live_ai.py`:
  - `_default_tools()` lists the tools exposed to the model; memory tools
    (`recall_memory`, `store_memory`, `clear_chat`) are included.
  - `_TOOL_TOPICS["memory"] = ("recall_memory", "store_memory", "clear_chat")`
    drives `/tools` display and `[agent] tool_groups` filtering.
  - Isolated agents get per-agent instances via `MemoryTools.for_owner`;
    `_build_tools` maps module functions to per-agent methods.
- There is **no** way to shrink the history while keeping an editable summary:
  either everything is kept or `/clear` wipes it.

## Expected behavior

- `MemoryTools.compact_chat(keep_last: int = 4) -> str`:
  1. Returns `"Error: no chat session is bound."` when no history is bound.
  2. Returns `"Context is already compact; nothing to do."` when the history
     has fewer than `keep_last + 2` messages (nothing meaningful to compact).
  3. Otherwise:
     - Summarizes all messages **except the last `keep_last`** into a short
       plain-text summary (key facts, decisions, user preferences, open tasks,
       file paths, in-progress work).
     - Preserves the first `system` message of the history (falls back to
       `[brain] system_prompt` config when absent) and the `recall_memory()`
       seeding message, exactly like `clear_chat` does.
     - Replaces the history **in place** (same list object) with:
       `[system, recalled personal data, summary message, last keep_last messages]`
       where the summary message is:
       `{"role": "user", "content": "[Compacted context on <YYYY-mm-dd HH:MM>] Summary of the earlier conversation:\n<summary>"}`.
  4. Returns a human-readable confirmation, e.g.
     `"Chat compacted: 12 messages summarized into 1, 4 kept."`.
  5. Never raises: any exception is returned as `"Error: <e>"` (existing
     memory-tool convention).
- The summary is produced by an **injectable summarizer seam**
  (`MemoryTools.bind_summarizer` / module-level `bind_summarizer`), defaulting
  to a deterministic extractive fallback (first N chars of the concatenated
  texts) so the tool is fully unit-testable without an LLM. Wiring the real
  LLM call is a separate task.
- `compact_chat` is exposed:
  - as an agent tool in `_default_tools()` and topic `"memory"` in
    `_TOOL_TOPICS`;
  - as the `/compact [keep_last]` command in `ConversationSession`
    (added to `COMMANDS`, dispatcher and `/help`), printing the tool result;
  - for isolated agents via the `per_agent` mapping in `_build_tools`.
- Existing behavior of `/clear`, `clear_chat`, recall/store memory is
  unchanged.

## Tasks (one commit per task)

> Rule: implement exactly one task per commit, message format
> `task N: <short description>`. Every commit must keep the full test suite
> green (`pytest -q`).

### Task 1 — `MemoryTools.compact_chat` with fallback summarizer + unit tests
- File(s): `src/tools/memory.py`, `tests/test_tools.py`.
- Add `compact_chat(self, keep_last: int = 4) -> str` to `MemoryTools`,
  plus:
  - `bind_summarizer(provider)` storing an injectable callable
    `summarizer(messages: list[dict]) -> str` (default: internal
    `_fallback_summary` that joins/truncates message contents).
  - Module-level `bind_summarizer(provider)` delegating to the module-default
    instance (same pattern as `bind_history`).
  - Add `"compact_chat"` to `__all__`.
- Implementation rules:
  - mutate the provider list **in place** (`messages[:] = new_history`);
  - reuse the system-prompt / recall seeding logic of `clear_chat`
    (factor a small private helper if it avoids duplication, without
    changing `clear_chat` behavior);
  - guard `keep_last` (minimum 2, default 4).
- Unit tests in `tests/test_tools.py` (new section `# T-compaction`):
  - `test_compact_chat_without_bound_session` → starts with `"Error:"`;
  - `test_compact_chat_already_short` → returns the "already compact" message
    and leaves the history untouched;
  - `test_compact_chat_keeps_recent_turns_and_summary` → with a bound
    summarizer stub returning `"SUMMARY"`, verify the final history is
    `[system, recall_seed, summary_msg, last keep_last msgs]` and that it is
    the **same list object** (`assert messages is same_object`);
  - `test_compact_chat_uses_bound_summarizer` → the bound summarizer receives
    exactly the non-recent messages;
  - `test_compact_chat_reseeds_personal_data` (history file present);
  - `test_module_default_compact_chat` (module-level function delegates).
- Commit: `task 1: MemoryTools.compact_chat with injectable summarizer`.

### Task 2 — Export from `src.tools` and register as agent tool
- File(s): `src/tools/__init__.py`, `src/live_ai.py`, `tests/test_agent_isolation.py`.
- Import + re-export `compact_chat` from `.memory` and add it to `__all__`.
- Add `compact_chat` to `_default_tools()` right after `clear_chat`.
- Add `"compact_chat"` to `_TOOL_TOPICS["memory"]`.
- In `_build_tools`, map `memory_mod.compact_chat` to the per-agent
  `memory.compact_chat` (isolated agents).
- Tests:
  - extend the existing isolation tests so an isolated agent gets its own
    `compact_chat` (no shared module state);
  - a test asserting `_TOOL_TOPICS` contains `compact_chat` under `memory`
    (keeps `/tools` display and `tool_groups` filtering correct).
- Commit: `task 2: expose compact_chat as agent tool`.

### Task 3 — `/compact` session command
- File(s): `src/session.py`, `tests/test_input_reader.py` (or the session
  test file already covering slash commands).
- Add `/compact` to `COMMANDS`; dispatcher branch:
  `if cmd == "/compact": self._cmd_compact(arg); return True`.
- `_cmd_compact(arg)`: parse optional integer `keep_last` (invalid/missing →
  default 4), call `compact_chat(keep_last)`, `cprint(result, "green"|"red")`
  depending on success prefix.
- Add `("/compact [keep_last]", "compact old history into a summary, keeping the last N messages")`
  to the `/help` rows.
- Tests: `/compact` with no arg uses default; `/compact 2` parses the argument;
  `/compact abc` falls back to the default; `/help` lists `/compact`.
- Commit: `task 3: add /compact session command`.

### Task 4 — Wire the real LLM summarizer
- File(s): `src/live_ai.py` (or `Brain` binding site where `bind_history` is
  called), `src/tools/memory.py`.
- When binding the history (`bind_history`), also bind a summarizer that calls
  the same chat model with a fixed summarization system prompt
  ("Summarize this conversation: key facts, decisions, user preferences,
  open tasks, file paths. Be concise.") and the messages to compact.
  The call must be non-interactive, with tool use disabled, and must fall
  back to the built-in extractive summarizer on any error/timeout.
- Config: optional `[brain] compact_keep_messages` (int, default 4) read the
  same way `context_keep_messages` is read in `Brain.__init__`; the session
  command passes it only when the user explicitly provided one. This value is
  also used by the automatic trigger added in Task 5.
- Tests: with a fake client injected, the summarizer produces the fake reply;
  on client exception the fallback summary is used (no exception escapes).
- Commit: `task 4: bind LLM summarizer for compaction`.

### Task 5 — `[brain] auto_compaction` config + automatic trigger
- File(s): `config.toml`, `src/live_ai.py`, `src/tools/memory.py`,
  `tests/test_tools.py` (or the brain/config test file).
- Config: add to the `[brain]` section of `config.toml` (next to
  `context_trim_ratio` / `context_keep_messages`):

  ```toml
  auto_compaction = false           # run compact_chat when context usage exceeds context_trim_ratio (false = legacy pruning only)
  ```

  Default is **false** (the historical pruning behavior is unchanged); set
  `true` to trigger full compaction instead of/in addition to pruning.
- Reading the flag: `Brain.__init__` reads it the same way the neighbouring
  keys are read (`getattr(config.brain, "auto_compaction", False)`),
  storing it as e.g. `self._auto_compaction` (any non-`"true"`/`True` value,
  including missing key, means false).
- Trigger behavior (when `auto_compaction = true`):
  - at the same checkpoint where the existing `context_trim_ratio` pruning
    runs (approximate context usage vs. the model context window), if the
    ratio is exceeded, call the module-default `compact_chat()` (keeping the
    configured `compact_keep_messages` recent turns) instead of the plain
    oldest-round pruning;
  - compaction runs at most once per turn (re-check after compaction; if the
    usage is still above the ratio, fall back to the existing pruning so the
    loop cannot spin);
  - the user is notified once per compaction (e.g. a printer/log line
    "Context compacted: N messages summarized");
  - any failure is swallowed and logged — a failed compaction must never
    break the ongoing turn.
- Tests:
  - config parsing: missing key / `"false"` / `"true"` mapping to the flag;
  - with `auto_compaction = true` and usage above the ratio, compaction is
    invoked once and the history contains the summary message;
  - with `auto_compaction = false` (default), the legacy pruning path runs
    unchanged (regression);
  - compaction failure does not raise and falls back to pruning.
- Commit: `task 5: [brain] auto_compaction trigger on context-usage ratio`.

### Task 6 — Documentation & final pass
- File(s): `README`/docs section for commands, `config.toml` comment,
  module docstrings.
- Document `/compact` next to `/clear`; document the `compact_chat` tool in
  the `src/tools/__init__.py` module list (`memory` bullet); document
  `[brain] auto_compaction` (default false) next to `context_trim_ratio`.
- Run the full suite plus the integration tests
  (`pytest -q tests/test_integration.py`), verify `/compact` manually in a
  live session and, with `auto_compaction = true`, verify the automatic
  trigger on a long session.
- Commit: `task 6: document /compact, compact_chat and auto_compaction`.

## Verification (unit tests overview)

All tests live in `tests/test_tools.py` (tasks 1/4) and the session/isolation
test files (tasks 2/3). Minimal acceptance set:

| Test | Asserts |
|---|---|
| `test_compact_chat_without_bound_session` | result starts with `"Error:"` |
| `test_compact_chat_already_short` | no-op message; history unchanged |
| `test_compact_chat_keeps_recent_turns_and_summary` | in-place list; `[system, seed, summary, last N]` |
| `test_compact_chat_uses_bound_summarizer` | summarizer gets only the old messages |
| `test_compact_chat_reseeds_personal_data` | recall seed preserved after compaction |
| `test_module_default_compact_chat` | module-level delegate works via `_history_provider` seam |
| isolation tests | isolated agent uses its own `compact_chat` instance |
| session command tests | `/compact` arg parsing, default, `/help` entry |

Run everything with `pytest -q`; every commit must pass.

## Out of scope

- Persistent storage of compaction summaries (only in-history).
- Automatic compaction **outside** the existing `context_trim_ratio`
  checkpoint (the flag added in Task 5 hooks into that single, already tested
  checkpoint; no new usage-estimation machinery).
