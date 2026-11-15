# PLAN — Fix `compact_chat` (src/tools/memory.py)

## Goals

Fix all 14 issues found in the review of `MemoryTools.compact_chat` and its
call sites, without changing the public tool API (`compact_chat(keep_last=4)`
returning a human-readable string, in-place mutation of the live history list,
module-default instance + per-agent `MemoryTools` isolation).

## Decisions (confirmed by the user)

1. **Summary role stays `role: "user"`.** The summary is *not* injected as a
   mid-history `system` message (some OpenAI-compatible servers reject those —
   cf. `_prune_history`'s "never a mid-history system message"). It is made
   recognizable instead by its `[Compacted context` prefix and filtered out of
   personal-memory capture (`store_memory`) and the `clear_chat` replay (T4).
2. **`truncate_text` moves to `src/common.py`.** The head+tail truncation helper
   currently at `live_ai._truncate_tool_result` (`live_ai.py:199-222`) is moved
   verbatim to `src/common.py` as the public `truncate_text(text, limit)`
   (T2). `live_ai._truncate_tool_result` stays as a thin delegating wrapper so
   the existing test seam (`live_ai_mod._truncate_tool_result`,
   `tests/test_token_reduction.py:14-55`) and any external callers keep working
   unchanged.

Issue index (from the review):

| # | Issue | Severity |
|---|-------|----------|
| 1 | `messages[-keep:]` can split a tool-call group → orphan `role: "tool"` in the kept tail → server 400 | critical |
| 2 | `old` can end with `assistant(tool_calls)` whose results are in the tail → summarizer request invalid → silent fallback | critical |
| 3 | Auto-compaction is a no-op for isolated agents (module-default history unbound) | moderate |
| 4 | Isolated agents never get an LLM summarizer (`bind_summarizer` not called) | moderate |
| 5 | Not thread-safe vs. parallel tool rounds / session thread | moderate |
| 6 | Summary injected as `role: "user"` → captured by `store_memory`, replayed by `clear_chat` | moderate |
| 7 | Not idempotent: repeated `/compact` re-summarizes the summary | minor |
| 8 | Can grow a tiny history (2 msgs → head + summary) | minor |
| 9 | Recalled personal data duplicated (head re-seed + summary) | minor |
| 10 | Summarizer request carries two system messages (own prompt + history system) | minor |
| 11 | Nested `[CALL]` dump + uncapped recall injection | minor |
| 12 | Misleading log `"Context auto-compacted: Context is already compact..."` | minor |
| 13 | `keep_last` unbounded; `for_owner(agent)` ignores `agent` | cosmetic |
| 14 | The injected summary is never length-capped: a bound summarizer that ignores `max_tokens` (custom provider, overrunning local model) can inject an arbitrarily long `role: "user"` message into the history; the extractive fallback has its own hard-coded 2000-char split with a different marker and no original-length report | moderate |

## Current behavior

`compact_chat(keep_last=4)`:

1. `keep = max(2, int(keep_last))` (fallback 4 on `TypeError`/`ValueError`).
2. Bails out when `len(messages) < keep + 2`.
3. `old = messages[:-keep]`, `recent = list(messages[-keep:])` — plain slices.
4. `summary = (self._summarizer or self._fallback_summary)(old)`; empty result
   falls back to `_fallback_summary(old)`.
5. Rebuilds the list as `_reseeded_head(messages)` (`[system, recall_memory()]`)
   + one `role: "user"` message `"[Compacted context on <stamp>] Summary of the
   earlier conversation:\n<summary>"` + `recent`, then `messages[:] = new_history`.
6. Returns `"Chat compacted: {len(old)} messages summarized into 1, {len(recent)} kept."`

Length handling today: the extractive fallback truncates its own join at 2000
chars (private `head[:1000] + "\n[...]" + tail[-1000:]` split, no original-length
report); a bound summarizer is only bounded by the `max_tokens=500` it was
called with, and whatever it returns is injected verbatim — nothing caps the
summary at injection time (issue #14).

Call sites:

- `Agent._build_tools` (`src/live_ai.py:1972`) — isolated agents bind
  `memory.bind_history` only; the main agent uses the module-default instance
  bound in `Brain.__init__` (`live_ai.py:763-766`, which also binds
  `_summarize_messages`).
- `Agent._auto_compact` (`live_ai.py:1510`) — always calls the **module-default**
  `memory_tools.compact_chat(keep)`; maps `"Error:"` results to `""`.
- `Agent._prune_history` (`live_ai.py:1454`) — calls `_auto_compact()` when
  `[brain] auto_compaction = true` and usage > `context_trim_ratio`, then logs
  `"Context auto-compacted: %s"`.
- `/compact [keep_last]` (`src/session.py:298`) — module-default tool.
- `_summarize_messages` (`live_ai.py:1620`) — prepends `_PROMPTS["summarizer_system"]`
  to the passed messages and returns `""` on any failure.

## Expected behavior

1. **Group-safe boundary.** The kept tail never starts with a `role == "tool"`
   message, and the summarized range never ends with an `assistant` message
   carrying `tool_calls`. The boundary is walked back (keeping *more* messages)
   until it sits on a group boundary; if that leaves nothing to summarize, the
   tool reports "already compact" and leaves the history untouched.
2. **Head-aware range.** The re-seeded head (`system` + recalled personal data)
   is *excluded* from the summarized range, so the personal data is never
   duplicated into the summary and the system prompt is never re-summarized.
3. **Valid summarizer request.** The request sent to the chat model contains
   exactly one system message (the summarizer prompt) and starts with a
   non-`tool` message.
4. **Marked summary.** The summary stays a `role: "user"` message (mid-history
   `system` messages are rejected by some servers and the codebase deliberately
   avoids them) but is recognizable by its `[Compacted context` prefix, and is
   excluded from personal-memory capture (`store_memory`) and from the
   `clear_chat` replay.
5. **Idempotent and non-growing.** A compaction only happens when the
   replacement is strictly smaller in both message count and characters;
   otherwise "already compact". Repeated `/compact` with no new messages is a
   no-op.
6. **Works for every agent.** Isolated agents get their own summarizer binding
   and auto-compaction uses *their* `MemoryTools` instance.
7. **Concurrency-safe.** History-mutating tools (`clear_chat`, `compact_chat`)
   never run in parallel with other tool calls of the same round, and the
   module-default instance is guarded by a lock (session thread vs. agent turn).
8. **Honest reporting.** The result string says when the extractive fallback was
   used; `_auto_compact` maps "nothing to do" to `""` so no bogus
   "Context auto-compacted" log line is emitted.
9. Public API, return-string shape (`"Chat compacted: ..."`, `"Error: ..."`,
   `"Context is already compact; nothing to do."`) and in-place list identity
   are preserved.
10. **Bounded summary.** Whatever the summarizer returns (LLM or extractive) is
    capped at injection time with the shared `truncate_text` helper and
    `[brain] max_tool_result_chars` (0 = uncapped, the historical behavior), so
    a misbehaving summarizer can never bloat the history. The extractive
    fallback uses the same helper and marker instead of its own private 2000-char
    split.

## Tasks (one commit per task)

Every task ships its own unit tests in the same commit and must leave the full
suite green (`python -m pytest tests -q`). I will ask before running any
process (pytest, git), per AGENT.md.

| Task | Scope | Issues | Commit subject |
|------|-------|--------|----------------|
| T1 | Group-aligned compaction boundary | #1, #2 | `fix(memory): align compact_chat boundary to tool-call groups` |
| T2 | Head-aware range + shared `truncate_text` | #9, #11 | `refactor(common): share the head+tail truncation helper with the memory tools` |
| T2b | Cap the injected summary | #14 | `fix(memory): cap the injected compaction summary` |
| T3 | Summarizer request hygiene | #10 | `fix(live_ai): send a single system prompt to the compaction summarizer` |
| T4 | Mark the summary, exclude it from capture | #6 | `fix(memory): keep compaction summaries out of personal-memory capture` |
| T5 | Idempotence, growth guard, `keep_last` clamp | #7, #8, #13 | `fix(memory): make compact_chat idempotent and non-growing` |
| T6 | Per-agent summarizer, isolated auto-compaction, honest reporting | #3, #4, #12 | `fix(live_ai): per-agent compaction summarizer and honest auto-compaction` |
| T7 | Concurrency safety | #5 | `fix: serialize history-mutating tools and lock the memory instance` |
| T8 | Docs + full-suite verification | — | `docs: document the compact_chat invariants and guards` |

---

### T1 — Group-aligned compaction boundary (issues #1, #2)

**Files:** `src/tools/memory.py`, `tests/test_tools.py`

**Changes**

- Add a pure module-level helper (no `live_ai` import — that would be circular):

  ```python
  def _group_aligned_start(messages: list[dict], keep: int) -> int:
      """First index of the kept tail: never splits a tool-call group.

      Starts from ``len(messages) - keep`` and walks *backwards* (keeping more
      messages) while the tail would begin with a ``role == "tool"`` message
      (whose ``assistant``/``tool_calls`` parent is in the summarized range).
      Because tool results are contiguous, a tail that does not start with a
      ``tool`` message also guarantees the summarized range does not end with a
      dangling ``assistant(tool_calls)``. Returns ``len(messages)`` when nothing
      can be summarized.
      """
  ```

- `compact_chat` uses `start = _group_aligned_start(messages, keep)` instead of
  `len(messages) - keep`; `old = messages[head_len:start]`,
  `recent = list(messages[start:])`.
- Bail out with `"Context is already compact; nothing to do."` when
  `start >= len(messages)` or `not old`.
- Reuse the same helper name/idea as `Agent._next_prunable_group_end`
  (`live_ai.py:1421`) and cross-reference it in both docstrings so the two
  invariants stay in sync.

**Unit tests** (`tests/test_tools.py`, new)

- `test_compact_chat_never_orphans_tool_result`: history ending in
  `assistant(tool_calls)+tool` groups with `keep_last=4` such that the naive
  slice would start on a `tool` message → assert the kept tail's first message
  is not `role == "tool"`, that every kept `tool` message has its
  `assistant(tool_calls)` parent kept, and that the summarized range does not
  end with an `assistant` carrying `tool_calls` (assert on the list the
  summarizer received).
- `test_compact_chat_no_op_when_only_tool_tail_prunable`: a history whose only
  droppable content is a single tool group that cannot be split → returns
  "already compact" and leaves `messages` untouched.
- `test_compact_chat_summarizer_receives_group_complete_range`: assert the
  captured `old` starts at a non-`tool` message and every `tool_calls` id in it
  has its matching `tool` message.

**Commit:** `fix(memory): align compact_chat boundary to tool-call groups`

---

### T2 — Head-aware summarized range + shared `truncate_text` (issues #9, #11)

**Files:** `src/common.py`, `src/live_ai.py`, `src/tools/memory.py`,
`tests/test_common.py`, `tests/test_tools.py`

**Changes**

- **Move the truncation helper to `src/common.py`** (decision 2): add
  `truncate_text(text: str, limit: int) -> str` with the exact body and
  docstring of today's `live_ai._truncate_tool_result` (`live_ai.py:199-222`):
  `limit <= 0` or short-enough text returned unchanged, otherwise
  `head = limit * 3 // 4` + `"\n… [truncated, {len(text)} chars total] …\n"` +
  `tail = limit // 4`, head `rstrip()`-ed and tail `lstrip()`-ed. Export it in
  `src/common.py`'s `__all__` when that list exists.
- `src/live_ai.py`: `_truncate_tool_result` becomes a thin delegating wrapper
  (`return truncate_text(text, limit)`) so the historical private name and the
  test seam (`tests/test_token_reduction.py:14-55`) keep working; `_run_tool_calls`
  may call either.
- Add `_read_memory()` (undecorated): returns the memory file text or
  `"No personal data found!"`. `recall_memory()` (still `@dump_func_call`, tool
  API unchanged) delegates to it.
- Add `_head_len(messages) -> int`: `2` when `messages[1]` is the recall seed
  (`role == "user"` and `content == self._read_memory()`), else `1` when
  `messages[0]` is the system message, else `0`.
- `_reseeded_head` uses `_read_memory()` (no nested `[CALL]` block, issue #11)
  and caps the injected text with
  `truncate_text(text, self._config("brain", "max_tool_result_chars", 0) or 0)`.
- `compact_chat` summarizes `messages[head_len:start]` (T1), so the recall seed
  and the system prompt are never summarized.

**Unit tests**

- `test_truncate_text_*` (in `tests/test_common.py`): the moved helper keeps the
  documented semantics — unchanged under the limit, unchanged when disabled
  (`0`/negative), head+tail kept, 75/25 split, original length in the marker,
  clean cuts at the marker. The existing `tests/test_token_reduction.py` cases
  stay as-is (they exercise the `live_ai` wrapper) plus one assertion that the
  wrapper delegates to `common.truncate_text`.
- `test_compact_chat_does_not_duplicate_personal_data`: history seeded with
  `[system, recall_seed, ...]` and a memory file → after compaction the recall
  text appears **once** (head) and not inside the summary; assert the captured
  `old` contains no message equal to the recall text.
- `test_compact_chat_excludes_system_prompt_from_summary`: captured `old` has no
  `role == "system"` message.
- `test_compact_chat_caps_reseeded_recall`: a `memory.log` far larger than
  `max_tool_result_chars` → the re-seeded head message is truncated and carries
  the `[truncated, N chars total]` marker.
- `test_recall_memory_not_dumped_when_called_internally`: monkeypatch the
  `[CALL]` printer (`src.common._print_call_block`) and assert `compact_chat`
  triggers exactly one block (its own), not one for `recall_memory`.

**Commit:** `refactor(common): share the head+tail truncation helper with the memory tools`

---

### T2b — Cap the injected summary (issue #14)

**Files:** `src/tools/memory.py`, `tests/test_tools.py`

**Changes**

- `compact_chat`: after `summary = (summarizer(old) or "").strip()` and the
  empty-result fallback, cap the text before it is embedded in the
  `[Compacted context ...]` message:
  `summary = truncate_text(summary, self._config("brain", "max_tool_result_chars", 0) or 0)`
  (`0`/unset = uncapped, i.e. today's behavior). The cap applies to the LLM
  summary and to the extractive fallback alike.
- `_fallback_summary`: replace the private `limit = 2000` head/tail split
  (`joined[:limit // 2] + "\n[...]" + joined[-limit // 2:]`) with the shared
  helper — `truncate_text(joined, 2000)` — so there is one truncation format and
  the marker reports the original length. Keep the `"(no content)"` placeholder
  for an empty join.
- Document in the docstring that the summary is bounded by
  `[brain] max_tool_result_chars` and why (a bound summarizer may ignore
  `max_tokens`; an unbounded `role: "user"` message would defeat the compaction).

**Unit tests**

- `test_compact_chat_caps_oversized_summary`: summarizer returning a 50k-char
  string with `max_tool_result_chars = 4000` → the injected message contains
  `[truncated, 50000 chars total]` and the summary body is ≤ 4000 chars.
- `test_compact_chat_summary_uncapped_when_limit_zero`: same summarizer with
  `max_tool_result_chars = 0` → the full summary is injected (historical
  behavior preserved).
- `test_fallback_summary_uses_shared_marker`: a long extractive join carries the
  `[truncated, N chars total]` marker (replacing the old `[...]` marker) and the
  `"(no content)"` placeholder still appears for an empty history slice.
- `test_compact_chat_capped_summary_still_counted`: the result string still
  reports the pre-cap summarized message count.

**Commit:** `fix(memory): cap the injected compaction summary`

---

### T3 — Summarizer request hygiene (issue #10)

**Files:** `src/live_ai.py`, `tests/test_live_ai.py`

**Changes**

- `Brain._summarize_messages`: drop a leading `role == "system"` message and any
  leading `role == "tool"` messages from the received list before prepending
  `_SUMMARIZER_SYSTEM_PROMPT`, so the request carries exactly one system message
  and starts with a user/assistant turn. Return `""` unchanged on failure.

**Unit tests**

- `test_summarize_messages_single_system_prompt`: fake client capturing
  `kwargs["messages"]` → exactly one `system` message, first message is the
  summarizer prompt, no leading `tool`.
- `test_summarize_messages_keeps_conversation_body`: the user/assistant/tool
  messages of the input are forwarded in order.
- `test_summarize_messages_returns_empty_on_failure`: client raising → `""`.

**Commit:** `fix(live_ai): send a single system prompt to the compaction summarizer`

---

### T4 — Mark the summary and exclude it from capture (issue #6)

**Files:** `src/tools/memory.py`, `tests/test_tools.py`

**Changes**

- Keep `role: "user"` for the summary (decision 1; documented reason:
  mid-history `system` messages are rejected by some OpenAI-compatible servers,
  cf. `_prune_history`'s "never a mid-history system message").
- Add `_is_summary(content)` (`str(content).startswith(self._SUMMARY_PREFIX)`)
  and `_selectable_user_messages(messages)`: user messages that are neither the
  recall seed (`_head_len`) nor a compaction summary.
- `_shaped_user_messages` (used by `store_memory`) and `clear_chat`'s
  `user_msgs[-1]` replay both use `_selectable_user_messages`.
- Extend the summarizer system prompt (`_PROMPTS["summarizer_system"]`) with one
  sentence stating the summary is system-generated context, not a user
  instruction (prompt-injection hardening).

**Unit tests**

- `test_store_memory_ignores_compaction_summary`: history whose last user
  message is a `[Compacted context ...]` summary and an earlier real user
  message → `store_memory()` stores the real message, never the summary.
- `test_store_memory_ignores_recall_seed`: freshly cleared chat
  (`[system, "No personal data found!"]`) → `store_memory()` returns
  `"Error: no user message found to store."` (today it persists the seed).
- `test_clear_chat_does_not_replay_summary`: after a compaction, `clear_chat`
  keeps the last *real* user message, not the summary.
- `test_store_memory_explicit_content_still_wins`: explicit `content` is stored
  verbatim regardless of the filters.

**Commit:** `fix(memory): keep compaction summaries out of personal-memory capture`

---

### T5 — Idempotence, growth guard, `keep_last` clamp (issues #7, #8, #13)

**Files:** `src/tools/memory.py`, `tests/test_tools.py`

**Changes**

- Build `new_history` first, then apply both guards; return
  `"Context is already compact; nothing to do."` (history untouched) when:
  - `old` consists solely of a previous summary message (`_is_summary`), or
  - `len(new_history) >= len(messages)` (message-count guard), or
  - `_history_chars(new_history) >= _history_chars(messages)` (char guard),
    with `_history_chars` = `sum(len(str(m.get("content") or "")) for m in ...)`.
- Clamp `keep_last` to `min(max(2, keep), max(2, len(messages) - 1))` and report
  the *effective* value in the result string.
- `MemoryTools.for_owner(agent)`: keep returning a fresh base `MemoryTools()`
  (never `type(self)`, which would re-share the module global through
  `_ModuleDefaultMemoryTools`) and document why `agent` is unused.

**Unit tests**

- `test_compact_chat_is_idempotent`: compact twice with no new messages → second
  call returns "already compact", history byte-identical.
- `test_compact_chat_does_not_grow_tiny_history`: `len(messages) == keep + 2` with
  a summarizer returning a long string → "already compact", history untouched.
- `test_compact_chat_skips_when_summary_not_smaller`: summarizer returning a
  string longer than the summarized range → "already compact".
- `test_compact_chat_clamps_huge_keep_last`: `keep_last=10_000` on a 10-message
  history → no-op / effective clamp, no exception.
- `test_for_owner_returns_isolated_base_instance`: `_default.for_owner(None)` is
  a `MemoryTools` whose `_history_provider` is not the module global.

**Commit:** `fix(memory): make compact_chat idempotent and non-growing`

---

### T6 — Per-agent summarizer, isolated auto-compaction, honest reporting (issues #3, #4, #12)

**Files:** `src/live_ai.py`, `src/tools/memory.py`, `tests/test_token_reduction.py`,
`tests/test_agent_isolation.py`

**Changes**

- `Agent._build_tools` (isolated branch): also
  `memory.bind_summarizer(self._summarize_messages)` and keep the instance on
  `self._memory_tools = memory` for auto-compaction.
- `Agent._auto_compact`: call the agent's own instance when present
  (`self._memory_tools.compact_chat(keep)`), else the module-default
  `memory_tools.compact_chat(keep)` (main agent, unchanged); map both `"Error:"`
  and `"Context is already compact"` results to `""` (fixes the misleading log,
  issue #12).
- `compact_chat` result string mentions the fallback when the bound summarizer
  returned empty, e.g.
  `"Chat compacted: 12 messages summarized into 1, 4 kept (extractive fallback)."`
  (issue #2 visibility). `"Chat compacted: ..."` prefix preserved.

**Unit tests**

- `test_isolated_agent_binds_its_own_summarizer`: isolated agent's
  `compact_chat` uses `_summarize_messages` (monkeypatched to a marker).
- `test_auto_compact_uses_agent_instance_for_isolated_agent`: monkeypatch the
  per-agent instance → `_auto_compact()` calls it, module-default untouched.
- `test_auto_compact_maps_nothing_to_do_to_empty`: result
  `"Context is already compact; nothing to do."` → `""` and no
  `"Context auto-compacted"` log record (caplog).
- `test_compact_chat_reports_extractive_fallback`: summarizer returning `""` →
  result contains `"extractive fallback"` and the summary text is the extractive
  one.

**Commit:** `fix(live_ai): per-agent compaction summarizer and honest auto-compaction`

---

### T7 — Concurrency safety (issue #5)

**Files:** `src/tools/memory.py`, `src/live_ai.py`, `tests/test_token_reduction.py`,
`tests/test_tools.py`

**Changes**

- `MemoryTools.__init__`: `self._lock = threading.RLock()`; `clear_chat` and
  `compact_chat` hold it around read + mutate (module-default instance is shared
  by the session thread and the agent turn).
- `Agent._execute_tool_calls`: force the sequential path when the round contains
  a history-mutating tool (`clear_chat`, `compact_chat`) even with
  `max_parallel_tools > 1`, so a compaction can never interleave with
  `_run_tool_calls` appending `tool` results. Document the rule in the docstring
  and in the `[agent] max_parallel_tools` comment.

**Unit tests**

- `test_history_mutating_tools_run_sequentially`: round with
  `[compact_chat, some_other_tool]` and `max_parallel_tools=4` → the executor
  takes the sequential branch (assert via a spy on the pool / on call order).
- `test_compact_chat_is_reentrant_under_lock`: two threads calling
  `compact_chat` on the same instance leave a valid history (no lost messages,
  no orphan `tool` messages).

**Commit:** `fix: serialize history-mutating tools and lock the memory instance`

---

### T8 — Documentation and full-suite verification

**Files:** `README.md`, `config.toml`, `config-code.toml`, `PLAN-compact-chat.md`

**Changes**

- README `auto_compaction` bullet (`README.md:706-714`): document the
  group-aligned boundary, head exclusion, idempotence/growth guards, the
  `(extractive fallback)` marker, per-agent summarizer, the summary being a
  `[Compacted context ...]` user message excluded from personal-memory capture,
  the summary/recall being capped by `max_tool_result_chars` (shared
  `truncate_text` helper, one marker format), and the sequential-execution rule
  for history-mutating tools.
- `config.toml` / `config-code.toml`: extend the `auto_compaction` /
  `compact_keep_messages` comments accordingly.
- Tick off this plan's tasks (status table) and record the final test run.

**Verification**

- `python -m pytest tests -q` — full suite green.
- Focused: `python -m pytest tests/test_tools.py tests/test_token_reduction.py
  tests/test_agent_isolation.py tests/test_live_ai.py tests/test_session.py -q`.
- Manual smoke: `/compact` twice in a session with a tool-heavy history, then
  `/info` to confirm the context shrank and the next turn still succeeds.

**Commit:** `docs: document the compact_chat invariants and guards`

---

## Existing tests that must be updated (behavior changes)

These assertions encode the old, head-inclusive counting and the old
`keep_last`/slice semantics; they are updated inside the task that changes the
behavior (T1/T2/T5), keeping the intent:

- `tests/test_tools.py::test_compact_chat_keeps_recent_turns_and_summary`
  — expected `"13 messages summarized"` → new count excludes the system prompt
  (12) and the kept tail is group-aligned.
- `tests/test_tools.py::test_compact_chat_uses_bound_summarizer` — the captured
  `old` no longer contains the system prompt / recall seed.
- `tests/test_tools.py::test_module_default_compact_chat` — expected
  `"Chat compacted: 4 messages summarized into 1, 4 kept."` → new count.
- `tests/test_tools.py::test_compact_chat_fallback_summary_without_summarizer`
  and `::test_compact_chat_reseeds_personal_data` — index-based assertions on
  `messages[2]` stay valid; re-check after the head handling change.
- `tests/test_token_reduction.py::test_auto_compact_*` — add the
  "nothing to do" → `""` case (new) and keep the existing error mapping.
- `tests/test_token_reduction.py::test_truncate_tool_result_*` — unchanged
  (they exercise the `live_ai` wrapper); one new assertion that the wrapper
  delegates to `common.truncate_text`.
- `tests/test_tools.py::test_compact_chat_falls_back_when_llm_summarizer_fails`
  — still valid (asserts `"old 0"` and the `[Compacted context on ` prefix); the
  extractive marker inside the summary changes from `[...]` to
  `… [truncated, N chars total] …` (T2b), which that test does not assert on.

## Risks / open points

1. **Counting change** is user-visible only in the tool's result string; the
   `"Chat compacted: ..."` prefix is preserved so `_auto_compact`'s
   `startswith("Error:")` check and the session colouring keep working.
2. **`_head_len` heuristic** (compare `messages[1]` with the memory file text)
   is exact for histories seeded by `clear_chat`/`_reseeded_head`; a user
   message that happens to equal the memory file content would also be treated
   as the seed — harmless (it is the same text).
3. **Char-count guard** is a heuristic (chars, not tokens) but deterministic and
   cheap; it never blocks a compaction that genuinely shrinks the history.
4. **Sequential fallback** for history-mutating rounds reduces parallelism only
   for rounds that touch the history (rare, and those tools are order-sensitive
   anyway).
5. **Shared truncation helper** (`src/common.py`, decision 2) touches
   `live_ai`'s import graph; the move is mechanical, `_truncate_tool_result`
   stays as a delegating wrapper, and the existing `max_tool_result_chars` tests
   keep covering it.
6. **Extractive-fallback marker change** (`\n[...]\n` → `\n… [truncated, N chars
   total] …\n`) is visible in the compacted summary text; no test or code
   depends on the old marker (checked: `tests/test_tools.py` only asserts on
   `"old 0"` / `"[Compacted context on "`).
7. **Summary cap default** keeps `max_tool_result_chars = 0` (uncapped) as the
   historical behavior, so the cap only kicks in for setups that already opted
   into tool-result capping; the shipped `config.toml` sets `4000`.
8. **Task order** T2 (helper move) precedes T2b (summary cap) because T2b reuses
   `truncate_text`; T2's commit is a pure refactor and is independently
   revertible.
