# PLAN — New `audio` tool topic (move `allow_speech` + `transcribe_audio` out of `other`)

## Goals

1. Introduce a new tool topic named `audio` (lowercase key, displayed as `audio:`).
2. Move the two audio tools — `allow_speech` and `transcribe_audio` — into it.
3. Keep the existing `other` catch-all entity unchanged (it stays the fallback
   for ungrouped tools: MCP-injected tools, test doubles, etc.).
4. Place `audio` in the curated order **immediately before `other`** (i.e. last
   named topic in the list).
5. Make `audio` a first-class value for `[agent] tool_groups` and for
   `/tools on|off audio`.

## Current behavior

- `_TOOL_TOPICS` (`src/live_ai.py`, ~line 343) maps topic → tool names. It has
  no `audio` entry, so `allow_speech` and `transcribe_audio` are ungrouped.
- `tool_topics()` (display grouping for `/tools` and `/tools list`) resolves
  each tool name through `_TOOL_TOPICS` and falls back to `"other"` for any
  name not found (`src/live_ai.py:380`); `"other"` is appended last
  (`src/live_ai.py:383-384`).
- `_select_tools()` (`[agent] tool_groups` filter) uses the same map with the
  same `"other"` fallback (`src/live_ai.py:335`), and whitelists `"other"` as a
  valid group name (`src/live_ai.py:323`).
- Net effect today: `allow_speech` / `transcribe_audio` are shown under
  `other:`, and `tool_groups = "audio"` is rejected as an unknown group.

## Expected behavior

- `/tools` and `/tools list` show a group `audio:` (just before `other:`)
  containing exactly `allow_speech` and `transcribe_audio`.
- `other:` keeps containing only genuinely ungrouped tools (MCP tools, test
  doubles); it is never removed.
- `[agent] tool_groups = "audio"` keeps only the two audio tools (plus nothing
  else); `tool_groups = "audio, other"` keeps the audio tools and the
  catch-all.
- `/tools off audio` disables both audio tools; `/tools on audio` restores
  them.
- No change to any other topic, to the curated order of existing topics, or to
  the `other` fallback semantics.

## Implementation notes

- Single source of truth: add one entry to `_TOOL_TOPICS`:
  ```python
  "audio": ("allow_speech", "transcribe_audio"),
  ```
  placed as the **last named key** (after `"mcp"`), so `tool_topics()` lists it
  just before the appended `"other"`.
- `_select_tools()` and `tool_topics()` need **no logic change**: both already
  resolve names via `_TOOL_TOPICS`, so the two tools move out of `other`
  automatically and `audio` becomes a valid group for free.
- Update the `_TOOL_TOPICS` comment and the `_select_tools` / `tool_topics`
  docstrings only if they would otherwise be inaccurate (they describe the
  `other` fallback, which is unchanged).

## Verification with unit tests

- `tests/test_live_ai.py`
  - New: `allow_speech` + `transcribe_audio` group under `audio`, and `audio`
    appears immediately before `other` in `tool_topics()` output.
  - Keep: `test_tool_topics_unknown_tools_land_in_other_last` (unknown tool
    still lands in `other`, last) — must still pass unchanged.
  - Keep: `test_tool_topics_groups_known_tools_in_curated_order` — must still
    pass unchanged (no `audio` tool in its input).
- `tests/test_token_reduction.py`
  - New: `_select_tools(tools, "audio")` keeps only `allow_speech` +
    `transcribe_audio`; `"audio, other"` also keeps ungrouped tools.
  - New (or extend `test_toggle_tools_by_topic`): `toggle_tools(["audio"],
    enabled=False)` removes both audio tools from the enabled definitions.
- `tests/test_session.py`
  - `test_tools_lists_tools_grouped_by_topic` (test doubles → `other`) must
    still pass unchanged.
- Run the full suite (`pytest`) after each task; all tests green before the
  commit.

## Tasks (one commit per task)

> Workflow: implement **one task at a time**, run the tests, then **commit**.
> **Ask for confirmation before starting the next task.** Each commit message
> must detail what was done.

### Task 1 — Add the `audio` topic + display grouping tests
- Add `"audio": ("allow_speech", "transcribe_audio")` to `_TOOL_TOPICS`
  (`src/live_ai.py`), as the last named key (just before the `other`
  fallback).
- Add unit tests in `tests/test_live_ai.py` asserting the grouping and the
  position (immediately before `other`).
- Run `pytest`; commit, e.g.
  `feat(tools): add audio tool topic for allow_speech/transcribe_audio`.

### Task 2 — `tool_groups` filtering + runtime toggle coverage
- Add unit tests in `tests/test_token_reduction.py` for
  `_select_tools(..., "audio")` / `"audio, other"` and for
  `toggle_tools(["audio"], ...)`.
- No production code change expected (the map entry from Task 1 already
  enables both). If a gap is found, fix it here.
- Run `pytest`; commit, e.g.
  `test(tools): cover audio topic in tool_groups filtering and toggling`.

### Task 3 — Documentation
- Update `README.md` (tool-topics list around the `[agent] tool_groups`
  example, ~line 806) to mention the `audio` topic.
- Optionally note `audio` in the `config-code.toml` `tool_groups` comment.
- Run `pytest`; commit, e.g.
  `docs(tools): document the audio tool topic`.

## Open points / confirmations needed

- Commit the plan file `PLAN-AUDIO-TOOL-GROUP.md` itself? (as its own commit,
  or leave it untracked, or delete it after the work — the previous
  `PLAN-SCHEDULER-AGENT.md` was removed in commit `84893e1`.)
- Permission to run the test suite (`pytest`) locally between tasks.
