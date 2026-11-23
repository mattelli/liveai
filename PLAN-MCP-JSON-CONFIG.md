# PLAN — MCP servers configured through `mcp.json` in the data dir

## Goals

1. MCP servers are defined **only** in a JSON file (`mcp.json`) stored under
   `[cwd] data_dir_cwd` (default `<launch>/.liveai/mcp.json`).
2. The JSON file location is configurable through a new key
   `servers_file_cwd` in the `[mcp]` section of `config.toml`.
3. `connect_timeout_s` / `call_timeout_s` stay in `config.toml` (not in JSON).
4. `[mcp.servers.*]` tables in TOML are **no longer supported** (removed).
5. A missing/unreadable/malformed `mcp.json` is ignored **silently** (no file
   is auto-created, no error surfaced to the user).
6. `/mcp refresh` (and `Agent.refresh_mcp_tools`) **re-read `mcp.json` from
   disk** before reconnecting, so edits take effect without a restart.

## Current behavior

- `src/tools/mcp.py`
  - `configure(config)` reads `[mcp]` from `config.toml`:
    `connect_timeout_s`, `call_timeout_s`, and one `[mcp.servers.<name>]`
    table per server (`command` + optional `args`, or `url`), plus the auth
    extras `env` / `headers` / `token_env` with `${VAR}` expansion.
  - Parsed specs are normalized to plain dicts and stored in the module state
    `_SERVERS`; `get_manager()` builds the process-wide `Manager` lazily.
  - `Manager.__init__` copies `_SERVERS`; `Manager.refresh()` only closes the
    clients and reconnects **with the specs it already holds** (no disk read).
  - `mcp_tool_bindings()` injects `mcp_<server>_<tool>` tools into the agent.
- `src/live_ai.py` — `Agent.refresh_mcp_tools()` calls `manager.refresh()`,
  then re-runs the injection; returns a summary string.
- `src/session.py` — `/mcp` renders `manager.status()`; `/mcp refresh` calls
  `agent.refresh_mcp_tools()`.
- `src/common.py` — `load_config()` resolves every `*_cwd` key under
  `[cwd] data_dir_cwd` (default `.liveai`); `ensure_data_dirs()` creates the
  `[cwd]` folders (data root, tmp, behaviour).
- `config.toml` / `config-code.toml` — `[mcp]` holds the two timeouts and the
  `[mcp.servers.*]` examples (`finance`, `filesystem`, `deepwiki`, ...).
- `tests/test_mcp.py` (926 lines) asserts the TOML `[mcp.servers]` parsing;
  `tests/test_live_ai.py`, `tests/test_session.py`, `tests/test_main.py`,
  `tests/test_token_reduction.py` touch MCP indirectly.

## Expected behavior

- `config.toml`:

  ```toml
  [mcp] # MCP (Model Context Protocol) servers as agent tools (optional; pip install mcp)
  servers_file_cwd = "mcp.json"  # server definitions (under [cwd] data_dir_cwd)
  connect_timeout_s = 15.0       # per-server connect + tools/list budget
  call_timeout_s    = 20.0       # per-tool-call budget
  ```

  `servers_file_cwd` is a `_cwd` key, so `load_config` resolves it under
  `data_dir_cwd` → `<launch>/.liveai/mcp.json`. When the key is absent the
  default `"mcp.json"` is used (same resolution).

- `mcp.json` (flat object, one entry per server, same keys as before):

  ```json
  {
    "finance": { "command": "uvx", "args": ["yfmcp@latest"] },
    "deepwiki": { "url": "https://mcp.deepwiki.com/mcp" },
    "telegram": {
      "command": "python",
      "args": ["-m", "telegram_bot_mcp"],
      "env": { "TELEGRAM_BOT_TOKEN": "${TELEGRAM_BOT_TOKEN}" }
    }
  }
  ```

  - Entry needs `command` (+ optional `args`) or `url`; other entries are
    skipped with a warning (unchanged rule).
  - `env` / `headers` / `token_env` and `${VAR}` expansion keep working
    exactly as today (secrets stay in `.env`, never in `mcp.json`).
  - A missing file, invalid JSON, a non-object root, or a non-object entry is
    ignored silently → MCP integration stays off (no exception, no log spam
    above `debug`).

- `[mcp.servers.*]` in TOML is ignored (the key is simply not read anymore).
- `/mcp refresh` re-reads `mcp.json` from disk, rebuilds the manager's server
  map, reconnects, and re-injects the tools; a server added/removed/edited in
  the file is picked up without restarting the session.
- `Manager.status()` / `/mcp` reflect the servers of the last read.

## Design notes

- New module state `_SERVERS_FILE: Path | None` set by `configure()`.
- New pure helper `_load_servers_file(path) -> dict[str, dict]`: reads JSON,
  normalizes specs (reusing `_string_map`, `_resolve_placeholders`,
  `_resolve_token_env`), returns `{}` on any problem (silent).
- New module function `refresh_servers() -> dict[str, list[str]]`: re-reads
  the file, updates `_SERVERS`, and asks the existing manager to
  `reload(_SERVERS)` (no-op `{}` when no manager).
- New `Manager.reload(servers)`: replaces `self._servers` (closing removed
  servers' clients) and reconnects; `Manager.refresh()` keeps its current
  meaning (reconnect with the held specs) and is reused by `reload`.
- `src/live_ai.py` `refresh_mcp_tools()` calls `mcp_tools.refresh_servers()`
  instead of `manager.refresh()`; the session `/mcp refresh` fallback path
  (no `refresh_mcp_tools`) calls `mcp_tools.refresh_servers()` too.
- `_spec_items()` is repurposed/kept for plain-dict input; the TOML
  namespace branch is dropped.

## Verification (unit tests)

All tests are offline (no real MCP server, no `mcp` SDK), following the
existing `tests/test_mcp.py` style (`FakeClient`, `_CONNECTOR_FACTORY` seam,
`tmp_path` for the JSON file).

- `configure()`:
  - reads servers from `mcp.json` (stdio + url entries) and stores them;
  - missing file / invalid JSON / non-object root → `configured_servers() == {}`
    and no exception;
  - entry without `command`/`url` is skipped, valid sibling kept;
  - `env`/`headers`/`token_env` + `${VAR}` expansion still applied;
  - `connect_timeout_s` / `call_timeout_s` still read from TOML (defaults and
    overrides);
  - `[mcp.servers.*]` in TOML is ignored (no servers parsed from it);
  - absent `servers_file_cwd` falls back to `mcp.json` under the data dir.
- `_load_servers_file()`: pure-function cases (missing, malformed, wrong
  types, scalars in `env` dropped).
- `refresh_servers()` / `Manager.reload()`:
  - re-reads the file: a server added on disk appears after refresh, a
    removed one disappears, and its client is closed;
  - no manager configured → returns `{}` and does not raise.
- `Agent.refresh_mcp_tools()`: picks up a server added to `mcp.json` after
  startup (fake connector), keeps `/tools off` state, drops stale tools.
- `session` `/mcp refresh`: uses the re-read specs (existing tests adapted).
- Regression: `tests/test_mcp.py`, `tests/test_live_ai.py`,
  `tests/test_session.py`, `tests/test_main.py`,
  `tests/test_token_reduction.py`, `tests/test_mcp_integration.py` all pass
  (`python -m pytest tests -q`).

## Tasks (one commit per task)

> Process: implement **one task at a time**, run its tests, commit, then
> **ask for confirmation before starting the next task**. Each commit message
> details what was done (files touched, behavior change, tests added).

### Task 1 — JSON loader + `servers_file_cwd` plumbing (no behavior switch)

- Add `_SERVERS_FILE` module state and `_load_servers_file(path)` (silent on
  any error) to `src/tools/mcp.py`.
- Add the `servers_file_cwd` default (`"mcp.json"`) in `src/common.py`
  `load_config` (`[mcp]` section defaults) so the key always resolves under
  `data_dir_cwd`.
- Unit tests for `_load_servers_file` (valid, missing, malformed, wrong
  types, auth extras, `${VAR}`).
- Commit message: describes the new loader, the config default, the silent
  error policy, and the tests added. No runtime behavior change yet.

### Task 2 — `configure()` reads servers from `mcp.json` only

- `configure()`: read `servers_file_cwd` from `[mcp]`, load the servers via
  `_load_servers_file`, keep `connect_timeout_s` / `call_timeout_s` from TOML,
  drop the `[mcp.servers.*]` TOML parsing (and the namespace branch of
  `_spec_items`).
- Update `tests/test_mcp.py` `configure()` tests to the JSON file
  (`tmp_path`), including "TOML `[mcp.servers]` ignored" and the
  missing-file no-op cases.
- Commit message: describes the switch to `mcp.json`, the removed TOML
  support, the unchanged timeout handling, and the adapted tests.

### Task 3 — `/mcp refresh` re-reads `mcp.json`

- Add `Manager.reload(servers)` (replace specs, close removed clients,
  reconnect) and module-level `refresh_servers()` (re-read file → update
  `_SERVERS` → `manager.reload`).
- `src/live_ai.py` `refresh_mcp_tools()` and the `src/session.py` `/mcp
  refresh` fallback call `mcp_tools.refresh_servers()`.
- Unit tests: server added/removed/edited on disk is picked up after refresh;
  removed client closed; no-manager path returns `{}`; agent-level test with
  the fake connector; `/tools off` state preserved.
- Commit message: describes the hot-reload path, the new `Manager.reload`,
  the wiring in `live_ai`/`session`, and the tests.

### Task 4 — Config files, README and docstrings

- `config.toml` / `config-code.toml`: remove every `[mcp.servers.*]` example,
  add `servers_file_cwd = "mcp.json"`, document the JSON shape in comments.
- `src/tools/mcp.py` module docstring: replace the `[mcp.servers.*]` TOML
  example with the `mcp.json` format + `servers_file_cwd`.
- `README.md` "MCP servers" section: JSON file, location under
  `data_dir_cwd`, silent-missing policy, `/mcp refresh` re-reads the file.
- Commit message: describes the documentation/config update and the removal
  of the deprecated TOML examples.

### Task 5 — Full verification

- Run `python -m pytest tests -q` (and `python main.py --tools-list`) to
  confirm no regression; fix anything found.
- Commit message: reports the test run result and any fix applied.

## Open points (assumed unless you say otherwise)

- `servers_file_cwd` is resolved under `data_dir_cwd` (because of the `_cwd`
  suffix); an absolute path in the config is used as-is.
- `mcp.json` is never auto-created; `ensure_data_dirs` is untouched.
- JSON root must be an object; a JSON list or scalar is ignored silently.
- `token_env` / `env` / `headers` semantics are unchanged.
