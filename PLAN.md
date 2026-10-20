# PLAN.md — MCP (Model Context Protocol) server integration — Option C

> **Audience:** an AI coding agent (or a developer) implementing this feature
> task by task. Follow the tasks **in order**. **One commit per task** — never
> mix two tasks in the same commit, and only commit when that task's tests pass.

---

## 1. Goals

Integrate MCP servers as an additional tool source for the Live Assistant agent:

1. **G1 — Dynamic tool injection (primary).** Tools exposed by configured MCP
   servers appear to the model as *native* OpenAI function-calling tools
   (definitions built from the MCP `inputSchema`), callable in the existing
   `act()` loop without any extra discovery round.
2. **G2 — Generic bridge fallback.** Two always-available tools,
   `mcp_list_tools` and `mcp_call_tool`, let the model discover and invoke MCP
   tools at runtime — covering servers that are offline at startup or added
   later via `/mcp refresh`.
3. **G3 — Zero impact when unused.** With no `[mcp]` section in `config.toml`
   (or the `mcp` package not installed), behavior is exactly today's: no new
   tools, no startup cost, no import of the SDK.
4. **G4 — Robustness.** A failing/unreachable MCP server must never prevent
   the agent from booting; it degrades to an error string (the existing tool
   error contract) and can be retried via `/mcp refresh`.
5. **G5 — Thread safety.** MCP calls are safe under the multi-agent pool
   (`ask_parallel` fans out on threads) and under Ctrl+C cancellation.
6. **G6 — Verified by unit tests.** Every task ships pytest tests that run
   offline (no real MCP server, no network); one optional integration test may
   use a local echo server.

## 2. Current behavior (as of this plan)

Facts the implementer must map to (verified against the code):

- **Tools are plain callables.** `src/live_ai.py::_default_tools()` returns a
  list of functions; `Agent.__init__` builds:
  - `self.tools` — the callables (also listed by `/tools` in `src/session.py`,
    which reads `__name__`/`__doc__`),
  - `self.tool_definitions` — OpenAI JSON-Schema definitions built by
    `OpenAICore._openai_tool_definition(tool)` (docstring → description,
    type hints → parameter schema),
  - `self._tool_impls` — `{tool_name: callable}` used by
    `Agent._execute_tool(call)`, which does `json.loads(arguments)`,
    `tool(**args)` and coerces the result with `str(...)`.
- **Tool results must be `str`.** Errors are returned as `"Error: ..."`
  strings (or raised and caught by `_execute_tool`); tools never crash the
  turn.
- **Config injection pattern.** `main.py` calls `src.tools.configure(config)`
  once after `load_config`; each tool submodule keeps the config section it
  needs (see `src/tools/__init__.py::configure`). Session-bound state is bound
  by `Brain.__init__` via `bind_*` functions.
- **Test seams live in their owning submodule** (e.g. `_VIDEO_CAPTURE_FACTORY`
  in `src/tools/camera.py`) and are patched there, not on the package.
- **Multi-agent pool.** `AgentPool` builds quiet agents with
  `isolated_tools=True`; `_build_tools()` re-instantiates session-bound tools
  per agent. `ask_parallel` runs tasks on a `ThreadPoolExecutor` — tool calls
  happen on multiple threads concurrently (per agent, serialized).
- **Slash commands.** `src/session.py::ConversationSession.COMMANDS` +
  `handle_command()` dispatch to `_cmd_*` methods; `/model`, `/tools`,
  `/session` are the patterns to copy.
- **UI/decorator.** Tools exposed to the model are wrapped with
  `@dump_func_call` (from `src.common`) for the `[CALL]` console block.
- **Tests.** pytest, offline. `tests/test_live_ai.py` builds bare agents via
  `Agent.__new__` (`_BareAgent`) and scripts `act()` rounds via
  `_bare_act_agent`; `tests/test_session.py` uses a FakeAgent; tool tests live
  in `tests/test_tools.py`.
- **Platform.** Development happens on Windows (the repo is used there);
  anything spawning subprocesses must work on Windows.

Today there is **no** MCP support: the model can only use the built-in tools
listed in `_default_tools()`.

## 3. Expected behavior (after implementation)

### 3.1 User-visible

- With `[mcp]` configured, the agent can call MCP tools **directly**, e.g. the
  model emits a tool call `mcp_filesystem_read_file {"path": "..."}` and the
  result feeds the next round — indistinguishable from built-in tools.
- `/mcp` lists configured servers, their connection state and their tools;
  `/mcp refresh` (re)connects all servers and rebuilds the injected tools.
- `mcp_list_tools` / `mcp_call_tool` exist as regular tools for runtime
  discovery/fallback.
- Without `[mcp]`: nothing changes (no new tools, no log noise beyond one
  debug line).

### 3.2 Configuration (added to `config.toml` in Task 7)

```toml
[mcp]
connect_timeout_s = 10.0   # per-server connect + tools/list budget
call_timeout_s    = 120.0  # per-tool-call budget

[mcp.servers.filesystem]      # stdio server: local process
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/Max/data"]

[mcp.servers.deepwiki]        # HTTP server: streamable HTTP endpoint
url = "https://mcp.deepwiki.com/mcp"
```

Rules: a server entry needs `command` (+ optional `args` list) **or** `url`;
anything else is skipped with a warning. Windows note for stdio servers: if
spawning `npx` directly fails, use `command = "cmd"`,
`args = ["/c", "npx", "-y", ...]`.

### 3.3 Architecture

New module `src/tools/mcp.py` (single file, ~300–400 lines) with:

- `configure(config)` — parse `[mcp]`; store server specs + timeouts in
  module state. **Does not connect** (cheap import/config; the connection
  happens lazily on first use).
- `Manager` — owns a dedicated background thread running an `asyncio` event
  loop; all MCP I/O happens on that loop. Public API is **synchronous**
  (called from agent threads):
  - `connect_all() -> dict[str, list[str]]` — connect every server, return
    `{server: [tool names]}`, capture per-server errors (never raise),
  - `list_tools(server) -> list[dict]`,
  - `call(server, tool, arguments) -> str`,
  - `refresh() -> dict` — drop and rebuild all connections,
  - `shutdown()`.
- **Connector seam (testability):** the Manager delegates transport work to a
  connector object injected via a module-level factory
  `_CONNECTOR_FACTORY` (patched in tests, mirroring the
  `_VIDEO_CAPTURE_FACTORY` pattern). The connector protocol:

  ```python
  class Client(Protocol):
      def list_tools(self) -> list[dict]: ...           # [{"name","description","inputSchema"}]
      def call(self, tool: str, arguments: dict) -> str: ...
      def close(self) -> None: ...

  class Connector(Protocol):
      def connect(self, name: str, spec: dict) -> Client: ...  # may be a coroutine function
  ```

  **Uniform sync/async invocation:** the Manager runs every connector/client
  method on its loop thread through a helper that awaits awaitables and
  returns plain results directly:

  ```python
  async def _invoke(self, fn, *args):
      result = fn(*args)
      if inspect.isawaitable(result):
          result = await result
      return result
  ```

  scheduled with `asyncio.run_coroutine_threadsafe(...).result(timeout)`.
  This lets unit tests inject **plain sync fakes** while the real connector
  (Task 3) is async — same Manager code path for both.
- Because all MCP I/O is funneled through **one** event-loop thread and each
  `_invoke` runs to completion before the next, per-server calls are naturally
  serialized — no extra locking is required for correctness (callers may block
  on `fut.result()` from several threads; that is safe).
- `mcp_tool_bindings() -> tuple[list[dict], dict[str, Callable]]` — builds
  OpenAI tool definitions (MCP `inputSchema` used verbatim as `parameters`)
  plus wrapper callables, with names `mcp_<server>_<tool>` sanitized to the
  OpenAI-compatible pattern `^[a-zA-Z0-9_-]{1,64}$` and collision-resolved
  with a `_2`, `_3` … suffix.
- Bridge tools `mcp_list_tools(server: str = "")` and
  `mcp_call_tool(server: str, tool: str, arguments: dict)` decorated with
  `@dump_func_call`, returning error strings when no manager is configured.
- `Agent.__init__` (in `src/live_ai.py`) appends the bindings after the
  built-in tool setup (works for the main agent **and** pooled agents, since
  the manager is process-wide); `Agent.refresh_mcp_tools()` replaces the
  injected set in place (used by `/mcp refresh`).

## 4. Global constraints (apply to every task)

- C1. Tools **never raise** to the model: return `"Error: ..."` strings.
- C2. All results returned to the model are `str`.
- C3. `import mcp` is **lazy** (inside functions), so the module imports
  without the SDK installed.
- C4. No behavior change when `[mcp]` is absent or empty.
- C5. Tests are offline: patch `_CONNECTOR_FACTORY` (or pass a fake connector)
  — never spawn real servers in unit tests.
- C6. Follow existing style: module docstring documenting the section,
  `getLogger(__name__)` logging, type hints, `@dump_func_call` on model-facing
  tools.
- C7. Each task ends with: `python -m pytest tests/test_mcp.py -v` green
  (from Task 1 on) **and** the pre-existing suite green
  (`python -m pytest`), then exactly one commit.

---

## 5. Tasks

### Task 1 — Plan + `[mcp]` config parsing and module skeleton

**Goal:** the module exists, parses config, wires into the existing
`configure()` chain, and is a no-op without `[mcp]`. No I/O yet.

**Files:** `PLAN.md` (this file, added by this commit), `src/tools/mcp.py`
(new), `src/tools/__init__.py`, `tests/test_mcp.py` (new).

**Steps:**
1. Commit `PLAN.md` together with this task's changes.
2. Create `src/tools/mcp.py` with a module docstring (document the `[mcp]`
   section like other tool modules do) and:
   - `_SERVERS: dict[str, dict]`, `_CONNECT_TIMEOUT_S`, `_CALL_TIMEOUT_S`
     module state, reset by `configure`,
   - `configure(config)`: read `getattr(config, "mcp", None)`; missing/empty →
     clear state and return; parse `mcp.servers` (a dict of
     `{server: {"command"|"url", "args"?}}`), keep only valid specs, warn and
     skip invalid ones; store timeouts with defaults 10.0 / 120.0.
   - `configured_servers() -> dict[str, dict]` and `get_manager()` returning
     `None` for now (Task 2 fills it).
3. In `src/tools/__init__.py`: import and re-export the two bridge tool names
   (stubs are fine in this task — see Task 4 for the real ones; to keep
   `_default_tools` stable, define them now as functions returning
   `"Error: MCP is not configured"`), add them to `__all__`, extend the
   docstring module list with `mcp`, and add the module to the
   `configure()` loop.

**Unit tests (`tests/test_mcp.py`):**
- `test_configure_without_section_is_noop` — `configure(SimpleNamespace())`
  leaves `configured_servers() == {}`.
- `test_configure_empty_servers_is_noop` — `[mcp]` with no `servers`.
- `test_configure_parses_stdio_and_url_specs` — both shapes stored verbatim.
- `test_configure_skips_invalid_server_spec` — entry with neither `command`
  nor `url` is dropped, valid siblings kept.
- `test_configure_timeout_defaults_and_overrides`.
- `test_mcp_bridge_tools_exist_in_default_tools` — in `tests/test_live_ai.py`
  style: `Agent.__new__(Agent)._build_tools()` contains `mcp_list_tools` and
  `mcp_call_tool`.

**Commit:** `feat(mcp): [mcp] config parsing and tool-module skeleton (plan)`

---

### Task 2 — Manager core (event-loop thread + sync facade, fake-connector ready)

**Goal:** a working `Manager` against the connector seam, fully testable
offline.

**Files:** `src/tools/mcp.py`, `tests/test_mcp.py`.

**Steps:**
1. Implement `Manager` as in §3.3:
   - constructor takes `servers`, timeouts and an optional `connector`
     (default: `_CONNECTOR_FACTORY()`),
   - `start()` spawns the daemon thread running `asyncio.new_event_loop()` +
     `run_forever()`; idempotent,
   - `_invoke` helper (§3.3) + `_submit(fn, *args, timeout) -> Any` using
     `asyncio.run_coroutine_threadsafe(...).result(timeout)`; converts
     `TimeoutError`/exceptions into raised `McpError` (module-level
     exception),
   - `connect_all()`: for each server call `connector.connect(name, spec)`
     (per-server `try/except` → store `error` string, continue), then
     `client.list_tools()`; store clients + tool lists; returns
     `{server: [tool names]}`; never raises,
   - `list_tools(server)`, `call(server, tool, arguments)`: raise `McpError`
     on unknown server/tool or unconnected server; `call` enforces
     `call_timeout_s` via the `_submit` timeout,
   - `refresh()`: close all clients, reconnect, return the same shape as
     `connect_all()`,
   - `shutdown()`: close clients, stop the loop (`loop.call_soon_threadsafe(
     loop.stop)`), join the thread (short timeout).
2. `get_manager()` returns the lazily-created module singleton
   (`_MANAGER`); `configure()` only stores state — the singleton is built on
   first `get_manager()` call **only when servers are configured**.
3. `connect_all` is called lazily by callers (Task 5 / Task 4), not by
   `configure`.

**Unit tests (FakeConnector — sync, records calls):**
- `test_manager_connect_all_returns_tool_lists`.
- `test_manager_connect_all_captures_per_server_errors` — one server's
  `connect` raises; the other still connects; error string stored.
- `test_manager_call_returns_client_result` (str passthrough).
- `test_manager_call_unknown_server_raises`, `test_manager_call_unknown_tool_raises`.
- `test_manager_call_timeout_raises` — fake client `call` sleeps longer than
  a tiny `call_timeout_s` (e.g. 0.2 s); expect `McpError` mentioning timeout.
- `test_manager_list_tools_unconnected_raises`.
- `test_manager_refresh_reconnects_and_replaces_clients`.
- `test_manager_invokes_async_connector_methods` — a connector whose
  `connect`/client methods are `async def` works identically (guards the
  `_invoke` coroutine contract used by Task 3).
- `test_get_manager_none_without_servers`,
  `test_get_manager_builds_singleton_with_servers` (monkeypatch
  `_CONNECTOR_FACTORY` with the fake; reset `_MANAGER` in a fixture).

**Commit:** `feat(mcp): manager with event-loop thread and connector seam`

---

### Task 3 — Real MCP connector (stdio + streamable HTTP)

**Goal:** talk to real MCP servers using the official `mcp` SDK, behind the
same seam.

**Files:** `src/tools/mcp.py`, `tests/test_mcp.py`, `README.md` (one line:
optional dependency `pip install mcp`).

**Steps:**
1. `McpSdkConnector` with **async** `connect(name, spec)`:
   - lazy `from mcp import ClientSession, StdioServerParameters` and
     `from mcp.client.stdio import stdio_client` /
     `from mcp.client.streamable_http import streamablehttp_client`
     (import errors → `McpError("mcp package not installed (pip install mcp)")`),
   - keep one module-level `AsyncExitStack` **owned by the loop thread's
     task context**: enter `stdio_client(StdioServerParameters(command=...,
     args=...))` or `streamablehttp_client(url)` plus `ClientSession(...)`,
     call `initialize()`, and return a `_SdkClient` that holds the session,
   - `_SdkClient.list_tools()` → normalize `tools/list` results into
     `{"name", "description", "inputSchema"}` dicts,
   - `_SdkClient.call(tool, arguments)` → `session.call_tool(...)`; flatten
     the result with a **pure helper** `_flatten_result(result) -> str`:
     join `text` parts of `result.content`; if empty and
     `result.structuredContent` exists, `json.dumps` it; prefix
     `"Error: ..."` when `result.isError`,
   - wrap `call` in `asyncio.wait_for(..., call_timeout_s)` so a hung call
     cannot block the shared loop (the outer `_submit` timeout stays as the
     hard net),
   - `close()` exits the stack entries for that server (best effort).
2. `_CONNECTOR_FACTORY` returns `McpSdkConnector()` by default.

**Unit tests:**
- All Task 2 tests still green (the fakes never touch the SDK).
- `test_flatten_result_joins_text_content`,
  `test_flatten_result_uses_structured_content`,
  `test_flatten_result_marks_is_error`,
  `test_flatten_result_empty_returns_placeholder` — pure-function tests with
  `SimpleNamespace` stand-ins for `CallToolResult`; **no SDK import needed**.
- `test_connector_missing_sdk_raises_mcperror` — monkeypatch
  `sys.modules` so `import mcp` fails; expect `McpError` with the pip hint.
- `test_connector_builds_stdio_parameters` — patch the SDK symbols with
  fakes and assert `StdioServerParameters` receives command/args (skipif
  `mcp` not installed).

**Commit:** `feat(mcp): real connector for stdio and streamable-http servers`

---

### Task 4 — Bridge tools (`mcp_list_tools`, `mcp_call_tool`)

**Goal:** the two fallback tools become real, model-facing tools.

**Files:** `src/tools/mcp.py`, `src/tools/__init__.py` (replace the Task 1
stubs with re-exports of the real functions), `tests/test_mcp.py`.

**Steps:**
1. Implement with `@dump_func_call` and model-oriented docstrings (mention
   that `mcp_list_tools` should be used first to discover names/schemas):
   - `mcp_list_tools(server: str = "") -> str`: empty `server` → all servers
     (name, status, tool names + one-line descriptions); unknown/failed
     server → `"Error: ..."`.
   - `mcp_call_tool(server: str, tool: str, arguments: dict) -> str`:
     delegates to `Manager.call`; `arguments` may arrive as a JSON string —
     accept both (`json.loads` when `str`).
2. Both return `"Error: MCP is not configured (no [mcp] servers)"` when
   `get_manager()` is `None`.

**Unit tests:**
- `test_mcp_list_tools_without_manager_returns_error`,
  `test_mcp_call_tool_without_manager_returns_error`.
- `test_mcp_call_tool_delegates_to_manager` — monkeypatch `get_manager` with
  a fake; assert arguments and str result.
- `test_mcp_call_tool_accepts_json_string_arguments`.
- `test_mcp_list_tools_renders_all_servers` — fake manager with two servers
  (one failed): output contains both names and the error.

**Commit:** `feat(mcp): mcp_list_tools and mcp_call_tool bridge tools`

---

### Task 5 — Dynamic tool injection into the Agent

**Goal:** MCP tools appear as native function-calling tools (G1) with refresh
support.

**Files:** `src/live_ai.py`, `src/tools/mcp.py`, `tests/test_mcp.py`,
`tests/test_live_ai.py`.

**Steps:**
1. In `src/tools/mcp.py` implement `mcp_tool_bindings()`:
   - `get_manager()` is `None` → `([], {})` (cheap, no connection attempt),
   - otherwise `connect_all()` (already-connected servers are reused), then
     for every server/tool build:
     - definition
       `{"type": "function", "function": {"name": tool_name, "description":
       description or "", "parameters": inputSchema}}` — **bypass**
       `_openai_tool_definition` (the schema comes from MCP); guard against
       missing/invalid `inputSchema` (skip the tool, warn),
     - wrapper callable `make_wrapper(server, tool)`:
       `def wrapper(**kwargs) -> str: return manager.call(server, tool, kwargs)`
       with `wrapper.__name__ = tool_name` and a `__doc__` built from the
       description (so `/tools` listing works),
     - name = `mcp_{server}_{tool}` sanitized (`[^A-Za-z0-9_-]` → `_`,
       truncated to 64); collisions with built-in tools or other MCP tools
       get a `_2`, `_3` … suffix,
   - return `(definitions, {name: wrapper})`.
2. In `src/live_ai.py::Agent.__init__`, after `self._tool_impls` is built:

   ```python
   from src.tools import mcp as mcp_tools
   extra_defs, extra_impls = mcp_tools.mcp_tool_bindings()
   self._mcp_tool_names: list[str] = []
   if extra_defs:
       self.tools.extend(extra_impls.values())
       self.tool_definitions.extend(extra_defs)
       self._tool_impls.update(extra_impls)
       self._mcp_tool_names = list(extra_impls)
   ```

   (Pooled agents get the same treatment automatically — the manager is
   process-wide; isolated memory/vision tools are unaffected.)
3. Add `Agent.refresh_mcp_tools()`: drop every name in
   `self._mcp_tool_names` from `tools`/`tool_definitions`/`_tool_impls`,
   call `manager.refresh()`, re-run the injection, update
   `_mcp_tool_names`; returns a short human-readable summary string.
   No-op with a log line when there is no manager.

**Unit tests:**
- In `tests/test_mcp.py` (fake manager via monkeypatched `get_manager`):
  - `test_bindings_empty_without_manager`,
  - `test_bindings_build_definitions_from_input_schema` — schema passes
    through verbatim; description from MCP description,
  - `test_bindings_prefix_and_sanitize_names` — e.g. server `web.data`, tool
    `read file!` → `mcp_web_data_read_file_`,
  - `test_bindings_resolve_name_collisions` — two servers exposing the same
    tool name (and a name colliding with a built-in such as `web_search`),
  - `test_bindings_skip_tools_with_invalid_schema`,
  - `test_bindings_wrapper_calls_manager` — wrapper forwards kwargs and
    returns the manager result.
- In `tests/test_live_ai.py` (follow the `_fake_openai`/`_remote_config`
  pattern; monkeypatch `src.tools.mcp.get_manager`):
  - `test_agent_init_appends_mcp_tools` — after `Agent(...)`,
    `tool_definitions` contains the MCP definition, `_tool_impls` the
    wrapper, `tools` the wrapper, `_mcp_tool_names` recorded,
  - `test_agent_without_mcp_manager_unchanged` — no manager → tool count and
    definitions identical to today (regression guard for G3),
  - `test_act_loop_calls_mcp_tool` — scripted `_stream_round` rounds
    (`_bare_act_agent` style, or the `_thinking_act_agent` pattern): round 1
    requests `mcp_test_echo`, round 2 answers; assert the wrapper ran and the
    `tool` message landed in the history,
  - `test_refresh_mcp_tools_replaces_entries`.

**Commit:** `feat(mcp): inject MCP server tools as native agent tools`

---

### Task 6 — `/mcp` slash command

**Goal:** visibility + runtime refresh from the chat.

**Files:** `src/session.py`, `tests/test_session.py`.

**Steps:**
1. Add `"/mcp"` to `COMMANDS` (Tab completion picks it up automatically) and
   a `handle_command` branch → `_cmd_mcp(arg)`.
2. `_cmd_mcp`:
   - no arg → table of servers: name, transport (`stdio`/`http`), status
     (`connected`/`error: ...`/`not configured`), tool count; plus a usage
     line for `refresh`,
   - `refresh` → `agent.refresh_mcp_tools()` when available, else
     `src.tools.mcp` manager `refresh()`; print the summary,
   - unknown sub-command → usage message.
3. Style: follow `_cmd_model`/`_model_list` (rich table with plain-text
   fallback).

**Unit tests (FakeAgent pattern of `tests/test_session.py`):**
- `test_cmd_mcp_lists_servers` — output contains server names and statuses.
- `test_cmd_mcp_refresh_calls_agent_refresh` — FakeAgent records the call.
- `test_cmd_mcp_without_manager_shows_not_configured`.
- `test_mcp_command_listed_in_commands` — `"/mcp" in ConversationSession.COMMANDS`.

**Commit:** `feat(session): /mcp command to inspect and refresh MCP servers`

---

### Task 7 — Docs, config example, integration test, final verification

**Goal:** ship it usable and prove it end-to-end once.

**Files:** `config.toml` (commented-out example block only), `README.md`,
`tests/test_mcp_integration.py` (new, optional), `src/tools/__init__.py`
(docstring already updated in Task 1 — final read-through).

**Steps:**
1. `config.toml`: add the `[mcp]` block from §3.2 **fully commented out**
   (feature stays off by default), including the Windows `cmd /c npx` note.
2. `README.md`: short "MCP servers (optional)" section — what it does, the
   config snippet, `pip install mcp`, `/mcp` usage, security note (MCP
   servers execute code with the user's privileges; only configure servers
   you trust — same class of trust as `run_command`).
3. Optional but recommended integration test
   `tests/test_mcp_integration.py`: a tiny in-repo MCP **echo server**
   (script using the `mcp` SDK's server API, started via
   `sys.executable tests/mcp_echo_server.py` over stdio), marked
   `@pytest.mark.skipif(...)` when the `mcp` package is missing; asserts
   `connect_all` lists the echo tool and `call` echoes a payload. Skipped
   silently in offline/CI runs without the SDK.
4. **Full verification** (§6) and fix anything found.

**Commit:** `docs(mcp): config example, README, integration smoke test`

---

## 6. Verification

Run from the project root (Windows PowerShell):

```powershell
python -m pytest tests/test_mcp.py -v            # Task 1 onward
python -m pytest tests/test_live_ai.py tests/test_session.py -v   # Tasks 5–6
python -m pytest                                  # full suite, every task
```

**Manual smoke test (Task 7, optional but recommended):**

1. `pip install mcp`
2. Uncomment a stdio server in `config.toml` (e.g.
   `@modelcontextprotocol/server-everything`).
3. `python main.py --chat`
4. Check startup log lines for MCP connection results; run `/mcp` (lists
   servers/tools); ask the model something that requires an MCP tool (e.g.
   "use the echo tool from the everything server with text 'ping'") and
   confirm the `[CALL]` block shows the `mcp_*` tool and the result feeds the
   answer; run `/mcp refresh` and confirm tools still work.

**Definition of Done (global checklist):**

- [ ] All tasks committed separately, each with its tests green before the commit.
- [ ] Full pre-existing test suite green (no regressions, G3 verified by
      `test_agent_without_mcp_manager_unchanged`).
- [ ] With `[mcp]` unset: startup output and `/tools` listing identical to
      today except the two bridge tools.
- [ ] A deliberately broken server URL/command degrades to error strings;
      the agent still boots and answers (G4).
- [ ] Two parallel `ask_parallel` tasks hitting the same MCP server both
      return correct results (G5; covered by the single-loop design +
      `test_manager_call_returns_client_result` under threads if feasible).

## 7. Risks & gotchas

- **Async/sync boundary:** never call `asyncio.run()` inside a tool (it would
  collide with the pool's threads and kill the stdio transports). Only the
  Manager's dedicated loop touches async APIs.
- **Stdio transports are stateful processes:** the `AsyncExitStack` must stay
  open for the session lifetime; closing it kills the server process. Only
  `refresh()`/`shutdown()` may close it.
- **Windows subprocess spawning:** `npx` is `npx.cmd` on Windows; document
  the `cmd /c` workaround (§3.2) rather than special-casing code.
- **Model-dependent schema quality:** some small local models mishandle large
  or unusual JSON Schemas; MCP `inputSchema` is passed through as-is — if a
  server's schema is exotic, the bridge tools (`mcp_call_tool`) remain a
  working fallback.
- **Timeout semantics:** a call that hits the outer timeout may still complete
  later on the loop; its result is discarded. Acceptable; documented.
- **Name collisions** with built-in tools are resolved by suffixing — never
  silently shadow `web_search` etc.

## 8. Out of scope (explicitly)

- MCP *resources* and *prompts* (a future plan could map resources onto the
  `query_corpus` machinery and prompts onto slash commands).
- OAuth/authenticated remote servers (plain URL only).
- Per-agent MCP tool filtering.
- Automatic reconnection timers (manual `/mcp refresh` only).
