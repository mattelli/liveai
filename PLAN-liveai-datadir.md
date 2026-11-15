# PLAN — Per-directory runtime data root (`.liveai`)

## Goals

Multiple runs of `main.py` currently share the same files/folders, because
every `*_file` / `*_dir` / `*_path` config key is resolved relative to the
**config file's directory** (the project root), regardless of where the
process was launched from.

Introduce a single **runtime data root** — a `.liveai/` folder created in the
**directory from which `main.py` is launched** — and move all per-run *state*
into it, so each launch directory keeps its own data:

```
<cwd>/.liveai/
├── memory.log          # personal memory        ([brain] memory_path)
├── sessions/           # /session store|list|load ([brain] sessions_dir)
├── agents.jsonl        # activity monitor log   ([agent] monitor_log_file)
├── input.log           # prompt_toolkit history ([input] history_file)
├── schedule.json       # task persistence       ([delegation.scheduler] persist_file)
├── tmp/                # scratch files the agent writes ([paths] tmp_dir)
└── behaviour/          # markdown instruction files ([paths] behaviour_dir)
```

Shared **assets** stay in the project root `.res/` (model weights, corpora,
voices, embeddings), and the **config files stay in the project root**
(shared). This is **Option B**: an explicit `[paths]` section drives the
resolution, keeping the "assets vs. state" split self-documenting.

## Current behavior

`src/common.py::load_config` (line 402):

```python
config_dir = config_path.parent.resolve()
...
elif k.endswith("_file") or k.endswith("_dir") or k.endswith("_path"):
    result[k] = resolve_path(v, base_dir)   # base_dir == config_dir
```

Every path key resolves against the config directory. The launchers
(`bin/liveai.bat`, `bin/liveai`, `bin/liveai-code`) run
`python %LIVEAI_DIR%\main.py --config %LIVEAI_DIR%\config.toml` **without
`cd`**, so the launch directory is irrelevant and all state lands in the
project root and is shared across runs.

Consumers (all receive already-absolute paths via `configure()`):

| Data | Config key | Consumer |
|---|---|---|
| personal memory | `[brain] memory_path` | `src/tools/memory.py:137` |
| sessions | `[brain] sessions_dir` | `src/session.py:615` |
| monitor log | `[agent] monitor_log_file` | `src/live_ai.py:421` |
| input history | `[input] history_file` | `main.py:249` |
| schedule | `[delegation.scheduler] persist_file` | `src/tools/scheduler.py:66` |
| tmp | *(none — ad-hoc `tempfile` / `Path.cwd()`)* | `files.py`, `pyeval.py`, `web.py`, `finance.py` |
| behaviour | *(none — `prompts_file` / `AGENT.md`)* | `src/live_ai.py:109` |

## Expected behavior

1. On startup, `load_config` computes a **data root**:
   - `[paths] data_dir` (default `".liveai"`), resolved relative to the
     **launch cwd** (`Path.cwd()`), not the config directory.
   - The folder is created (`mkdir(parents=True, exist_ok=True)`) if missing.
2. A fixed **allow-list** of *state* keys is resolved against the data root;
   every other `*_file` / `*_dir` / `*_path` key keeps resolving against the
   config directory (so `.res/...` assets are unchanged).
3. `[paths] tmp_dir` (default `tmp`) and `[paths] behaviour_dir`
   (default `behaviour`) are resolved under the data root and created.
4. Tools that currently default to `Path.cwd()` for scratch output
   (`download_file`, `save_financial_chart`, `extract_archive` default dest,
   the `pyeval` sandbox) default to `tmp_dir` instead. Explicit `out_dir`
   arguments still win.
5. `behaviour/` holds markdown instruction files; they are discovered by the
   agent through the **bootstrap prompt** (the prompt tells the agent to read
   the markdown files in the behaviour folder). No automatic prompt injection
   — discovery stays agent-driven.
6. Existing root-level `memory.log`, `.sessions/`, `input.log` are **not**
   migrated; runs start fresh in `.liveai/` (manual migration by the user).

## Design notes

- **`data_dir` must not be resolved against the config dir.** The existing
  suffix rule (`k.endswith("_dir")`) would resolve `data_dir` against
  `config_dir`; the new code resolves it against the launch cwd *first*, then
  uses it as the base for the state keys.
- **Recursion needs a key prefix.** `namespace_from_dict` is recursive and
  currently has no notion of the dotted key path. Add a `prefix: tuple`
  parameter so the resolver can look up `("brain","memory_path")` etc. in the
  state-key map.
- **State-key map** (dotted key -> base):
  ```
  ("brain", "memory_path")                -> data_root
  ("brain", "sessions_dir")               -> data_root
  ("agent", "monitor_log_file")           -> data_root
  ("input", "history_file")               -> data_root
  ("delegation", "scheduler", "persist_file") -> data_root
  ("paths", "tmp_dir")                    -> data_root
  ("paths", "behaviour_dir")              -> data_root
  ```
- **Out of scope:** an `LIVEAI_DATA_DIR` env override (not requested); moving
  `.res/` assets; migrating existing root state.

## Verification

- Unit tests for `load_config`:
  - `data_dir` resolves against `cwd` (monkeypatched) and the folder is
    created;
  - each state key resolves under the data root;
  - asset keys (`[embedding] model_path`, `[image_gen] out_dir`) still resolve
    against the config directory;
  - a custom `[paths] data_dir` / `tmp_dir` / `behaviour_dir` is honored;
  - missing `[paths]` section falls back to the defaults.
- Unit tests for the tmp defaults: `download_file`, `save_financial_chart`,
  `extract_archive` and the `pyeval` sandbox write under the configured
  `tmp_dir` when `out_dir`/dest is omitted, and honor an explicit value.
- Integration: `main.py --config <tmp config>` creates `.liveai/` in the cwd
  and the session/memory/monitor files land there.
- Update the existing tests that assert config-dir resolution for the moved
  keys (`tests/test_common.py::test_load_config_resolves_input_history_file`,
  `tests/test_agent_monitor.py::test_monitor_settings_load_config_round_trip`,
  `tests/test_scheduler_registration.py`).
- Full suite: `python -m pytest tests -q` green.

## Tasks (one commit each)

1. **`[paths]` section + data-root resolution in `load_config`.** Add the
   `prefix`-aware recursion, the state-key map, `data_dir` resolution against
   cwd, and folder creation for `data_dir` / `tmp_dir` / `behaviour_dir`.
   Update the affected existing tests. New tests in `tests/test_common.py`.
2. **Config files.** Add the `[paths]` block to `config.toml` and
   `config-code.toml`; change the moved keys to data-root-relative names
   (`memory_path = "memory.log"`, `sessions_dir = "sessions"`,
   `monitor_log_file = "agents.jsonl"`, `history_file = "input.log"`,
   `persist_file = "schedule.json"`), with comments explaining the split.
3. **tmp_dir wiring.** Add a small accessor for the configured tmp dir and
   make `download_file`, `save_financial_chart`, `extract_archive` (default
   destination) and the `pyeval` sandbox default to it instead of
   `Path.cwd()`. Explicit arguments still win. Tests per tool.
4. **behaviour_dir.** Ensure the folder is created; update the bootstrap
   prompt in both config files to instruct the agent to read the markdown
   files in the behaviour folder; document the convention. Tests: folder
   created, prompt text present.
5. **Docs + `.gitignore`.** README section on the `.liveai` data root and the
   assets-vs-state split; add `.liveai/` to `.gitignore`; record the
   verification run.
