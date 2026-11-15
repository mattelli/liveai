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
├── memory.log          # personal memory        ([brain] memory_path_cwd)
├── sessions/           # /session store|list|load ([brain] sessions_dir_cwd)
├── agents.jsonl        # activity monitor log   ([agent] monitor_log_file_cwd)
├── input.log           # prompt_toolkit history ([input] history_file_cwd)
├── schedule.json       # task persistence       ([delegation.scheduler] persist_file_cwd)
├── tmp/                # scratch files the agent writes ([cwd] tmp_dir_cwd)
└── behaviour/          # markdown instruction files ([cwd] behaviour_dir_cwd)
```

Shared **assets** stay in the project root `.res/` (model weights, corpora,
voices, embeddings), and the **config files stay in the project root**
(shared).

**Resolution rule (refinement):** `[cwd] data_dir_cwd` is the **data root**,
resolved against the **launch directory** (`Path.cwd()`). Every **other** key
ending in `_cwd` is resolved **under that data root**, so its value stays
short (`memory_path_cwd = "memory.log"` -> `<launch>/.liveai/memory.log`).
Every other `*_file` / `*_dir` / `*_path` key keeps resolving against the
**config directory**. The `[cwd]` section holds the three folder keys
(`data_dir_cwd`, `tmp_dir_cwd`, `behaviour_dir_cwd`).

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
| personal memory | `[brain] memory_path_cwd` | `src/tools/memory.py:137` |
| sessions | `[brain] sessions_dir_cwd` | `src/session.py:615` |
| monitor log | `[agent] monitor_log_file_cwd` | `src/live_ai.py:421` |
| input history | `[input] history_file_cwd` | `main.py:249` |
| schedule | `[delegation.scheduler] persist_file_cwd` | `src/tools/scheduler.py:66` |
| tmp | *(none — ad-hoc `tempfile` / `Path.cwd()`)* | `files.py`, `pyeval.py`, `web.py`, `finance.py` |
| behaviour | *(none — `prompts_file` / `AGENT.md`)* | `src/live_ai.py:109` |

## Expected behavior

1. `load_config` resolves `[cwd] data_dir_cwd` against the **launch cwd**
   (`Path.cwd()`), then resolves every other `_cwd` key **under that data
   root**; all other path keys resolve against the config directory.
2. The `[cwd]` defaults (`data_dir_cwd = ".liveai"`, `tmp_dir_cwd = "tmp"`,
   `behaviour_dir_cwd = "behaviour"`) are filled in when the section/key is
   absent.
3. `ensure_data_dirs(config)` creates the data root and its `tmp` /
   `behaviour` subfolders (best-effort), called by `main.py` after
   `load_config`.
4. Tools that currently default to `Path.cwd()` for scratch output
   (`download_file`, `save_financial_chart`, the `pyeval` sandbox) default to
   `[cwd] tmp_dir_cwd` instead. Explicit `out_dir` arguments still win.
   (`extract_archive` already defaults *beside the archive*, not to the cwd,
   so it is left unchanged.)
5. `behaviour/` holds markdown instruction files; they are discovered by the
   agent through the **bootstrap prompt**. The prompt references the folder
   with a `{behaviour_dir}` placeholder, which `Agent.bootstrap()` replaces
   with the resolved `[cwd] behaviour_dir_cwd` path (literal `str.replace`, so
   other braces are untouched). No automatic prompt injection — discovery
   stays agent-driven.
6. Existing root-level `memory.log`, `.sessions/`, `input.log` are **not**
   migrated; runs start fresh in `.liveai/` (manual migration by the user).

## Design notes

- **Single rule, no allow-list.** `data_dir_cwd` selects the data root; every
  other `_cwd` key is resolved under it, so values stay short and the data
  root is defined once. The earlier `_STATE_PATH_KEYS` allow-list is dropped.
- **Out of scope:** an `LIVEAI_DATA_DIR` env override (not requested); moving
  `.res/` assets; migrating existing root state.

## Verification

- Unit tests for `load_config`:
  - `_cwd` keys resolve against the launch directory (monkeypatched cwd);
  - asset keys (`[embedding] model_path`, `[image_gen] out_dir`) still resolve
    against the config directory when the launch dir differs;
  - a custom `[cwd]` section is honored;
  - missing `[cwd]` section falls back to the defaults;
  - `ensure_data_dirs` creates the folders and tolerates a config without
    `[cwd]`.
- Unit tests for the tmp defaults: `download_file`, `save_financial_chart`
  and the `pyeval` sandbox write under the configured `tmp_dir_cwd` when
  `out_dir`/dest is omitted, and honor an explicit value.
- Unit tests for the bootstrap placeholders: `{behaviour_dir}` / `{data_dir}`
  substituted with the resolved `[cwd]` paths, other braces left untouched,
  and a missing `[cwd]` section keeping the prompt verbatim.
- Integration: `main.py --config <tmp config> --tools-list` launched from a
  different directory creates `.liveai/` (with `tmp/` and `behaviour/`) in the
  **launch** directory, not the project root. Verified manually.
- Full suite: `python -m pytest tests -q` -> 1130 passed.

## Tasks (one commit each)

1. **`[cwd]` section + `_cwd` resolution in `load_config`.** Add the `_cwd`
   suffix rule, the `[cwd]` defaults, and `ensure_data_dirs`; update the
   consumers and the affected tests. — DONE
2. **Config files.** Add the `[cwd]` block and rename the state keys to the
   `_cwd` form in `config.toml` and `config-code.toml`. — DONE
3. **tmp_dir wiring.** Make `download_file`, `save_financial_chart` and the
   `pyeval` sandbox default to `[cwd] tmp_dir_cwd` instead of `Path.cwd()`.
   Explicit arguments still win. Tests per tool. — DONE
4. **behaviour_dir.** Ensure the folder is created; update the bootstrap
   prompt in both config files to instruct the agent to read the markdown
   files in the behaviour folder; document the convention. Tests: folder
   created, prompt text present. — DONE
5. **Docs + `.gitignore`.** README section on the `.liveai` data root and the
   assets-vs-state split; add `.liveai/` to `.gitignore`; record the
   verification run. — DONE
