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

**Resolution rule (refinement):** every config key whose name ends in
**`_cwd`** is resolved against the **launch directory** (`Path.cwd()`); every
other `*_file` / `*_dir` / `*_path` key keeps resolving against the **config
directory**. The `_cwd` values carry the data-root prefix themselves
(`memory_path_cwd = ".liveai/memory.log"`), so no allow-list or data-root
computation is needed — the suffix alone selects the base. The `[cwd]`
section holds the three folder keys (`data_dir_cwd`, `tmp_dir_cwd`,
`behaviour_dir_cwd`).

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

1. `load_config` resolves any key ending in `_cwd` against the **launch cwd**
   (`Path.cwd()`); all other path keys resolve against the config directory.
2. The `[cwd]` defaults (`data_dir_cwd = ".liveai"`,
   `tmp_dir_cwd = ".liveai/tmp"`, `behaviour_dir_cwd = ".liveai/behaviour"`)
   are filled in when the section/key is absent.
3. `ensure_data_dirs(config)` creates the data root and its `tmp` /
   `behaviour` subfolders (best-effort), called by `main.py` after
   `load_config`.
4. Tools that currently default to `Path.cwd()` for scratch output
   (`download_file`, `save_financial_chart`, `extract_archive` default dest,
   the `pyeval` sandbox) default to `[cwd] tmp_dir_cwd` instead. Explicit
   `out_dir` arguments still win.
5. `behaviour/` holds markdown instruction files; they are discovered by the
   agent through the **bootstrap prompt** (the prompt tells the agent to read
   the markdown files in the behaviour folder). No automatic prompt injection
   — discovery stays agent-driven.
6. Existing root-level `memory.log`, `.sessions/`, `input.log` are **not**
   migrated; runs start fresh in `.liveai/` (manual migration by the user).

## Design notes

- **Single rule, no allow-list.** The `_cwd` suffix alone selects the launch
  directory base; the earlier `_STATE_PATH_KEYS` allow-list and the
  `data_root` computation are dropped.
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
- Unit tests for the tmp defaults: `download_file`, `save_financial_chart`,
  `extract_archive` and the `pyeval` sandbox write under the configured
  `tmp_dir_cwd` when `out_dir`/dest is omitted, and honor an explicit value.
- Integration: `main.py --config <tmp config>` creates `.liveai/` in the cwd
  and the session/memory/monitor files land there.
- Full suite: `python -m pytest tests -q` green.

## Tasks (one commit each)

1. **`[cwd]` section + `_cwd` resolution in `load_config`.** Add the `_cwd`
   suffix rule, the `[cwd]` defaults, and `ensure_data_dirs`; update the
   consumers and the affected tests. — DONE
2. **Config files.** Add the `[cwd]` block and rename the state keys to the
   `_cwd` form in `config.toml` and `config-code.toml`. — DONE
3. **tmp_dir wiring.** Make `download_file`, `save_financial_chart`,
   `extract_archive` (default destination) and the `pyeval` sandbox default to
   `[cwd] tmp_dir_cwd` instead of `Path.cwd()`. Explicit arguments still win.
   Tests per tool.
4. **behaviour_dir.** Ensure the folder is created; update the bootstrap
   prompt in both config files to instruct the agent to read the markdown
   files in the behaviour folder; document the convention. Tests: folder
   created, prompt text present.
5. **Docs + `.gitignore`.** README section on the `.liveai` data root and the
   assets-vs-state split; add `.liveai/` to `.gitignore`; record the
   verification run.
