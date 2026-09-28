# PLAN-MULTI-CORPUS.md — Per-corpus folders and agent corpus discovery

This plan replaces the single-corpus model of `PLAN-EMBEDDING-TOOL.md` with a
**named-corpus** model stored under `.res/corpora/`, and makes the agent aware of
the corpora through a discovery tool (`list_corpora`).

It is written for a **small automated coding agent**. Execute the tasks in order.
Each task is one commit and is not done until its tests pass.

> **Single design only.** This plan defines one mandatory path. Do not add
> optional parameters, aliases, fallbacks, "search all corpora" modes, legacy
> file-path support, or alternative storage layouts. `corpus_file` is removed
> everywhere in the agent-facing API.

---

## 1. Background

### 1.1 Current behavior (to be replaced)

- `[embedding]` in `config.toml` has a single `corpus_path`
  (`.res/corpus/corpus.pkl`) plus `processed_name`.
- `Brain.store_corpus(directory, corpus_file="", overwrite=False)` writes a
  pickle (and `PROCESSED.md`, and a sibling `<stem>_images/` folder) next to the
  given file.
- `Brain.query_corpus(query, corpus_file="", top_k=0, kind="")` reads that file.
- The agent cannot discover corpora: it must be told a path, and there is no
  listing tool. `corpus_file` also lets the model point at arbitrary paths.

### 1.2 Expected end state

- Every corpus is a **folder** under `.res/corpora/`:

  ```
  .res/corpora/
    autosar/
      corpus.pkl
      metadata.json
      processed.md
      images/
    quora/
      corpus.pkl
      metadata.json
      processed.md
      images/
  ```

- The agent-facing tools take a **corpus name**, never a file path:

  ```python
  store_corpus(self, directory: str, corpus_name: str, overwrite: bool = False) -> str
  query_corpus(self, query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str
  ```

- A new `list_corpora()` tool enumerates the available corpora from
  `metadata.json` (no pickle loading).
- The agent discovers the available corpora by calling `list_corpora()` (a
  registered tool).
- `corpus_file` no longer exists in `store_corpus`, `query_corpus`,
  `EmbeddingCorpus.store` or `EmbeddingCorpus.query`.

---

## 2. Design

### 2.1 Architecture

```
src/live_ai.py::Brain.store_corpus / query_corpus / list_corpora
        │  resolve corpus_name -> .res/corpora/<name>/
        ▼
src/embedding_search.py::EmbeddingCorpus        (build/query a corpus folder)
        │  uses
        ├── src/document_loader.py               (per-format text + image extraction)
        ├── sentence_transformers (text or CLIP), lazily
        └── faiss.IndexFlatIP (lazy; injectable backend)
```

Fixed in-folder names (constants in `src/embedding_search.py`):

```python
CORPUS_FILE = "corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_FILE = "processed.md"
IMAGES_DIR = "images"
```

### 2.2 Locked signatures

Implement exactly these. No extra parameters.

```python
# src/embedding_search.py
DEFAULT_CORPORA_DIR = ".res/corpora"
CORPUS_FILE = "corpus.pkl"
METADATA_FILE = "metadata.json"
PROCESSED_FILE = "processed.md"
IMAGES_DIR = "images"

def sanitize_corpus_name(name: str) -> str: ...
def resolve_corpus_dir(corpus_name: str, corpora_dir) -> "Path": ...
def save_metadata(corpus_dir, metadata: dict) -> None: ...
def load_metadata(corpus_dir) -> dict: ...
def list_corpora(corpora_dir) -> list[dict]: ...

class EmbeddingCorpus:
    def __init__(self, model_path, device="cpu", extensions=DEFAULT_EXTENSIONS,
                 chunk_size=DEFAULT_CHUNK_SIZE, top_k=DEFAULT_TOP_K,
                 model_factory=None, index_factory=None, sentence_tokenizer=None,
                 image_model_path=None, multimodal=False,
                 image_extensions=IMAGE_EXTENSIONS,
                 image_model_factory=None): ...
    def store(self, directory: str, corpus_dir, overwrite: bool = False) -> str: ...
    def query(self, corpus_dir, query: str, top_k=None, kind=None) -> str: ...

# src/live_ai.py
class Brain:
    @dump_func_call
    def list_corpora(self) -> str: ...

    @dump_func_call
    def store_corpus(self, directory: str, corpus_name: str, overwrite: bool = False) -> str: ...

    @dump_func_call
    def query_corpus(self, query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str: ...
```

Removed: `summary_name`/`processed_name`, `corpus_path`, `default_corpus`, and
every `corpus_file` parameter. `corpus_name` is a **required** parameter (no
default): an empty/blank value is still rejected (see §2.5).

### 2.3 Folder layout and `metadata.json`

`resolve_corpus_dir(corpus_name, corpora_dir)` returns
`<corpora_dir>/<sanitized_name>`. `sanitize_corpus_name`:

- strips surrounding whitespace,
- accepts `^[A-Za-z0-9][A-Za-z0-9._-]*$`,
- rejects `""`, `"."`, `".."`, any path separator, and any other character
  (raise `ValueError`),
- `corpus_name` is required (no default); empty/blank raises `ValueError`
  (see §2.5).

`metadata.json` (written on every successful `store`):

```json
{
  "name": "autosar",
  "version": 2,
  "mode": "multimodal",
  "text_model_path": ".../all-MiniLM-L6-v2",
  "image_model_path": ".../clip-ViT-B-32",
  "source_dir": "C:/.../.tmp/corpus",
  "records": 3762,
  "text_records": 3684,
  "image_records": 78,
  "sources": 22,
  "built_at": "2026-09-27 17:29:26 UTC"
}
```

`load_metadata(corpus_dir)` raises `FileNotFoundError` when `metadata.json` is
missing. `list_corpora(corpora_dir)` returns one dict per
`<corpora_dir>/*/metadata.json`, sorted by `name`; a missing `corpora_dir`
returns `[]`.

### 2.4 Config schema

Replace the single-corpus keys in `[embedding]`:

```toml
[embedding] # semantic search over one or more named local corpora
model_path = ".res/embedding/all-MiniLM-L6-v2"     # text-only-mode encoder
image_model_path = ".res/embedding/clip-ViT-B-32"  # shared text+image encoder (multimodal)
multimodal = true          # true -> use image_model_path for BOTH text and images
device = "cpu"             # "cpu" or "cuda"
corpora_dir = ".res/corpora"  # resolved relative to this config file
extensions = "txt, md, markdown, tex, rst, log, csv, html, htm, pdf, docx, xlsx, png, jpg, jpeg, webp, bmp, tiff"
image_extensions = "png, jpg, jpeg, webp, bmp, tiff"
chunk_size = 3             # sentences per indexed passage
top_k = 5                  # default number of results returned by query_corpus
```

`corpora_dir` ends in `_dir`, so `load_config` resolves it to an absolute path.
Remove `corpus_path` and `processed_name`.

### 2.5 `store_corpus` algorithm (Brain)

1. If `corpus_name.strip()` is empty → return
   `"Error: a corpus name is required (call list_corpora to see available corpora)."`.
2. `corpus_dir = resolve_corpus_dir(corpus_name, self.config.embedding.corpora_dir)`
   (raises `ValueError` → return `f"Error: {e}"`).
3. `self._embedding_corpus().store(directory, corpus_dir, overwrite)`.
4. Wrap in `try/except Exception` and return `f"Error: {e}"` on failure.

### 2.6 `query_corpus` algorithm (Brain)

1. If `corpus_name.strip()` is empty → return `"Error: a corpus name is required."`.
2. `corpus_dir = resolve_corpus_dir(corpus_name, self.config.embedding.corpora_dir)`.
3. `self._embedding_corpus().query(corpus_dir, query, top_k or None, kind or None)`.
4. Wrap in `try/except Exception` and return `f"Error: {e}"`.

### 2.7 `list_corpora` algorithm (Brain)

```python
def list_corpora(self) -> str:
    entries = list_corpora(self.config.embedding.corpora_dir)
    if not entries:
        return "No corpora available."
    lines = [f"Available corpora ({len(entries)}):"]
    for e in entries:
        lines.append(
            f"- {e['name']}: mode={e['mode']}, "
            f"records={e['records']} (text {e['text_records']} / image {e['image_records']}), "
            f"sources={e['sources']}, built={e['built_at']}"
        )
    return "\n".join(lines)
```

### 2.8 Agent awareness (tool surface)

Register `self.list_corpora` in `Agent._build_tools()` immediately before
`self.store_corpus`. Its docstring tells the model to call it before querying
when the user refers to a topic rather than a corpus name. No changes are made to
the system prompt, `Brain.__init__`, `clear_chat()` or `Agent.bootstrap()`.

### 2.9 `EmbeddingCorpus.store` / `query` (folder-based)

`store(directory, corpus_dir, overwrite)`:

1. `corpus_dir = Path(corpus_dir)`. If `corpus_dir / CORPUS_FILE` exists and
   `overwrite` is `False` → return `"Corpus already exists."`.
2. Delete and recreate `corpus_dir / IMAGES_DIR` (clean rebuild; a stale
   `corpus_dir` for a different corpus is never touched).
3. `documents = iter_documents(directory, self.extensions, image_dir=corpus_dir / IMAGES_DIR)`.
4. Build records exactly as today (per-page text sections, content filter, image
   dedupe). If none → `"No supported content found in '<directory>'."`.
5. Embed (CLIP for `multimodal`, text model otherwise) and
   `save_corpus(corpus_dir / CORPUS_FILE, payload)`.
6. `prune_orphan_images(corpus_dir / IMAGES_DIR, records)`.
7. Append processed absolute paths to `corpus_dir / PROCESSED_FILE`.
8. `save_metadata(corpus_dir, {name, version, mode, text_model_path,
   image_model_path, source_dir, records, text_records, image_records, sources,
   built_at})` — `name` is `corpus_dir.name`.
9. Return `f"Corpus '{corpus_dir.name}' saved to '{corpus_dir}' ({n_text} text / {n_image} image records)."`

`query(corpus_dir, query, top_k=None, kind=None)`: unchanged logic, but reads
`corpus_dir / CORPUS_FILE`; "Corpus not found" message includes the folder path.

### 2.10 Rules for the executing agent

1. Do one task at a time, in order. A task is not done until its tests pass.
2. `corpus_file` must not appear in any function signature, docstring or test.
3. Keep the signatures in §2.2 exactly.
4. Never add optional modes (no `corpus="all"`, no path arguments, no fallback
   when `corpus_name` is empty). A corpus name is always required.
5. Never load a pickle in `list_corpora`; read only `metadata.json`.
6. Tests must not download models, build a real FAISS index, or open real
   PDF/DOCX/XLSX files. Inject the existing seams.
7. Run `python -m pytest tests/ -q` before finishing each task.

---

## 3. Tasks

### T0 — Config: `corpora_dir`

**Goal:** Make the corpora root and default name configurable.

**Current behavior:** `[embedding]` has `corpus_path` and `processed_name`.

**Expected behavior:** `[embedding]` matches §2.4; `corpus_path`,
`default_corpus` and `processed_name` are gone. `config.embedding.corpora_dir`
loads as an absolute `Path`.

**Files:** `config.toml`.

**Tests:** extend `tests/test_common.py`
- `test_load_config_resolves_corpora_dir`: a temp config with
  `[embedding]\ncorpora_dir = ".res/corpora"` → `ns.embedding.corpora_dir == (tmp_path / ".res/corpora").resolve()`.
- `test_real_config_has_corpora_keys`: real `config.toml` has an absolute
  `corpora_dir` and **no** `default_corpus`, `corpus_path` or `processed_name`.

**Verify:** `python -m pytest tests/test_common.py -q`

---

### T1 — Corpus-folder helpers in `src/embedding_search.py`

**Goal:** Add the layout constants, name sanitizer, resolver and metadata I/O.

**Current behavior:** No notion of a corpus folder or metadata.

**Expected behavior:** Implement the constants and functions in §2.2–§2.3:
`sanitize_corpus_name`, `resolve_corpus_dir`, `save_metadata`, `load_metadata`,
`list_corpora`. `resolve_corpus_dir(name, corpora_dir)` returns
`Path(corpora_dir) / sanitize_corpus_name(name)`.

**Files:** `src/embedding_search.py`.

**Tests:** new `tests/test_corpus_registry.py`
- `test_sanitize_accepts_valid_names` (`"autosar"`, `"quora_v2"`, `"a.b-c"`).
- `test_sanitize_rejects_invalid_names` (`""`, `"."`, `".."`, `"a/b"`,
  `"a\\b"`, `"a b"` → `ValueError`).
- `test_resolve_corpus_dir`: name + root → `<root>/<name>`.
- `test_save_and_load_metadata_roundtrip` (tmp dir).
- `test_load_metadata_missing_raises`.
- `test_list_corpora_empty_root_returns_empty` and `test_list_corpora_reads_and_sorts`.

**Verify:** `python -m pytest tests/test_corpus_registry.py -q`

---

### T2 — Folder-based `EmbeddingCorpus.store` / `query`

**Goal:** Store/query a corpus folder with the four fixed artifacts.

**Current behavior:** `store(directory, output_file, overwrite)` writes a pickle
next to `output_file`; `query(corpus_file, ...)`.

**Expected behavior:** Implement §2.9. The folder contains `corpus.pkl`,
`metadata.json`, `processed.md`, `images/`. `metadata.json` has the fields in
§2.3. `processed.md` lists the processed documents' absolute paths.

**Files:** `src/embedding_search.py`.

**Tests:** update `tests/test_embedding_search.py`
- `test_store_creates_corpus_folder`: assert all four artifacts exist and
  `metadata.json` fields match the records/embeddings.
- `test_store_refuses_existing_corpus_without_overwrite`: second call returns
  `"Corpus already exists."` and does not change files.
- `test_store_overwrite_rebuilds_images`: stale `images/junk.png` is removed.
- `test_query_reads_corpus_folder`: query by folder path.
- `test_query_missing_corpus_folder_returns_message`.
- Keep the kind-filter, dedupe and orphan-prune tests, adapted to folders.

**Verify:** `python -m pytest tests/test_embedding_search.py -q`

---

### T3 — `Brain.list_corpora` + new tool signatures

**Goal:** Agent-facing API uses `corpus_name` only and can list corpora.

**Current behavior:** `store_corpus(directory, corpus_file, overwrite)` and
`query_corpus(query, corpus_file, top_k, kind)`.

**Expected behavior:**
1. `Brain.list_corpora()` per §2.7.
2. `Brain.store_corpus(directory, corpus_name, overwrite=False)` per §2.5.
3. `Brain.query_corpus(query, corpus_name, kind="", top_k=0)` per §2.6.
4. `_build_tools()` registers `self.list_corpora` before `self.store_corpus`.

**Files:** `src/live_ai.py`.

**Tests:** update `tests/test_live_ai.py`
- `test_store_corpus_requires_name`: `corpus_name` has **no default** (assert
  via `inspect.signature`), and empty/blank returns `"Error: a corpus name is
  required"` without calling `EmbeddingCorpus.store`.
- `test_query_corpus_requires_name`: same for `query_corpus`.
- `test_store_corpus_uses_named_corpus`.
- `test_query_corpus_resolves_name_and_forwards`: forwards folder path, query,
  `top_k`, `kind` to `EmbeddingCorpus.query` (`EmbeddingCorpus` replaced by a
  `Recorder`).
- `test_store_corpus_invalid_name_returns_error` (e.g. `corpus_name="a/b"`).
- `test_list_corpora_formats_entries` (monkeypatch `live_ai_mod.list_corpora` to
  return two metadata dicts).
- `test_list_corpora_empty_returns_message`.
- `test_agent_build_tools_includes_list_corpora`.
- Existing tests that referenced `corpus_file`/`summary_name` are rewritten to
  the new signatures.

**Verify:** `python -m pytest tests/test_live_ai.py -q`

---

### T4 — Update `samples/inspect_corpus.py`

**Goal:** Inspect a named corpus, not a file path.

**Current behavior:** `inspect_corpus.py` takes a positional pickle path
(default `.res/corpus/corpus.pkl`).

**Expected behavior:**
- Replace the positional path with a required `--corpus <name>` and read
  `<corpora_dir>/<name>/corpus.pkl` (`--list` does not require it).
- Add `--list` to print `list_corpora(corpora_dir)` output.
- Keep the existing read-only filters (`--kind`, `--source`, `--grep`,
  `--record`, `--json`, `--limit`).
- `--help` must not mention `corpus_file` or a raw pickle path.

**Files:** `samples/inspect_corpus.py`.

**Tests:** update `tests/test_inspect_corpus.py`
- `test_main_list_reads_metadata`: with temp corpora folders + `metadata.json`,
  `--list` prints names.
- `test_main_reads_named_corpus`: `--corpus <name> --json` reads
  `<corpora_dir>/<name>/corpus.pkl`.
- Existing helper tests (`as_records`, `summarize`, `select_records`,
  `_checks`) stay.

**Verify:** `python -m pytest tests/test_inspect_corpus.py -q`

---

### T5 — Rebuild corpora and update docs

**Goal:** Move from the old single-corpus layout to named folders and document it.

**Current behavior:** `.res/corpus/corpus.pkl` (single corpus);
`README.md`/`PLAN-EMBEDDING-TOOL.md` describe `corpus_path`/`corpus_file`.

**Expected behavior:**
1. Delete `.res/corpus/` and rebuild the AUTOSAR corpus as
   `.res/corpora/autosar/` using the new folder API
   (`corpus_name="autosar"`, `overwrite=True`).
2. Update the README semantic-search section: folder layout, `corpora_dir`,
   `store_corpus(directory, corpus_name=...)`,
   `query_corpus(query, corpus_name=...)`, `list_corpora()`, and the
   two-layer agent awareness.
3. Add a short "Multi-corpus" note to `PLAN-EMBEDDING-TOOL.md` pointing here and
   mark the single-corpus `corpus_file` API as superseded.

**Files:** `README.md`, `PLAN-EMBEDDING-TOOL.md`, `.res/corpora/` (data).

**Tests:** documentation-only for the prose; the rebuild is validated by
`list_corpora` and `inspect_corpus.py --list`.

**Verify:** `python samples/inspect_corpus.py --list`

---

### T6 — Final validation

**Goal:** Confirm the multi-corpus design is wired end to end.

**Steps:**
1. `python -m pytest tests/ -q` → all pass.
2. Static checks:
   ```bash
   python -c "import src.live_ai as m, inspect; n=[f.__name__ for f in m.Agent._build_tools(m.Agent.__new__(m.Agent))]; print('list_corpora' in n and 'store_corpus' in n and 'query_corpus' in n)"
   python -c "import inspect, src.live_ai as m; print('corpus_file' not in inspect.signature(m.Brain.query_corpus).parameters and 'corpus_file' not in inspect.signature(m.Brain.store_corpus).parameters)"
   ```
   Both must print `True`.
3. Static text check:
   ```bash
   python -c "import pathlib; t=pathlib.Path('src/live_ai.py').read_text(encoding='utf-8'); print('corpus_file' not in t)"
   ```
   Must print `True`.
4. Manual smoke test with a real corpus: `list_corpora()` returns `autosar`, and
   `query_corpus(query="cross cluster communication", corpus_name="autosar")`
   returns hits.

**Verify:** `python -m pytest tests/ -q`

---

## 4. Acceptance checklist

- [ ] Corpora live at `.res/corpora/<corpus_name>/` with `corpus.pkl`,
      `metadata.json`, `processed.md`, `images/`.
- [ ] `store_corpus(directory, corpus_name, overwrite=False)` and
      `query_corpus(query, corpus_name, kind="", top_k=0)` are the only
      agent-facing signatures; `corpus_name` is **required** (no default) and
      `corpus_file` appears nowhere.
- [ ] `corpus_name` is required (no default): empty/blank returns an error, and
      there is no `default_corpus` config key.
- [ ] Invalid names raise/return an error; no path traversal is possible.
- [ ] `list_corpora()` reads only `metadata.json` and lists every corpus.
- [ ] `list_corpora` is registered as a tool and tells the model to use
      `query_corpus(corpus_name=...)`.
- [ ] No system-prompt, `clear_chat()` or `bootstrap()` changes were made for
      corpus awareness.
- [ ] `samples/inspect_corpus.py` uses `--corpus <name>` / `--list`, never a file path.
- [ ] README and `PLAN-EMBEDDING-TOOL.md` reflect the new layout.
- [ ] No optional modes, aliases, fallbacks or legacy paths were added.
- [ ] `python -m pytest tests/ -q` is green.
