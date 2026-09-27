# PLAN-EMBEDDING-TOOL.md — Add corpus build + semantic query tools

This plan adds two related **semantic search function tools** to the live
assistant, derived from `samples/sentence_transformers/semantic_search_quora_faiss.py`:

1. **`store_corpus`** — build a searchable corpus from a set of text files in a
   directory and persist it to a single file.
2. **`query_corpus`** — query the stored corpus with a query text and return the
   most relevant passages.

It follows the same task-guide style as `PLAN.md` and `PLAN-SUMMARY-TOOL.md`
(small, independent tasks; goals / current behavior / expected behavior;
verified by unit tests).

The plan is written for a **small automated coding agent**. Execute the tasks in
order. Each task is one commit and is not done until its tests pass.

---

## 1. Background

### 1.1 Sample algorithm (reference only)

`semantic_search_quora_faiss.py`:

1. Collect a corpus of sentences.
2. Embed them with a `SentenceTransformer` (the sample uses
   `quora-distilbert-multilingual`).
3. L2-normalize the embeddings (so inner product == cosine similarity).
4. Build a FAISS index (`IndexIVFFlat` in the sample; `IndexFlatIP` is the exact
   equivalent without training).
5. For a query: embed + normalize the query, `index.search(query, top_k)`, then
   print `(score, sentence)` hits.

The sample's ANN/IVF tuning (`n_clusters`, `nprobe`, Recall@k comparison) is
**out of scope**. This plan uses an exact `IndexFlatIP` so results are
deterministic; the search backend stays injectable so an IVF index can replace it
later.

### 1.2 Current repository behavior

- `src/summarizer.py` already loads a local `SentenceTransformer` lazily and
  exposes `is_model_available` and `split_sentences`. It is the template to copy
  for lazy loading + test seams.
- `src/live_ai.py::Brain` holds config-driven, stateful tools
  (`image_response`, `summarize_text`, `allow_speech`, ...); `Agent._build_tools`
  registers callables for the LLM.
- `config.toml` has `[summarization].model_path` pointing at
  `.res/embedding/all-MiniLM-L6-v2`; `load_config` resolves any key ending in
  `_path`/`_dir` relative to the config file.
- `setup-lms-cpu.ps1` already installs `sentence-transformers` and `faiss-cpu`.
- `.res/embedding/all-MiniLM-L6-v2` is present; `.res/` is git-ignored.
- No corpus/semantic-search code or config exists yet.

### 1.3 Expected end state

- Two new tools callable by the LLM: `store_corpus(directory, corpus_file, overwrite)`
  and `query_corpus(query, corpus_file, top_k)`.
- The embedding model is **not** loaded at import time; it is loaded only on
  first tool use from a local folder configured in `config.toml`.
- Corpus data is persisted as a single `.pkl` file (passages + normalized
  embeddings).
- No network access and no real model are used by unit tests.
- `README.md` documents how to store the embedding model and how to use the tools.

---

## 2. Design

### 2.1 Architecture

```
src/live_ai.py::Brain.store_corpus / query_corpus   (tools; read config, cache)
        │  delegates to
        ▼
src/embedding_search.py::EmbeddingCorpus            (pipeline + lazy model/index)
        │  uses
        ├── src/summarizer.py::is_model_available, split_sentences
        └── faiss.IndexFlatIP (lazy; injectable backend)
```

Rationale: mirror the summarization design. Keep `src/live_ai.py` thin (it
imports `lmstudio` at module level, so tests must `importorskip` it). All
offline-testable logic lives in `src/embedding_search.py`, which must **not**
import `faiss`, `sentence_transformers` or `torch` at module top level.

### 2.2 Locked public signatures

Implement these exactly; tests and the plan rely on them.

```python
# src/embedding_search.py
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 3
DEFAULT_EXTENSIONS = (".txt", ".md")

def collect_texts(directory, extensions=DEFAULT_EXTENSIONS) -> list[str]: ...
def normalize_rows(matrix) -> "np.ndarray": ...
def build_passages(texts, chunk_size=DEFAULT_CHUNK_SIZE, tokenizer=None) -> list[str]: ...
def load_corpus(path) -> dict: ...
def save_corpus(path, payload: dict) -> None: ...

class EmbeddingCorpus:
    def __init__(
        self,
        model_path,
        device: str = "cpu",
        extensions=DEFAULT_EXTENSIONS,      # tuple of suffixes or a "txt, md" string
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        top_k: int = DEFAULT_TOP_K,
        model_factory=None,                 # test seam: factory(model_path, device) -> embedder
        index_factory=None,                 # test seam: factory(embeddings) -> index with .search()
        sentence_tokenizer=None,            # test seam: tokenizer(text) -> list[str]
    ): ...
    def store(self, directory: str, output_file: str, overwrite: bool = False) -> str: ...
    def query(self, corpus_file: str, query: str, top_k: int | None = None) -> str: ...

# src/live_ai.py
class Brain:
    @dump_func_call
    def store_corpus(self, directory: str, corpus_file: str = "", overwrite: bool = False) -> str: ...

    @dump_func_call
    def query_corpus(self, query: str, corpus_file: str = "", top_k: int = 0) -> str: ...
```

`model_factory`, `index_factory` and `sentence_tokenizer` default to `None`; when
`None` the class uses the real lazy loaders. Tests always inject fakes.

Reuse `is_model_available` and `split_sentences` from `src/summarizer` instead of
duplicating them.

### 2.3 Config schema (new `[embedding]` block)

```toml
[embedding] # semantic search over a local corpus (sentence embeddings + FAISS)
model_path = ".res/embedding/all-MiniLM-L6-v2"  # resolved relative to this config file
device = "cpu"            # "cpu" or "cuda"
corpus_path = ".res/corpus/corpus.pkl"  # default corpus file, resolved relative to config.toml
extensions = "txt, md"    # comma-separated extensions collected from a directory
chunk_size = 3            # sentences per indexed passage
top_k = 5                 # default number of results returned by query_corpus
```

`model_path` and `corpus_path` end in `_path`, so `load_config` resolves them to
absolute paths.

### 2.4 Corpus file format (single pickle)

```python
{
    "version": 1,
    "model_path": str,           # informational; used to warn on mismatch
    "passages": list[str],       # indexed passages, in file order
    "embeddings": np.ndarray,    # float32, L2-normalized, shape (N, dim)
}
```

`save_corpus` creates parent directories and writes the pickle. `load_corpus`
raises `ValueError` when a required key is missing.

### 2.5 `EmbeddingCorpus.store` algorithm

1. `output = Path(output_file).resolve()`. If it exists and `overwrite` is
   `False` → return `"File already exists."` (do not modify it).
2. `texts = collect_texts(directory, self.extensions)`. If empty → return
   `f"No text files found in '{directory}'"`.
3. `passages = build_passages(texts, self.chunk_size, self._tokenizer)`; drop
   empty passages. If empty → return `"No passages extracted from the input files."`.
4. `embeddings = normalize_rows(self._encode(passages))`.
5. `save_corpus(output, {...})`.
6. Return `f"Corpus with {len(passages)} passages saved to '{output}'"`.

### 2.6 `EmbeddingCorpus.query` algorithm

1. `query = (query or "").strip()`. If empty → `"No query provided."`.
2. `path = Path(corpus_file).resolve()`. If missing → return
   `f"Corpus not found at '{path}'. Build it first with store_corpus."`.
3. `payload = load_corpus(path)`; `passages = payload["passages"]`;
   `embeddings = np.asarray(payload["embeddings"], dtype=float)`. If
   `len(passages) == 0` → `"Corpus is empty."`.
4. `query_vec = self._encode([query])[0]`. If its length differs from
   `embeddings.shape[1]` → return a mismatch message mentioning the configured
   model (`"Query embedding size does not match the corpus; rebuild the corpus."`).
5. `index = (self._index_factory or _default_index_factory)(embeddings)`.
6. `k = top_k or self.top_k`; clamp to `1..len(passages)`.
7. `scores, ids = index.search(query_vec[None, :], k)`.
8. Build lines from valid ids (`id >= 0`), format
   `f"{rank}) [{score:.3f}] {passages[i]}"`, and return them joined by `"\n"`,
   prefixed with `f"Top {k} matches for '{query}':"`.

`_default_index_factory(embeddings)`:

```python
def _default_index_factory(embeddings):
    import faiss
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    return index
```

`_encode(texts)` calls the lazily loaded embedder:
`np.asarray(embedder.encode(list(texts), convert_to_numpy=True), dtype=float)`.
`_embedder()` follows `TextSummarizer._embedder()`: use `model_factory` when
provided, otherwise check `is_model_available` and raise `FileNotFoundError`
mentioning `README.md`; import `SentenceTransformer` lazily. The model is cached
on the instance so `store` + `query` load it once.

### 2.7 `Brain` tool behavior

- Add `from src.embedding_search import EmbeddingCorpus` to `src/live_ai.py`.
- Add a private `Brain._embedding_corpus()` helper that lazily creates and caches
  `self._emb_corpus = EmbeddingCorpus(...)` from `self.config.embedding`, using
  `getattr(cfg, "key", default)` for optional keys.
- `store_corpus(directory, corpus_file="", overwrite=False)`:
  - `output = corpus_file or str(self.config.embedding.corpus_path)`
  - delegate to `self._embedding_corpus().store(directory, output, overwrite)`
- `query_corpus(query, corpus_file="", top_k=0)`:
  - `target = corpus_file or str(self.config.embedding.corpus_path)`
  - delegate to `self._embedding_corpus().query(target, query, top_k or None)`
- Both tools wrap the body in `try/except Exception` and return `f"Error: {e}"`.
- Register both in `Agent._build_tools()` right after `self.summarize_text`.

### 2.8 Rules for the executing agent

1. Do one task at a time, in order. A task is not done until its tests pass.
2. Never download or load a real model, and never build a real FAISS index, in
   tests. Inject `model_factory`, `index_factory` and `sentence_tokenizer` fakes.
3. Never import `faiss`, `sentence_transformers` or `torch` at the top level of
   `src/embedding_search.py`; import them lazily inside functions. `numpy` is fine.
4. Keep the signatures in §2.2 exactly.
5. Run `python -m pytest tests/ -q` from the repo root before finishing a task.

---

## 3. Tasks

### T0 — Add the `[embedding]` config section

**Goal:** Make the model folder, default corpus file and search knobs configurable.

**Current behavior:** `config.toml` has `[summarization]` but no embedding/search
settings; the sample hardcodes the model and index parameters.

**Expected behavior:** `config.toml` contains the block from §2.3. Loading it with
`src.common.load_config` returns `config.embedding.model_path` and
`config.embedding.corpus_path` as absolute `Path`s, and
`device`/`extensions`/`chunk_size`/`top_k` as plain values.

**Files:**
- Edit `config.toml` (append the `[embedding]` block after `[summarization]`).

**Tests:** extend `tests/test_common.py`
- `test_load_config_resolves_embedding_paths`: a temp config with
  `[embedding]\nmodel_path = ".res/models/st"\ncorpus_path = ".res/corpus/c.pkl"`
  → both attributes equal the resolved `(tmp_path / ...)` paths.
- `test_real_config_has_embedding_block`: load the repo `config.toml` and assert
  `model_path`/`corpus_path` are absolute and `extensions`, `chunk_size`, `top_k`
  are present.

**Verify:** `python -m pytest tests/test_common.py -q`

---

### T1 — Implement `src/embedding_search.py`

**Goal:** Implement the offline-testable corpus build + query pipeline.

**Current behavior:** No embedding-search module exists.

**Expected behavior:** `src/embedding_search.py` implements exactly the signatures
and algorithms in §2.2/§2.4–§2.6. Importing it must not import `faiss`,
`sentence_transformers` or `torch`.

Implementation notes:
- `collect_texts` walks the directory recursively with `Path.rglob("*")`,
  includes files whose lowercase suffix is in the normalized extensions, reads
  UTF-8, skips unreadable/empty files, and returns texts sorted by path for
  determinism. Raises `FileNotFoundError` when the directory is missing and
  `NotADirectoryError` when the path is not a directory.
- `normalize_rows` handles 1-D input and zero-norm rows (guard with `1.0`).
- `build_passages` calls `split_sentences` (or the injected tokenizer) per text,
  then groups `chunk_size` consecutive sentences into one space-joined passage.
- Normalize `extensions`: accept a `"txt, md"` string or a tuple of suffixes.

**Files:**
- Add `src/embedding_search.py`.

**Tests:** add `tests/test_embedding_search.py` (see T2 for the full list). At
minimum implement the module so all T2 tests pass.

**Verify:** `python -m pytest tests/test_embedding_search.py -q`

---

### T2 — Unit tests for `src/embedding_search.py`

**Goal:** Prove corpus building and querying work without a real model or FAISS.

**Current behavior:** No tests for embedding search.

**Expected behavior:** `tests/test_embedding_search.py` passes with injected fakes.
Shared fakes (define in the test module):

```python
class FakeEmbedder:
    """Deterministic text -> vector mapping; falls back to one-hot."""
    def __init__(self, mapping=None, dim=4): ...
    def encode(self, texts, convert_to_numpy=True): ...

class FakeIndex:
    """Exact inner-product search, mimicking faiss.IndexFlatIP.search()."""
    def __init__(self, embeddings): self.embeddings = embeddings
    def search(self, query_vec, k): ...   # -> (scores (1×k), ids (1×k))
```

Recommended test list:
- `test_import_is_lazy`: pop `faiss`, `sentence_transformers`, `torch` from
  `sys.modules`, import `src.embedding_search`, assert none are imported.
- `test_normalize_rows_unit_norm_and_zero_safe`.
- `test_collect_texts_reads_supported_files`: `a.txt`/`b.md` collected, `c.bin`
  ignored, order deterministic.
- `test_collect_texts_missing_dir_raises` / `test_collect_texts_not_a_dir_raises`.
- `test_build_passages_chunks`: injected tokenizer + `chunk_size=2` groups
  sentences correctly; `chunk_size=1` yields one passage per sentence.
- `test_store_writes_pickle`: fake embedder; assert the output exists,
  `load_corpus` returns the right passages and normalized embeddings, and the
  status string contains the passage count.
- `test_store_refuses_existing_by_default`: existing output + `overwrite=False`
  → `"File already exists."` and file unchanged.
- `test_store_overwrite_true`.
- `test_store_no_text_files_returns_message`.
- `test_store_missing_model_raises_friendly_error`: no `model_factory`,
  non-existent `model_path`, valid input dir → `FileNotFoundError` mentioning the path.
- `test_query_returns_ranked_hits`: build a corpus, then `query` with a fake index
  → hits ordered by score and formatted (`"1) [0.9]"`-style).
- `test_query_missing_corpus_returns_message`.
- `test_query_empty_query_returns_message`.
- `test_query_dimension_mismatch_returns_message`: stored dim ≠ query dim.
- `test_model_loaded_once_across_store_and_query`: factory counter == 1.
- `test_query_uses_instance_top_k_when_not_overridden`.

**Files:**
- Add `tests/test_embedding_search.py`.

**Verify:** `python -m pytest tests/test_embedding_search.py -q`

---

### T3 — Wire the tools into `src/live_ai.py`

**Goal:** Expose `store_corpus` and `query_corpus` to the LLM.

**Current behavior:** `Brain` has no corpus methods and `Agent._build_tools`
does not register them.

**Expected behavior:**
1. Import `EmbeddingCorpus` at the top of `src/live_ai.py`.
2. Add `Brain._embedding_corpus()`, `Brain.store_corpus` and `Brain.query_corpus`
   exactly as specified in §2.7, both decorated with `@dump_func_call`.
3. Add `self.store_corpus` and `self.query_corpus` to `Agent._build_tools()`
   immediately after `self.summarize_text`.

```python
def _embedding_corpus(self):
    if getattr(self, "_emb_corpus", None) is None:
        cfg = self.config.embedding
        self._emb_corpus = EmbeddingCorpus(
            model_path=cfg.model_path,
            device=getattr(cfg, "device", "cpu"),
            extensions=getattr(cfg, "extensions", DEFAULT_EXTENSIONS),
            chunk_size=getattr(cfg, "chunk_size", DEFAULT_CHUNK_SIZE),
            top_k=getattr(cfg, "top_k", DEFAULT_TOP_K),
        )
    return self._emb_corpus
```

(`DEFAULT_EXTENSIONS`/`DEFAULT_CHUNK_SIZE`/`DEFAULT_TOP_K` should be imported
from `src.embedding_search` or replaced with literal defaults.)

**Files:**
- Edit `src/live_ai.py`.

**Tests:** extend `tests/test_live_ai.py` (see T4).

**Verify:** `python -m pytest tests/test_live_ai.py -q`

---

### T4 — Unit tests for the live_ai integration

**Goal:** Prove both tools read config, cache the `EmbeddingCorpus`, forward
arguments, and that `_build_tools` registers them.

**Current behavior:** `tests/test_live_ai.py` covers `allow_speech` and
`summarize_text` only.

**Expected behavior:** New tests in `tests/test_live_ai.py`. Add a bare-brain
helper:

```python
def _bare_embedding_brain(**overrides):
    cfg = dict(model_path="m", device="cpu", extensions="txt, md",
               chunk_size=3, top_k=5, corpus_path="corpus.pkl")
    cfg.update(overrides)
    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(embedding=SimpleNamespace(**cfg))
    return brain
```

Recommended tests:
- `test_store_corpus_uses_config_and_caches`: monkeypatch
  `live_ai_mod.EmbeddingCorpus` with a `Recorder` that stores `__init__` kwargs
  and implements `store`/`query`. Call `store_corpus(str(dir))` twice; assert the
  corpus object is created once, kwargs match the config, and the configured
  `corpus_path` is used as the output when `corpus_file=""`.
- `test_store_corpus_forwards_explicit_file_and_overwrite`.
- `test_query_corpus_forwards_query_and_top_k`: assert `query_corpus("hello", top_k=2)`
  forwards `("corpus.pkl", "hello", 2)` (with the configured default file).
- `test_query_corpus_uses_explicit_corpus_file`.
- `test_store_corpus_reports_errors` / `test_query_corpus_reports_errors`: a
  `Recorder` that raises `RuntimeError("boom")` → result starts with `"Error:"`
  and contains `"boom"`.
- `test_agent_build_tools_includes_embedding_tools`: `_build_tools()` callable
  names include `"store_corpus"` and `"query_corpus"`.
- `test_agent_build_tools_keeps_summarize_text`: `"summarize_text"` is still present.

Because `src/live_ai.py` imports `lmstudio`, keep the existing
`pytest.importorskip("lmstudio", ...)` at the top of the test module.

**Files:**
- Edit `tests/test_live_ai.py`.

**Verify:** `python -m pytest tests/test_live_ai.py -q`

---

### T5 — Document model storage and usage in `README.md`

**Goal:** Tell the user how to store the embedding model and how to use the tools.

**Current behavior:** `README.md` documents summarization but nothing about
corpus/semantic search.

**Expected behavior:** Add a `## Semantic search (corpus + FAISS)` section that:

1. States that `store_corpus` indexes all `.txt`/`.md` files in a directory and
   `query_corpus` returns the most relevant passages, using `all-MiniLM-L6-v2`
   (sentence embeddings) + `faiss.IndexFlatIP` (cosine similarity). Note that the
   multilingual sample model `sentence-transformers/quora-distilbert-multilingual`
   can be used instead by changing `model_path`.
2. Shows the `[embedding]` block from §2.3 and explains that `model_path` and
   `corpus_path` are resolved relative to `config.toml`.
3. Gives three ways to store the model in `.res/embedding/all-MiniLM-L6-v2`
   (Hugging Face CLI `hf download`, `git lfs` clone, Python `.save()`), mirroring
   the summarization section, plus the multilingual alternative.
4. Notes the tool usage: `store_corpus(directory="docs")` writes the configured
   corpus file, and `query_corpus(query="...")` searches it; tune `chunk_size`,
   `top_k`, and pass `corpus_file` to use a different corpus.
5. Notes that `faiss-cpu` must be installed (already in `setup-lms-cpu.ps1`) and
   that `.res/` is git-ignored.

**Files:**
- Edit `README.md` (insert after the summarization section).

**Tests:** documentation only; no unit test. Manual check:
`grep -n "embedding\|Semantic search" README.md config.toml`.

**Verify:** `grep -n "Semantic search\|\[embedding\]" README.md config.toml`

---

### T6 — Final validation (integration gate)

**Goal:** Confirm the whole feature is wired and nothing regressed.

**Current behavior:** N/A.

**Expected behavior:** The full suite passes and both tools are discoverable.

**Steps:**
1. Run the full suite: `python -m pytest tests/ -q` → all tests pass.
2. Static check that the tools are registered:

   ```bash
   python -c "import src.live_ai as m; names=[f.__name__ for f in m.Agent._build_tools(m.Agent.__new__(m.Agent))]; print('store_corpus' in names and 'query_corpus' in names)"
   ```

   Must print `True`.
3. Manual smoke test with real models (optional, not part of CI):

   ```bash
   python main.py --chat --prompt "Store a corpus from the directory docs"
   python main.py --chat --prompt "Query the corpus: what is quantum computing?"
   ```

   If `.res/embedding/all-MiniLM-L6-v2` is absent, both tools must return a
   friendly `Error:` string instead of crashing.

**Files:** none (verification only).

**Verify:** `python -m pytest tests/ -q`

---

## 4. Acceptance checklist

- [ ] `config.toml` has an `[embedding]` block with resolvable `model_path` and
      `corpus_path`.
- [ ] `src/embedding_search.py` imports no `faiss`/`sentence_transformers`/`torch`
      at module load and follows the §2.2 signatures.
- [ ] The corpus file is a single pickle containing `passages` + normalized
      `embeddings`.
- [ ] `Brain.store_corpus` builds a corpus from a directory and writes it to the
      configured (or explicit) file; it refuses to overwrite unless asked.
- [ ] `Brain.query_corpus` loads the corpus, embeds the query, searches with
      FAISS (or the injected backend), and returns ranked passages.
- [ ] Both tools cache the `EmbeddingCorpus` on the `Brain` and never raise to
      the caller.
- [ ] `Agent._build_tools()` includes `store_corpus` and `query_corpus`.
- [ ] `tests/test_embedding_search.py` and the new `tests/test_live_ai.py` tests
      pass with no model download, no network and no real FAISS index.
- [ ] `README.md` documents how to store the embedding model and use the tools.
- [ ] `python -m pytest tests/ -q` is green.
