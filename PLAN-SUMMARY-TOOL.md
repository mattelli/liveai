# PLAN-SUMMARY-TOOL.md — Add an extractive `summarize_text` tool

This plan adds an **extractive text-summarization function tool** to the live
assistant. It is derived from `samples/sentence_transformers/text_summarization.py`
(LexRank + `sentence-transformers` sentence embeddings).

The plan is written for a **small automated coding agent**. Each task is
self-contained, states its goal, the current behavior, the expected behavior,
the files to touch, and the unit tests that prove it is done. Execute the tasks
in order. Each task is one commit.

---

## 1. Background

### 1.1 Sample algorithm (reference only)

`text_summarization.py` does the following:

1. Split a long document into sentences with NLTK (`nltk.sent_tokenize`).
2. Encode every sentence with `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`.
3. Compute pairwise cosine similarity across sentences.
4. Run LexRank (`degree_centrality_scores`, vendored in `samples/sentence_transformers/lex_rank.py`).
5. Keep the most central sentences (`n // 10`, random threshold).

### 1.2 Current repository behavior

- `src/tools.py` holds stateless tools (file/web/finance/camera). None of them
  needs a config or a loaded ML model.
- `src/live_ai.py::Brain` holds **stateful** tools that read `config`
  (`image_response`, `allow_speech`, `store_personal`, ...).
- `src/live_ai.py::Agent` registers the callable tools in `self.tools`, which is
  passed to `self._model.act(...)`.
- `config.toml` already has `[speech_det].model_path` and `[speech_gen].model_path`.
  `src/common.py::load_config` resolves any key ending in `_path` relative to the
  config file directory, so a new `[summarization].model_path` will be returned as
  an absolute `Path`.
- `tests/` uses pytest with fakes in `tests/conftest.py`; heavy modules are
  imported lazily and never loaded in tests.
- `setup-lms-cpu.ps1` already installs `sentence-transformers`, `nltk` and
  `faiss-cpu`; `scipy` and `torch` are available transitively.

### 1.3 Expected end state

- A new tool `summarize_text(text, max_sentences=0)` is callable by the LLM agent.
- The sentence-embedding model is **not** downloaded at import time and is loaded
  only on first tool use, from a local folder configured in `config.toml`.
- No network access and no real model are used by unit tests.
- `README.md` documents how to place the model in that folder.

---

## 2. Design

### 2.1 Architecture

```
src/live_ai.py::Brain.summarize_text   (tool exposed to the LLM, reads config)
        │  delegates to
        ▼
src/summarizer.py::TextSummarizer      (pure pipeline + lazy model loading)
        │  uses
        ├── src/summarizer.py::split_sentences   (NLTK, regex fallback)
        └── src/lex_rank.py::degree_centrality_scores
```

Rationale: keep `src/live_ai.py` thin (it imports `lmstudio` at module level, so
tests must `importorskip` it). All offline-testable logic lives in
`src/summarizer.py`, which must **not** import `sentence_transformers`, `nltk`,
`torch` or `scipy` at module top level.

### 2.2 Locked public signatures

Implement these exactly; tests and the plan rely on them.

```python
# src/lex_rank.py
def degree_centrality_scores(similarity_matrix, threshold=None, increase_power=True) -> "np.ndarray": ...

# src/summarizer.py
DEFAULT_RATIO = 0.1
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_SENTENCES = 0          # 0 -> use ratio
MAX_INPUT_CHARS = 20000             # guard against pathological documents

def split_sentences(text: str) -> list[str]: ...
def is_model_available(model_path) -> bool: ...

class TextSummarizer:
    def __init__(
        self,
        model_path,
        device: str = "cpu",
        ratio: float = DEFAULT_RATIO,
        threshold: float = DEFAULT_THRESHOLD,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        model_factory=None,          # test seam: factory(model_path, device) -> embedder
        sentence_tokenizer=None,     # test seam: tokenizer(text) -> list[str]
    ): ...
    def summarize(self, text: str, max_sentences: int | None = None) -> str: ...

# src/live_ai.py
class Brain:
    @dump_func_call
    def summarize_text(self, text: str, max_sentences: int = 0) -> str: ...
```

`model_factory` and `sentence_tokenizer` default to `None`; when `None` the class
uses the real lazy loaders. Tests always inject fakes.

### 2.3 Config schema (new `[summarization]` block)

```toml
[summarization]
model_path = ".res/sentence-transformers/all-MiniLM-L6-v2"
device = "cpu"            # "cpu" or "cuda"; keeps tests offline
ratio = 0.1               # fraction of sentences to keep (0 < ratio <= 1)
threshold = 0.5           # LexRank similarity threshold in [0, 1)
max_sentences = 0         # 0 -> derive from ratio; >0 -> hard cap
```

`model_path` is resolved to an absolute path by `load_config` because it ends in
`_path`.

### 2.4 `TextSummarizer.summarize` algorithm

1. `text = (text or "").strip()`. If empty → return `"No text to summarize."`.
2. `sentences = self._tokenize(text)`. If `len(sentences) <= 1` → return the
   original stripped `text`.
3. If `len(text) > MAX_INPUT_CHARS`, truncate to the first `MAX_INPUT_CHARS`
   characters and re-tokenize (summary computed on the truncated text).
4. `k` = `max_sentences` argument if `> 0`, else `self.max_sentences` if `> 0`,
   else `max(1, round(len(sentences) * self.ratio))`; clamp to `len(sentences)`.
5. Embed sentences: `emb = np.asarray(embedder.encode(sentences), dtype=float)`.
   If `emb` is 1-D (single sentence), reshape to `(1, -1)`.
6. Normalize rows (guard zero norms) and compute `sim = emb @ emb.T`.
7. `scores = degree_centrality_scores(sim, self.threshold)`.
8. `top = sorted(np.argsort(-scores)[:k])` → **re-sort by original position** so
   the summary reads in document order.
9. Return `" ".join(sentences[i] for i in top).strip()`.

Model loading (`_embedder()`):
- If `self._model is None`:
  - if `model_factory` was provided → `self._model = model_factory(str(self.model_path), self.device)`
  - else:
    - if `not is_model_available(self.model_path)` → raise `FileNotFoundError(...)`
    - lazy `from sentence_transformers import SentenceTransformer`
    - `self._model = SentenceTransformer(str(self.model_path), device=self.device)`
- Cache and return `self._model`.

`is_model_available(model_path)` returns `True` if the path is a directory
containing either `modules.json` or `config_sentence_transformers.json` or
`pytorch_model.bin`/`model.safetensors`; otherwise `False`.

`split_sentences(text)`:
- Try `import nltk; return nltk.sent_tokenize(text)`.
- On `ImportError`/`LookupError` (missing `punkt`/`punkt_tab`) fall back to a
  regex splitter: split on `(?<=[.!?])\s+`, strip empties.

### 2.5 `Brain.summarize_text` behavior

```python
@dump_func_call
def summarize_text(self, text: str, max_sentences: int = 0) -> str:
    """
    Summarize a long text into its most important sentences (extractive summary).
    Useful when the user asks to summarize, condense, shorten or outline a text.
    ``max_sentences`` = 0 uses the ratio configured in the ``[summarization]`` section.
    """
```

- Lazily build `self._summarizer = TextSummarizer(...)` from
  `self.config.summarization` using `getattr(cfg, "key", default)` for every
  optional key, then cache it.
- If `max_sentences > 0`, override the instance default for this call.
- Catch `Exception` and return `f"Error: {e}"` (consistent with other tools) so
  the agent reports a friendly message instead of crashing.

### 2.6 Rules for the executing agent

1. Do one task at a time, in order. A task is not done until its tests pass.
2. Never download or load a real model in tests. Always inject `model_factory`
   and `sentence_tokenizer` fakes.
3. Never import `sentence_transformers`, `nltk`, `torch` or `scipy` at the top
   level of `src/summarizer.py`; import them lazily inside functions.
   `src/lex_rank.py` may import `numpy` and `scipy` (already installed) at top level.
4. Keep the signatures in §2.2 exactly.
5. Run `python -m pytest tests/ -q` from the repo root before finishing a task.

---

## 3. Tasks

### T0 — Vendor the LexRank implementation

**Goal:** Provide `degree_centrality_scores` inside the `src` package.

**Current behavior:** The only LexRank implementation lives in
`samples/sentence_transformers/lex_rank.py` and is not importable as part of the
package.

**Expected behavior:** `src/lex_rank.py` exists and exports
`degree_centrality_scores(similarity_matrix, threshold=None, increase_power=True)`.
It is a copy of the sample module with `__all__ = ["degree_centrality_scores"]`.
`tests/test_tools.py::test_import_tools_is_lazy`-style constraints do not apply to
it (numpy/scipy are cheap enough and already required).

**Files:**
- Add `src/lex_rank.py` (copy `samples/sentence_transformers/lex_rank.py`, add `__all__`).

**Tests:** `tests/test_lex_rank.py`
- `test_degree_centrality_scores_returns_vector`: 3×3 identity-like similarity →
  returns a `numpy.ndarray` of length 3.
- `test_degree_centrality_scores_rejects_bad_threshold`: `threshold=1.5` raises
  `ValueError`; `threshold=None` works.
- `test_degree_centrality_scores_prefers_central_node`: a 3-node matrix where node
  1 is connected to both 0 and 2 has the highest score for that node.

**Verify:** `python -m pytest tests/test_lex_rank.py -q`

---

### T1 — Add the `[summarization]` config section

**Goal:** Make the model folder and synthesis knobs configurable.

**Current behavior:** `config.toml` has no summarization settings; the sample
hardcodes `sentence-transformers/all-MiniLM-L6-v2` and a random threshold.

**Expected behavior:** `config.toml` contains the block from §2.3. Loading it with
`src.common.load_config` returns `config.summarization.model_path` as an absolute
`Path`, `device`/`ratio`/`threshold`/`max_sentences` as plain values, and does not
raise.

**Files:**
- Edit `config.toml` (append the `[summarization]` block).

**Tests:** extend `tests/test_common.py`
- `test_load_config_resolves_summarization_model_path`: write a temp config with
  `[summarization]\nmodel_path = ".res/models/st"` → assert
  `ns.summarization.model_path == (tmp_path / ".res/models/st").resolve()`.
- `test_real_config_has_summarization_block`: load the repo `config.toml` and
  assert `ns.summarization` has `model_path`, `device`, `ratio`, `threshold`,
  `max_sentences`.

**Verify:** `python -m pytest tests/test_common.py -q`

---

### T2 — Implement `src/summarizer.py`

**Goal:** Implement the offline-testable summarization pipeline.

**Current behavior:** No summarization module exists.

**Expected behavior:** `src/summarizer.py` implements exactly the signatures and
algorithm in §2.2/§2.4. Importing it must not import
`sentence_transformers`, `nltk`, `torch` or `scipy`.

**Files:**
- Add `src/summarizer.py`.

**Tests:** add `tests/test_summarizer.py` (see T4 for the full list). At minimum
implement the module so all T4 tests pass.

**Verify:** `python -m pytest tests/test_summarizer.py -q`

---

### T3 — Wire the tool into `src/live_ai.py`

**Goal:** Expose `summarize_text` to the LLM and register it in the agent's tool list.

**Current behavior:** `Brain` has no summarization method and `Agent.__init__`
builds `self.tools` inline without a summarization entry.

**Expected behavior:**
1. `Brain.summarize_text(text, max_sentences=0)` exists per §2.5, lazily creates
   and caches `self._summarizer`, and is decorated with `@dump_func_call`.
2. Import `TextSummarizer` from `src.summarizer` at the top of `src/live_ai.py`.
3. Extract the tool list into a method so it can be unit-tested without loading a
   real LLM:

```python
class Agent(Brain):
    def _build_tools(self) -> list[Callable]:
        return [
            self.model_info, self.recall_personal, self.store_personal,
            self.image_response, self.allow_speech, self.clear_chat,
            self.summarize_text,                        # <-- new
            capture_image, system_info, current_date_time, get_working_directory,
            run_command, search_in_file, search_pattern_in_files,
            read_text_file, replace_string_in_file, write_text_file,
            web_search, webpage_content, webpage_headlines,
            fetch_financial_data, save_financial_chart,
            exit_gracefully,
        ]

    def __init__(self, config):
        super().__init__(config)
        ...
        self.tools: list[Callable] = self._build_tools()
```

**Files:**
- Edit `src/live_ai.py` (import, new `Brain` method, `_build_tools` refactor).

**Tests:** extend `tests/test_live_ai.py` (see T5).

**Verify:** `python -m pytest tests/test_live_ai.py -q`

---

### T4 — Unit tests for the summarizer

**Goal:** Prove the pipeline behavior without any real model or network.

**Current behavior:** No tests for summarization.

**Expected behavior:** `tests/test_summarizer.py` passes with injected fakes.
Define shared fakes in the test module:

```python
class FakeEmbedder:
    """Deterministic bag-of-keywords embeddings."""
    def encode(self, sentences): ...   # -> numpy array, one row per sentence

class FakeSummarizer(TextSummarizer):
    ...  # or pass model_factory=..., sentence_tokenizer=...
```

Recommended test list:

- `test_import_summarizer_is_lazy`: remove `sentence_transformers`, `nltk`,
  `torch`, `scipy` from `sys.modules`, import `src.summarizer`, assert none of
  them are imported by that module. (`scipy` may appear via `src.lex_rank`; assert
  only `sentence_transformers`, `nltk`, `torch` are absent.)
- `test_split_sentences_with_injected_tokenizer`: injected tokenizer returns
  `["a", "b"]`; assert passthrough.
- `test_split_sentences_regex_fallback`: force the NLTK path to fail (inject /
  monkeypatch so `nltk.sent_tokenize` raises `LookupError`) and assert the regex
  splitter splits `"One. Two! Three?"` into 3 sentences.
- `test_summarize_empty_returns_message`: `""` and `"   "` → `"No text to summarize."`.
- `test_summarize_single_sentence_returns_input`: one sentence → unchanged.
- `test_summarize_respects_ratio`: 10 sentences with a fake tokenizer and a fake
  embedder whose central sentence is index 5; `ratio=0.1` → summary contains
  sentence 5 and exactly 1 sentence.
- `test_summarize_respects_max_sentences_override`: `max_sentences=2` → at most 2
  sentences.
- `test_summarize_preserves_document_order`: chosen indices are returned in
  original order (assert positions of the chosen sentences are ascending).
- `test_summarize_truncates_long_input`: text longer than `MAX_INPUT_CHARS` is
  truncated; the injected tokenizer records the truncated string length.
- `test_missing_model_raises_friendly_error`: no `model_factory`, non-existent
  `model_path` → `summarize` raises `FileNotFoundError` mentioning the path.
- `test_model_is_loaded_once`: injected factory increments a counter; two
  `summarize` calls → counter == 1.

**Files:**
- Add `tests/test_summarizer.py`.

**Verify:** `python -m pytest tests/test_summarizer.py -q`

---

### T5 — Unit tests for the live_ai integration

**Goal:** Prove `Brain.summarize_text` reads config, caches the summarizer and
that `Agent._build_tools` registers the new tool.

**Current behavior:** `tests/test_live_ai.py` only covers `allow_speech`.

**Expected behavior:** New tests in `tests/test_live_ai.py`:

- `test_summarize_text_uses_config_and_caches`: monkeypatch
  `live_ai_mod.TextSummarizer` with a `Recorder` class that stores `__init__`
  kwargs and returns a fixed string from `summarize`. Build a bare brain:

  ```python
  brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
  brain.logger = getLogger("test")
  brain.config = SimpleNamespace(summarization=SimpleNamespace(
      model_path="m", device="cpu", ratio=0.2, threshold=0.4, max_sentences=3))
  ```

  Call `brain.summarize_text("long text")` twice. Assert: factory created once,
  kwargs match the config, `summarize` called twice, result is the fixed string.
- `test_summarize_text_forwards_max_sentences`: `brain.summarize_text("t", max_sentences=2)`
  forwards `2` to `Recorder.summarize`.
- `test_summarize_text_reports_errors`: a `Recorder` whose `summarize` raises
  `RuntimeError("boom")` → result starts with `"Error:"` and contains `"boom"`.
- `test_agent_build_tools_includes_summarize_text`:
  `Agent.__new__(Agent)._build_tools()` returns a list whose callable names
  include `"summarize_text"` (no `__init__`, so no LLM is loaded).
- `test_agent_build_tools_keeps_existing_tools`: assert `"system_info"`,
  `"read_text_file"`, `"web_search"` are still present.

Because `src/live_ai.py` imports `lmstudio`, the module already starts with
`pytest.importorskip("lmstudio", ...)`; keep that.

**Files:**
- Edit `tests/test_live_ai.py`.

**Verify:** `python -m pytest tests/test_live_ai.py -q`

---

### T6 — Document model storage in `README.md`

**Goal:** Tell the user how to place the summarization model in the configured folder.

**Current behavior:** `README.md` documents Whisper, LM Studio, Piper and
Diffusers, but nothing about sentence-transformers / summarization.

**Expected behavior:** Add a `## Text summarization` section that:

1. States that `summarize_text` uses `all-MiniLM-L6-v2` + LexRank and loads the
   model lazily on first use.
2. Shows the `[summarization]` block from §2.3 and explains that `model_path` is
   resolved relative to `config.toml`.
3. Gives three ways to store the model in `.res/sentence-transformers/all-MiniLM-L6-v2`:

   ```bash
   # A) Hugging Face CLI (>= 0.23)
   pip install -U huggingface_hub
   hf download sentence-transformers/all-MiniLM-L6-v2 \
       --local-dir .res/sentence-transformers/all-MiniLM-L6-v2

   # B) git + git-lfs
   git lfs install
   git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 \
       .res/sentence-transformers/all-MiniLM-L6-v2

   # C) Python (uses the current HF cache, then saves a local copy)
   python -c "from sentence_transformers import SentenceTransformer; \
   SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/sentence-transformers/all-MiniLM-L6-v2')"
   ```

4. Notes that the folder must contain `modules.json` / `config_sentence_transformers.json`
   and the model weights, that `.res/` is git-ignored, and that NLTK `punkt_tab`
   is used for sentence splitting with a regex fallback when absent.
5. Mentions tuning knobs (`ratio`, `max_sentences`, `threshold`, `device`).

**Files:**
- Edit `README.md` (insert the section after `## Diffusers` or near the other
  model-download sections).

**Tests:** documentation only; no unit test. Manual check: the commands and the
TOML block are valid and consistent with `config.toml` from T1.

**Verify:** `grep -n "summarization" README.md config.toml`

---

### T7 — Final validation (integration gate)

**Goal:** Confirm the whole feature is wired and nothing regressed.

**Current behavior:** N/A.

**Expected behavior:** The full suite passes and the tool is discoverable.

**Steps:**
1. Run the full suite: `python -m pytest tests/ -q` → all tests pass.
2. Static check that the tool is registered and the model is only loaded lazily:

   ```bash
   python -c "import src.live_ai as m; print('summarize_text' in [f.__name__ for f in m.Agent._build_tools(m.Agent.__new__(m.Agent))])"
   ```

   Must print `True`.
3. Manually smoke-test with a real model (optional, not part of CI):

   ```bash
   python main.py --chat --prompt "Summarize: <paste a long text>"
   ```

   If `.res/sentence-transformers/all-MiniLM-L6-v2` is absent, the agent must
   return a friendly `Error:` string instead of crashing.

**Files:** none (verification only).

**Verify:** `python -m pytest tests/ -q`

---

## 4. Acceptance checklist

- [ ] `src/lex_rank.py` exports `degree_centrality_scores`.
- [ ] `config.toml` has a `[summarization]` block with a resolvable `model_path`.
- [ ] `src/summarizer.py` imports no heavy ML package at module load and follows
      the §2.2 signatures.
- [ ] `Brain.summarize_text` is a `@dump_func_call` tool that reads
      `config.summarization`, caches the summarizer, and never raises to the caller.
- [ ] `Agent._build_tools()` includes `summarize_text`.
- [ ] `tests/test_lex_rank.py`, `tests/test_summarizer.py` and the new
      `tests/test_live_ai.py` tests pass with no model download or network.
- [ ] `README.md` documents how to store the model.
- [ ] `python -m pytest tests/ -q` is green.
```