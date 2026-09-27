# PLAN-EMBEDDING-TOOL.md — Add corpus build + semantic query tools

This plan adds two related **semantic search function tools** to the live
assistant, derived from `samples/sentence_transformers/semantic_search_quora_faiss.py`:

1. **`store_corpus`** — build a searchable corpus from a directory of documents
   and persist it to a single file.
2. **`query_corpus`** — query the stored corpus with a query text and return the
   most relevant passages/images.

It follows the same task-guide style as `PLAN.md` and `PLAN-SUMMARY-TOOL.md`
(small, independent tasks; goals / current behavior / expected behavior;
verified by unit tests).

> **Status.** **Phase 1 (T0–T6) is already implemented** (text-only corpus:
> `src/embedding_search.py`, `[embedding]` config, `store_corpus`/`query_corpus`,
> tests). **Phase 2 (T7–T13)** is the amendment described below: multi-format
> ingestion (pdf/txt/tex/md/html/excel/word), image records, and per-file
> the processed-document list in `PROCESSED.md`.

The plan is written for a **small automated coding agent**. Execute the tasks in
order. Each task is one commit and is not done until its tests pass.

---

## 1. Background

### 1.1 Sample algorithm (reference only)

`semantic_search_quora_faiss.py`:

1. Collect a corpus of sentences.
2. Embed them with a `SentenceTransformer`.
3. L2-normalize the embeddings (so inner product == cosine similarity).
4. Build a FAISS index.
5. For a query: embed + normalize the query, `index.search(query, top_k)`, then
   print `(score, sentence)` hits.

### 1.2 Current repository behavior

- `src/embedding_search.py` (Phase 1) reads only `.txt`/`.md`, chunks sentences
  into passages with `src.summarizer.split_sentences`, embeds them with
  `all-MiniLM-L6-v2`, and persists `{version, model_path, passages, embeddings}`.
- `Brain.store_corpus(directory, corpus_file, overwrite)` and
  `Brain.query_corpus(query, corpus_file, top_k)` wrap it; `EmbeddingCorpus` is
  cached per `Brain`.
- `src/summarizer.py` exposes `TextSummarizer` (LexRank extractive summary).
- `config.toml` has `[embedding]` (`model_path`, `device`, `corpus_path`,
  `extensions`, `chunk_size`, `top_k`).
- `.res/embedding/all-MiniLM-L6-v2` and `.res/embedding/clip-ViT-B-32`
  (2.3 GB, shared text+image space) are present; `.res/` is git-ignored.
- `setup-lms-cpu.ps1` installs `sentence-transformers[image]` and `faiss-cpu`.
- The active Python is the `.lms` virtual environment
  (`C:\...\live_assistant\.lms\Scripts\python.exe`).

### 1.3 Expected end state (after Phase 2)

- `store_corpus` ingests **pdf (including embedded images), txt, md, tex, html,
  docx (Word), xlsx (Excel), csv, rst, log, and common images** (png/jpg/jpeg/
  webp/bmp/tiff).
- Image files and images extracted from PDFs are embedded with the **CLIP**
  model and added to the corpus pickle as `image` records; text is embedded in
  the same shared space (multimodal mode) or with the text model (text mode).
- A **`PROCESSED.md`** manifest is appended next to the corpus pickle: a build
  header plus the absolute path of every processed document (errors annotated).
  The list is **never** stored in the pickle.
- `query_corpus` returns the most relevant text passages **and/or images**, with
  their source path.
- `setup-lms-cpu.ps1` installs the extra document-loader packages into `.lms`.
- No network access and no real model are used by unit tests.

---

## 2. Design

### 2.1 Architecture

```
src/live_ai.py::Brain.store_corpus / query_corpus   (tools; read config, cache)
        │  delegates to
        ▼
src/embedding_search.py::EmbeddingCorpus            (records, embeddings, index, processed list)
        │  uses
        ├── src/document_loader.py                   (per-format text + image extraction)
        ├── sentence_transformers (text model or CLIP), lazily
        └── faiss.IndexFlatIP (lazy; injectable backend)
```

`src/document_loader.py` must **not** import `pymupdf`, `docx`, `openpyxl`,
`bs4`, `PIL` or `pandas` at module top level; import them inside the extractor
functions so the module is import-safe and unit-testable with fakes.

### 2.2 Locked public signatures (Phase 2 additions in **bold**)

```python
# src/document_loader.py
TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".tex", ".rst", ".log", ".csv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".html", ".htm")
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS + DOCUMENT_EXTENSIONS + IMAGE_EXTENSIONS

@dataclass
class Document:
    source: str                 # path relative to the scanned root (POSIX-style)
    kind: str                   # "text" | "image"
    text: str = ""              # extracted text ("" for pure images)
    images: list = field(default_factory=list)   # absolute image paths
    pages: int | None = None
    error: str = ""

def strip_tex(text: str) -> str: ...
def extract_text(path) -> Document: ...
def extract_html(path) -> Document: ...
def extract_pdf(path, image_dir=None, source="") -> Document: ...
def extract_docx(path) -> Document: ...
def extract_xlsx(path) -> Document: ...
def extract_image(path) -> Document: ...
def load_document(path, root=None, image_dir=None) -> Document: ...
def iter_documents(directory, extensions=None, image_dir=None) -> list[Document]: ...

# src/embedding_search.py (amended)
class EmbeddingCorpus:
    def __init__(
        self,
        model_path,
        device="cpu",
        extensions=DEFAULT_EXTENSIONS,
        chunk_size=DEFAULT_CHUNK_SIZE,
        top_k=DEFAULT_TOP_K,
        model_factory=None,
        index_factory=None,
        sentence_tokenizer=None,
        image_model_path=None,          # CLIP model for image (and multimodal text) records
        multimodal=False,               # True -> one shared CLIP space for text + images
        image_extensions=IMAGE_EXTENSIONS,
        processed_name="PROCESSED.md",  # written next to the pickle; NOT inside it
        image_model_factory=None,       # test seam for the CLIP embedder
    ): ...
    def store(self, directory, output_file, overwrite=False) -> str: ...
    def query(self, corpus_file, query, top_k=None, kind=None) -> str: ...
```

`Brain.store_corpus` / `Brain.query_corpus` keep their Phase 1 signatures.

### 2.3 Config schema (amended `[embedding]` block)

```toml
[embedding] # semantic search over a local corpus (sentence/image embeddings + FAISS)
model_path = ".res/embedding/all-MiniLM-L6-v2"     # text-only-mode encoder
image_model_path = ".res/embedding/clip-ViT-B-32"  # shared text+image encoder (multimodal)
multimodal = true          # true -> use image_model_path for BOTH text and images
device = "cpu"             # "cpu" or "cuda"
corpus_path = ".res/corpus/corpus.pkl"  # resolved relative to this config file
processed_name = "PROCESSED.md"  # list of processed document absolute paths (NOT in the pkl)
extensions = "txt, md, markdown, tex, rst, log, csv, html, htm, pdf, docx, xlsx, png, jpg, jpeg, webp, bmp, tiff"
image_extensions = "png, jpg, jpeg, webp, bmp, tiff"
chunk_size = 3             # sentences per indexed passage
top_k = 5                  # default number of results returned by query_corpus
```

**Important:** the key is `processed_name` (not `*_file`) so `load_config` does not
resolve it against the config directory; it must be resolved **relative to the
corpus pickle**, per §2.5 step 9.

### 2.4 Corpus file format (version 2)

```python
{
    "version": 2,
    "mode": "multimodal" | "text",
    "text_model_path": str,
    "image_model_path": str,          # "" in text mode
    "records": [
        {"kind": "text",  "source": "docs/a.md",    "text": "passage ...", "page": None},
        {"kind": "image", "source": "docs/img.png", "text": "img.png",     "image_path": "<abs>"},
    ],
    "embeddings": np.ndarray,          # float32, L2-normalized, shape (N, dim); row i -> records[i]
}
```

`load_corpus` must accept **both** v2 (`records`) and the Phase 1 v1
(`passages` + `embeddings`) format, so existing corpora keep working.

### 2.5 `EmbeddingCorpus.store` algorithm (amended)

1. `output = Path(output_file).resolve()`. If it exists and `overwrite` is
   `False` → return `"File already exists."`.
2. `image_dir = output.parent / f"{output.stem}_images"`.
3. `documents = iter_documents(directory, self.extensions, image_dir=image_dir)`.
   If empty → return `f"No supported files found in '{directory}'"`.
4. Build `records`:
   - **text documents** (`doc.kind == "text"`): `build_passages([doc.text],
     chunk_size, self._tokenizer)` → one `{"kind":"text","source":doc.source,
     "text":passage,"page":doc.pages}` record per passage.
   - **image documents / extracted PDF images**: one
     `{"kind":"image","source":doc.source,"text":doc.source,"image_path":abs}`
     record per image. In **text mode** image records are skipped.
5. If no records → return `"No passages extracted from the input files."`.
6. Embed:
   - **multimodal**: build one embedding per record in order — text records via
     the CLIP text encoder (`model.encode(list_of_texts)`), image records via the
     CLIP image encoder (`model.encode([PIL.Image.open(p) ...])`) — then stack and
     L2-normalize.
   - **text mode**: embed text records with the text model.
7. `save_corpus(output, {version: 2, mode, text_model_path, image_model_path,
   records, embeddings})`.
8. Write `processed_path = output.parent / self.processed_name` in **append**
   mode: append a build header (`## Corpus build <UTC timestamp> — <output.name>`),
   record counts, then one bullet per processed document with its **absolute
   path** (`(Path(directory).resolve() / doc.source).resolve()`); failed files are
   annotated with `— ERROR: <doc.error>`. The list is not stored in the pickle.
9. Return e.g.
   `f"Corpus with {n_text} text / {n_image} image records saved to '{output}'; processed list appended to '{processed_path}'"`.

### 2.6 `EmbeddingCorpus.query` algorithm (amended)

1. Empty query → `"No query provided."`; missing corpus → friendly message.
   `kind` may be `"text"`/`"image"` to restrict the ranking to one record type;
   any other non-empty value → `"Invalid kind '<x>'. Use 'text' or 'image'."`.
2. `load_corpus` (v2 **or** v1).
3. **v2**: embed the query in the corpus `mode` (CLIP text encoder for
   multimodal, text model otherwise); dimension check; build the candidate set
   (`kind` filter or all records) and FAISS-search only those rows. Format each
   hit as `f"{rank}) [{score:.3f}] ({source}) {text}"`, appending ` [image]` and
   the image path for `image` records. Empty candidate set →
   `"No <kind> records found in the corpus."`.
4. **v1**: keep the Phase 1 behavior (`passages` + `embeddings`); `kind="image"`
   returns `"No image records found in the corpus."`.
5. Return a header line (`f"Top {k} matches for '{query}':"`, with the kind
   inserted when set) + ranked hits.

### 2.7 `Brain` tool behavior

- `_embedding_corpus()` passes the new config keys with
  `getattr(cfg, "key", default)` defaults:
  `image_model_path`, `multimodal`, `image_extensions`, `processed_name`.
- `store_corpus` / `query_corpus` keep their Phase 1 signatures, except
  `query_corpus` gains `kind: str = ""` (`"text"`/`"image"`, empty = both).

### 2.8 Rules for the executing agent

1. Do one task at a time, in order. A task is not done until its tests pass.
2. Never download or load a real model, never build a real FAISS index, and never
   open a real PDF/DOCX/XLSX in tests. Inject `model_factory`,
   `image_model_factory`, `index_factory`, `sentence_tokenizer` and the
   `document_loader` extractor seams.
3. Never import `faiss`, `sentence_transformers`, `torch`, `pymupdf`, `docx`,
   `openpyxl`, `bs4`, `PIL` or `pandas` at module top level; import them lazily.
4. Keep the signatures in §2.2 exactly.
5. The processed-document list must **never** be written into the pickle (assert in a test).
6. Run `python -m pytest tests/ -q` from the repo root before finishing a task.

---

## 3. Phase 1 tasks (T0–T6) — already implemented

These are retained for reference. If re-running from scratch, implement them
first; the current repository already satisfies them.

- **T0** `[embedding]` config block + config tests.
- **T1** `src/embedding_search.py` (text-only build/query).
- **T2** `tests/test_embedding_search.py`.
- **T3** wire `Brain.store_corpus` / `query_corpus` + register in `_build_tools`.
- **T4** `tests/test_live_ai.py` integration tests.
- **T5** README `## Semantic search (corpus + FAISS)` section.
- **T6** full-suite validation.

Phase 2 amends T0/T1/T3/T5 and adds new tasks below.

---

## 4. Phase 2 tasks (T7–T13)

### T7 — Implement `src/document_loader.py`

**Goal:** Extract text and images from the supported file types.

**Current behavior:** Phase 1 reads only `.txt`/`.md` via `collect_texts` in
`src/embedding_search.py`.

**Expected behavior:** `src/document_loader.py` implements exactly the signatures
and the `Document` dataclass from §2.2. Importing it must not import
`pymupdf`/`docx`/`openpyxl`/`bs4`/`PIL`/`pandas`.

Extraction rules:
- `.txt`/`.md`/`.markdown`/`.rst`/`.log`/`.csv`: `read_text(encoding="utf-8",
  errors="replace")`, `kind="text"`.
- `.tex`: read text, then `strip_tex` removes `%` comments and simple
  `\command{...}`/`\begin{...}`/`\end{...}`/`$...$` noise (regex; keep the
  sentence text), `kind="text"`.
- `.html`/`.htm`: `BeautifulSoup(text, "html.parser").get_text("\n")`,
  `kind="text"`.
- `.pdf`: `pymupdf.open(path)`; concatenate `page.get_text("text")`; set
  `pages=doc.page_count`; extract embedded images with
  `page.get_images(full=True)` + `doc.extract_image(xref)` and save them under
  `image_dir` (created on demand) as `<safe-source>_p<page>_<xref>.<ext>`;
  append their absolute paths to `doc.images`. `kind="text"` when text or images
  exist (images become separate records in §2.5).
- `.docx`: `docx.Document(path)`; join paragraphs and table cells; `kind="text"`.
- `.xlsx`: `openpyxl.load_workbook(path, read_only=True, data_only=True)`; per
  sheet, join non-empty cell values prefixed with the sheet name; `kind="text"`.
- images (`.png`/`.jpg`/`.jpeg`/`.webp`/`.bmp`/`.tif`/`.tiff`):
  `kind="image"`, `images=[str(path.resolve())]`, optional `(w, h)` in `text`.
- Unsupported/corrupt files: return a `Document` with `error=str(exc)` instead of
  raising, so one bad file never aborts a corpus build.
- `iter_documents` walks `Path(directory).rglob("*")` sorted, filters by
  normalized extensions, and calls `load_document` per file (relative `source`).

**Test seams:** module-level indirection functions the tests monkeypatch, e.g.
`_open_pdf`, `_open_docx`, `_open_xlsx`, `_open_image`. All imports stay lazy.

**Files:**
- Add `src/document_loader.py`.

**Tests:** add `tests/test_document_loader.py` (see T8).

**Verify:** `python -m pytest tests/test_document_loader.py -q`

---

### T8 — Unit tests for `src/document_loader.py`

**Goal:** Prove multi-format extraction and image handling with no real
documents and no heavy imports.

**Current behavior:** No loader module or tests.

**Expected behavior:** `tests/test_document_loader.py` passes with fakes.

Recommended test list:
- `test_import_is_lazy`: pop `pymupdf`, `docx`, `openpyxl`, `bs4`, `PIL`,
  `pandas`; import `src.document_loader`; assert none are imported.
- `test_extract_text_plain` and `test_extract_tex_strips_commands`
  (`\section{X}` -> text, `% comment` removed).
- `test_extract_html_uses_get_text`: write a small HTML file, assert tag text
  present and tags stripped.
- `test_extract_image_returns_image_record`: tiny file with an image extension,
  assert `kind == "image"` and the absolute path is recorded.
- `test_extract_pdf_monkeypatched`: monkeypatch `_open_pdf` to return a fake doc
  with two pages and one embedded image; assert concatenated text, `pages == 2`,
  and that the image was written under `image_dir` and recorded.
- `test_extract_docx_monkeypatched` and `test_extract_xlsx_monkeypatched`.
- `test_load_document_error_is_captured`: an unreadable file returns a `Document`
  with a non-empty `error` and does not raise.
- `test_iter_documents_filters_and_sorts`: directory with `.txt`, `.html`,
  `.bin`, and a nested `.md`; assert sources are relative, sorted and `.bin` is
  skipped.

**Files:** Add `tests/test_document_loader.py`.

**Verify:** `python -m pytest tests/test_document_loader.py -q`

---

### T9 — Extend `EmbeddingCorpus` (v2 records, multimodal, processed list)

**Goal:** Build v2 corpora from all supported files, embed images with CLIP, and
emit a `PROCESSED.md` list of processed document absolute paths.

**Current behavior:** `store` handles only `.txt`/`.md`, writes v1 `passages` +
`embeddings`, and produces no processed list.

**Expected behavior:** Implement the amended signatures and the algorithms in
§2.4–§2.6:
- New `__init__` parameters (`image_model_path`, `multimodal`,
  `image_extensions`, `processed_name`, `image_model_factory`).
- `store` uses `src.document_loader.iter_documents`, builds `text` and `image`
  records, embeds them (one shared CLIP space when `multimodal`, else text model
  only; images skipped in text mode), writes the v2 pickle, then **appends**
  the absolute path of every processed document to `PROCESSED.md` next to the
  pickle.
- `query` supports v2 `records` (returning text sources and image paths) and
  remains backward compatible with v1 `passages`.
- `load_corpus` accepts both versions (or add a `corpus_version(payload)` helper).
- `_default_index_factory` and `_encode`/`_encode_multimodal` stay lazy.
- The `image_model_factory` default builds a
  `SentenceTransformer(self.image_model_path, device=self.device)` after
  `is_model_available`.

**Files:**
- Edit `src/embedding_search.py`.

**Tests:** extend `tests/test_embedding_search.py` (see T10).

**Verify:** `python -m pytest tests/test_embedding_search.py -q`

---

### T10 — Unit tests for the extended corpus

**Goal:** Prove v2 records, image embedding, and the PROCESSED.md contract.

**Current behavior:** `tests/test_embedding_search.py` only covers v1.

**Expected behavior:** New tests (inject all seams; no real files unless easy
text files are used):
- `test_store_v2_writes_records_not_processed_list`: build from a dir with one
  `.txt` and one image; assert the pickle has `version == 2`, `records`, and
  `embeddings`; assert no `processed`/`summary` key exists.
- `test_store_appends_processed_md`: after two `store` calls (different corpus
  files in the same dir or overwrite), `PROCESSED.md` exists next to the pickle
  and lists each processed document's absolute path, appended (two build headers).
- `test_store_image_records_in_multimodal`: with a fake `image_model_factory`
  that records whether images were encoded, assert image records exist and the
  image encoder was used; assert the embeddings matrix has one row per record and
  rows are unit norm.
- `test_store_text_mode_skips_images`: `multimodal=False` → no image records.
- `test_query_v2_formats_sources`: query a v2 corpus, assert each hit shows
  `(<source>)` and image hits show ` [image]`.
- `test_query_reads_v1_corpus`: a manually written v1 pickle still returns hits.
- `test_missing_image_model_raises_friendly_error`: multimodal with a bad
  `image_model_path` and no factory raises `FileNotFoundError` mentioning the
  path.

**Files:** Edit `tests/test_embedding_search.py`.

**Verify:** `python -m pytest tests/test_embedding_search.py -q`

---

### T11 — Wire the new config into `Brain` + tests

**Goal:** `Brain` passes the Phase 2 config keys to `EmbeddingCorpus`.

**Current behavior:** `_embedding_corpus()` passes only Phase 1 keys.

**Expected behavior:** `_embedding_corpus()` also passes
`image_model_path=getattr(cfg, "image_model_path", None)`,
`multimodal=getattr(cfg, "multimodal", False)`,
`image_extensions=getattr(cfg, "image_extensions", ...)`,
`processed_name=getattr(cfg, "processed_name", "PROCESSED.md")`.

**Files:**
- Edit `src/live_ai.py`.

**Tests:** extend `tests/test_live_ai.py`
- `test_store_corpus_passes_multimodal_config`: the `Recorder.__init__` kwargs
  include `image_model_path`, `multimodal is True`, `processed_name`.
- Keep the existing Phase 1 tests green (they use partial configs; the defaults
  must not break them).

**Verify:** `python -m pytest tests/test_live_ai.py -q`

---

### T12 — Config, setup script, and README

**Goal:** Expose the new keys, install the new packages into `.lms`, and document
the behavior.

**Current behavior:** `config.toml` has Phase 1 keys only;
`setup-lms-cpu.ps1` lacks the document loaders; README documents text-only
semantic search.

**Expected behavior:**
1. Update `config.toml` `[embedding]` to the block in §2.3 (`image_model_path`,
   `multimodal`, `processed_name`, expanded `extensions`, `image_extensions`).
2. Amend `setup-lms-cpu.ps1`:
   - Activate `.lms` before installing, so packages land in the active venv:

     ```powershell
     Write-Host "=== Activating .lms virtual environment ==="
     if (-not (Test-Path ".\.lms\Scripts\Activate.ps1")) {
         Write-Host "ERROR: .lms virtual environment not found. Create it with: python -m venv .lms"
         exit 1
     }
     & ".\.lms\Scripts\Activate.ps1"
     ```
   - Add under a new comment:

     ```powershell
     # DOCUMENT LOADERS (store_corpus: pdf/docx/xlsx + images)
     pip install pymupdf
     pip install python-docx
     pip install openpyxl
     ```
3. Extend the README semantic-search section:
   - List the supported formats (pdf incl. embedded images, txt, md, tex, rst,
     csv, html/htm, docx, xlsx, and common images); note legacy `.doc`/`.xls`
     are not supported.
   - Explain multimodal mode: `image_model_path` = CLIP
     (`.res/embedding/clip-ViT-B-32`) encodes text **and** images into one shared
     space, so `query_corpus` can return images too.
   - Document the CLIP model storage command
     (`hf download sentence-transformers/clip-ViT-B-32 --local-dir
     .res/embedding/clip-ViT-B-32`) and that `.res/` is git-ignored.
   - Document `PROCESSED.md`: a build header plus the absolute path of every
     processed document is written next to the corpus pickle and is **not** part
     of the pickle; set `processed_name` to change the filename.
4. Install the new packages in the active `.lms` environment now:
   `python -m pip install pymupdf python-docx openpyxl`
   (already done during plan authoring; re-run if missing).

**Files:**
- Edit `config.toml`, `setup-lms-cpu.ps1`, `README.md`.

**Tests:** extend `tests/test_common.py`
- `test_real_config_has_multimodal_embedding_keys`: assert `image_model_path`,
  `multimodal`, `processed_name`, `image_extensions` exist; `image_model_path` is
  absolute; `processed_name` is a plain string (not resolved to a path).

**Verify:** `python -m pytest tests/test_common.py -q`

---

### T13 — Final validation (integration gate)

**Goal:** Confirm the whole amended feature is wired and nothing regressed.

**Steps:**
1. Run the full suite: `python -m pytest tests/ -q` → all tests pass.
2. Static checks:

   ```bash
   python -c "import src.live_ai as m; n=[f.__name__ for f in m.Agent._build_tools(m.Agent.__new__(m.Agent))]; print('store_corpus' in n and 'query_corpus' in n)"
   python -c "import src.document_loader, sys; print(all(x not in sys.modules for x in ('pymupdf','docx','openpyxl','bs4','PIL')))"
   ```

   Both must print `True`.
3. Manual smoke test with real models and documents (optional, not CI):
   put a `.pdf`, a `.docx`, an `.xlsx`, a `.md` and a `.png` in a folder, then:

   ```bash
   python main.py --chat --prompt "Store a corpus from the directory docs"
   python main.py --chat --prompt "Query the corpus: what is quantum computing?"
   ```

   Expect a `PROCESSED.md` next to the corpus pickle and ranked text/image hits.
   If a model is absent, the tools must return a friendly `Error:` string.

**Files:** none (verification only).

**Verify:** `python -m pytest tests/ -q`

---

## 5. Acceptance checklist

- [ ] `src/document_loader.py` extracts text from txt/md/tex/rst/log/csv/html/
      pdf/docx/xlsx and produces image paths from pdfs and image files, with lazy
      imports and no heavy imports at module load.
- [ ] `setup-lms-cpu.ps1` activates `.lms` and installs `pymupdf`,
      `python-docx`, `openpyxl` into it; README documents this.
- [ ] `[embedding]` has `image_model_path`, `multimodal`, `processed_name`,
      `image_extensions`; `processed_name` is not path-resolved.
- [ ] `EmbeddingCorpus.store` writes a **v2** pickle with `records` +
      `embeddings`; text uses the text model (or CLIP in multimodal mode) and
      images use the CLIP image encoder; images are skipped in text mode.
- [ ] The processed-document list (absolute paths) is **appended** to
      `PROCESSED.md` next to the pickle and is **never** written into the pickle
      (asserted by a test).
- [ ] `EmbeddingCorpus.query` handles v2 records (text + image sources) and still
      reads v1 corpora.
- [ ] `Brain._embedding_corpus` forwards the new config keys; existing tools and
      tests stay green.
- [ ] `tests/test_document_loader.py` and the extended
      `tests/test_embedding_search.py` / `test_live_ai.py` / `test_common.py`
      tests pass with no network and no real model/document.
- [ ] README documents supported formats, multimodal CLIP mode, `PROCESSED.md`,
      and model storage for both `all-MiniLM-L6-v2` and `clip-ViT-B-32`.
- [ ] `python -m pytest tests/ -q` is green.
