# PLAN — Replace `src/tools/web.py` implementation with the DDGS library

Reference: DDGS (Dux Distributed Global Search) — https://github.com/deedy5/ddgs
Python: 3.12 · Venv: `.liveai-cpu` · Test runner: pytest

---

## 1. Goal

Rewrite the web tools in `src/tools/web.py` so that all web search and page
extraction functionality is provided by the `DDGS` class (`pip install ddgs`)
instead of the current Selenium (headless Chrome) + requests/BeautifulSoup
implementation. Expose one agent tool per DDGS method — `text()`, `images()`,
`videos()`, `news()`, `books()`, `extract()` — keep `download_file()` as-is,
and update the exports, agent registration, tests, README and setup scripts
accordingly.

Benefits: no Chrome/ChromeDriver requirement, no DOM-scraping fragility,
multi-engine metasearch (bing, duckduckgo, google, …), markdown-aware page
extraction, and four new capabilities (images, videos, news, books).

## 2. Current behaviour

- `web_search(query, n_results, timeout_s)` — launches a real headless Chrome
  via Selenium + `webdriver-manager`, drives duckduckgo.com, scrapes the text
  of every `<article>` element, renders a numbered block via
  `format_search_results`. Heavy, slow, breaks whenever the DuckDuckGo DOM
  changes, requires Chrome installed on the machine.
- `webpage_headlines(url, ...)` and `webpage_content(url, ...)` — fetch the
  page with `requests`, parse with BeautifulSoup and return the text of
  `h1`/`h2` or `p` tags longer than `relevant_text_length`.
- `extract_text(soup, tags, min_len)` — internal BS4 helper for the two tools
  above.
- `download_file(url, out_dir, filename, overwrite, timeout_s)` — streams a
  URL to disk with `requests`, name from Content-Disposition / URL /
  Content-Type. **Unrelated to DDGS: keep unchanged.**
- Setup scripts (`setup-liveai-cpu.sh`, `setup-liveai-cpu.ps1`) install
  `requests`, `beautifulsoup4`, `selenium`, `webdriver-manager`.
- README has **no** section documenting the web tools.
- `selenium` is used nowhere else in the repo (only `src/tools/web.py`);
  `bs4` is also used by `src/document_loader.py` (must stay installed).
- Tests involved: `tests/test_tools.py` (T0 lazy-import list contains
  `"selenium"`, `tools.requests` seam for webpage tools, `_create_chrome_driver`
  / `_collect_results` seams for web_search), `tests/test_live_ai.py`
  (tool-name lists).

## 3. Expected behaviour

After this plan is implemented:

- `src/tools/web.py` contains exactly these public tools (all decorated with
  `@dump_func_call`, all returning strings, never raising):

  | Tool | DDGS call | Purpose (docstring must start with "Useful when…") |
  |---|---|---|
  | `web_search(query, n_results=5, region="us-en", safesearch="moderate", timelimit=None, timeout_s=10.0)` | `DDGS().text(...)` | web page search for a topic |
  | `web_images(query, n_results=5, ...)` | `DDGS().images(...)` | find pictures on the web (URLs can be saved with `download_file`) |
  | `web_videos(query, n_results=5, ...)` | `DDGS().videos(...)` | find videos on the web |
  | `web_news(query, n_results=5, timelimit=None, ...)` | `DDGS().news(...)` | recent news about a topic |
  | `web_books(query, n_results=5, ...)` | `DDGS().books(...)` | find books |
  | `extract_webpage(url, fmt="text_markdown", timeout_s=10.0)` | `DDGS().extract(url, fmt=...)` | extract a webpage's content (replaces `webpage_content` / `webpage_headlines`) |
  | `download_file(...)` | — | unchanged |

- `webpage_headlines`, `webpage_content`, `extract_text`, `_fetch_soup`,
  `_create_chrome_driver`, `_collect_results` are **removed**; no `selenium`
  or `webdriver-manager` imports remain anywhere in `src/` or `tests/`.
- All DDGS-backed tools go through a single test seam
  `_ddgs_client()` in `src/tools/web.py` (lazy-imports and returns a `DDGS`
  instance). Tests monkeypatch `src.tools.web._ddgs_client` with a fake —
  same seam pattern as `_VIDEO_CAPTURE_FACTORY` etc. described in the
  `src/tools/__init__.py` docstring.
- Results are rendered for the LLM as numbered blocks (one per result) built
  from the dict fields listed in §5 — keep the existing
  `format_search_results` style (`"1) …"` lines separated by 50 `=`).
- Errors (network failures, empty results, bad `fmt`) are returned as
  `"Error: …"` strings, matching the current convention.
- `from ddgs import DDGS` happens lazily inside `_ddgs_client()` so that
  importing `src.tools` stays cheap (the lazy-import test asserts `ddgs` is
  **not** loaded at package import).
- Registration: new tools exported from `src/tools/__init__.py` (imports +
  `__all__` + module docstring web line) and listed in `_default_tools()` in
  `src/live_ai.py`.
- Setup scripts install `ddgs` instead of `selenium` + `webdriver-manager`
  (`requests`, `beautifulsoup4` stay — still used by `download_file` and
  `document_loader`).
- README gains a `## Web search and page extraction (DDGS)` section
  documenting the six tools with one example call each.
- `config.toml` needs **no** changes (web tools use DDGS defaults; no config
  section is introduced).

### Result formats (exact fields per tool, from DDGS dicts)

- `web_search` → each item: `title`, `href`, `body`
- `web_images` → `title`, `image` (direct image URL), `thumbnail`,
  `width`, `height`, `source`
- `web_videos` → `title`, `content` (video URL), `duration`, `publisher`,
  `uploader`
- `web_news` → `date`, `title`, `body`, `url`, `source`
- `web_books` → `title`, `author`, `publisher`, `info`, `url`
- `extract_webpage` → `{"url": ..., "content": ...}` — return `"content"`
  (truncated to a sane size, e.g. first ~4000 chars) prefixed with
  `\nFrom [{url}]\n` like the old tools. Supported `fmt`: `text_markdown`
  (default), `text_plain`, `text_rich`, `text`.

## 4. Constraints (apply to every task)

- Python 3.12; keep the code style of the file (type hints, docstrings,
  lazy imports inside functions for heavy/optional deps).
- Every tool returns a `str`; on failure return `f"Error: {exc}"` — the LLM
  must be able to read the error.
- Public API churn must be reflected in **all** of: `src/tools/__init__.py`,
  `src/live_ai.py`, `tests/`.
- Run the full suite after each task: `python -m pytest tests/ -q`
  (Windows: `./.liveai-cpu/Scripts/python.exe -m pytest tests/ -q`).

---

## Tasks

### T1 — Swap the dependency in the setup scripts and venv

- **Files:** `setup-liveai-cpu.sh`, `setup-liveai-cpu.ps1`
- **Current behaviour:** `# TOOLS` block installs `selenium` and
  `webdriver-manager` (plus `requests`, `beautifulsoup4`).
- **Expected behaviour:** the `# TOOLS` block installs `ddgs`; the two lines
  `pip install selenium` and `pip install webdriver-manager` are removed.
  `requests` and `beautifulsoup4` stay. Then run the scripts' relevant part
  manually (or `pip install -U ddgs` in the venv) so the package is actually
  present for later tasks.
- **Validation:**
  - `grep -n selenium setup-liveai-cpu.sh setup-liveai-cpu.ps1` → no matches.
  - `grep -n "pip install ddgs" setup-liveai-cpu.sh setup-liveai-cpu.ps1` →
    one match in each.
  - `python -c "from ddgs import DDGS; print(DDGS)"` succeeds in the venv.

### T2 — Rewrite `src/tools/web.py`: `web_search` backed by `DDGS().text()`

- **Files:** `src/tools/web.py`
- **Current behaviour:** `web_search` creates a Chrome driver
  (`_create_chrome_driver`) and scrapes DuckDuckGo (`_collect_results`);
  `format_search_results` renders a list of strings.
- **Expected behaviour:**
  - Delete `_create_chrome_driver` and `_collect_results` (and all selenium
    imports).
  - Add module-level `def _ddgs_client(timeout_s: float = 10.0):` that lazily
    does `from ddgs import DDGS` and returns `DDGS(timeout=max(1, int(timeout_s)))`.
    This is the seam tests patch.
  - `web_search(query, n_results=5, region="us-en", safesearch="moderate",
    timelimit=None, timeout_s=10.0)`: call
    `_ddgs_client(timeout_s).text(query, region=region, safesearch=safesearch,
    timelimit=timelimit, max_results=n_results)`, then render via
    `format_search_results`.
  - Rework `format_search_results(results: list[dict], n_results=5)` to keep
    the current look (`"Search Results:"` header, `f"{i}) …"` items, 50-`=`
    separators) with each item showing `title`, `href`, `body` (skip empty
    fields). No ` › `→`/` replacement needed anymore.
  - Empty result list → return a short `"No results found for: {query}"`.
    Exceptions → `f"Error: {exc}"`.
  - Update the module docstring (no longer "DuckDuckGo search" via Selenium).
- **Validation:**
  - `python -c "import src.tools.web"` does not import `selenium` or `ddgs`.
  - `python -m pytest tests/test_tools.py -q` — web_search tests updated in
    T6 must pass (do them together or stub-test locally).
  - Network smoke (optional):
    `python -c "from src.tools import web_search; print(web_search('python programming', n_results=3))"`
    returns numbered results with title/href/body.

### T3 — Add `web_images`, `web_videos`, `web_news`, `web_books`

- **Files:** `src/tools/web.py`
- **Current behaviour:** these tools do not exist (only `web_search` exists).
- **Expected behaviour:** four thin tools, each following the `web_search`
  pattern: `@dump_func_call`, docstring starting with "Useful when …",
  shared private renderer `_format_ddgs_results(kind, results, n_results)`
  (or per-tool renderers) emitting the numbered-block format with the fields
  listed in §3, `"Error: …"` on failure. Suggested signatures:
  - `web_images(query, n_results=5, region="us-en", safesearch="moderate",
    timelimit=None, size=None, color=None, type_image=None, layout=None,
    license_image=None, timeout_s=10.0)`
  - `web_videos(query, n_results=5, region="us-en", safesearch="moderate",
    timelimit=None, resolution=None, duration=None, timeout_s=10.0)`
  - `web_news(query, n_results=5, region="us-en", safesearch="moderate",
    timelimit=None, timeout_s=10.0)`
  - `web_books(query, n_results=5, timeout_s=10.0)` (DDGS `books()` has no
    region/safesearch/timelimit)
  Mention in the `web_images`/`web_videos` docstrings that result URLs can be
  saved to disk with `download_file`.
- **Validation:**
  - `python -m pytest tests/test_tools.py -q` (with the new tests from T6).
  - Network smoke (optional): `web_images('butterfly', n_results=2)` etc.
    each return numbered results with the expected fields.

### T4 — Replace `webpage_content`/`webpage_headlines` with `extract_webpage`

- **Files:** `src/tools/web.py`
- **Current behaviour:** two requests+BS4 tools (`webpage_headlines`,
  `webpage_content`) plus helpers `extract_text` and `_fetch_soup`.
- **Expected behaviour:**
  - Remove `webpage_headlines`, `webpage_content`, `extract_text`,
    `_fetch_soup` and their BS4/requests usage from `web.py`.
  - Add `extract_webpage(url, fmt="text_markdown", timeout_s=10.0)`:
    `result = _ddgs_client(timeout_s).extract(url, fmt=fmt)`; return
    `f"\nFrom [{result['url']}]\n{content[:4000]}"`; unknown/failed `fmt` or
    fetch errors → `"Error: …"`; `fmt="content"` (raw bytes) is rejected with
    an error message because the tool must return text.
  - `__all__` in `web.py` updated to the final name set:
    `download_file, extract_webpage, format_search_results, web_images,
    web_news, web_search, web_videos, web_books`.
- **Decision record:** the old per-tag headline/paragraph filters are dropped
  on purpose — `extract_webpage` covers the "summarize this page" use case
  better (markdown keeps headers). If product feedback requires headline-only
  extraction later, add it as a new small tool instead of restoring BS4.
- **Validation:**
  - `grep -rn "selenium\|BeautifulSoup\|webdriver" src/tools/` → only matches
    outside `web.py` (none should remain in `web.py`).
  - Network smoke (optional):
    `extract_webpage("https://example.com")` contains "Example Domain".

### T5 — Update exports and agent registration

- **Files:** `src/tools/__init__.py`, `src/live_ai.py`
- **Current behaviour:** `src/tools/__init__.py` imports/exports
  `download_file, extract_text, format_search_results, webpage_content,
  webpage_headlines, web_search`; `_default_tools()` in `live_ai.py` registers
  `web_search, webpage_content, webpage_headlines, download_file, …`.
- **Expected behaviour:**
  - `src/tools/__init__.py`: import and re-export the seven final names
    (`web_search`, `web_images`, `web_videos`, `web_news`, `web_books`,
    `extract_webpage`, `download_file` + `format_search_results`), drop the
    removed ones, update the `web` line of the module docstring, extend
    `__all__` in sorted order.
  - `src/live_ai.py`: replace `webpage_content, webpage_headlines` with
    `extract_webpage` and add `web_images, web_videos, web_news, web_books`
    next to `web_search` in `_default_tools()` and its imports.
  - Keep the `requests` lazy `__getattr__` seam (used by `download_file`
    tests).
- **Validation:**
  - `python -c "import src.tools as t; [getattr(t, n) for n in ('web_search','web_images','web_videos','web_news','web_books','extract_webpage','download_file')]"`.
  - `python -m pytest tests/test_live_ai.py -q` (with updated name lists from
    T6).

### T6 — Update and extend the tests

- **Files:** `tests/test_tools.py`, `tests/test_live_ai.py`
- **Current behaviour:**
  - `test_import_tools_is_lazy` lists `"selenium"` as a heavy module.
  - T5/T6 sections patch `tools.requests.get` (webpage tools) and
    `tools.web._create_chrome_driver` / `tools.web._collect_results`
    (web_search); a `FakeDriver` with a `quit_calls` counter exists.
  - `test_live_ai.py` checks `"download_file"` is registered and probes
    `"web_search"` in the callable-tools list.
- **Expected behaviour:**
  - Replace `"selenium"` with `"ddgs"` in the lazy-import tuple.
  - Introduce a `FakeDDGS` class in the web section returning canned dicts
    per method (`text`, `images`, `videos`, `news`, `books`, `extract`) and a
    fixture/monkeypatch helper that patches `src.tools.web._ddgs_client` to
    return it.
  - Delete the Chrome-driver/FakeDriver tests and the `tools.requests.get`
    webpage tests; write one test per tool covering: formatted numbered
    output with the §3 fields, `n_results` limiting, and the
    `"Error:"`-string path (fake raises). Add a test that
    `extract_webpage` prefixes `\nFrom [url]` and truncates long content,
    and one that rejects `fmt="content"`.
  - Keep `download_file` tests untouched (they still patch
    `tools.requests.get`).
  - `test_live_ai.py`: assert the new tool names are registered
    (`extract_webpage`, `web_images`, `web_videos`, `web_news`, `web_books`)
    and none of the removed ones (`webpage_content`, `webpage_headlines`)
    are.
- **Validation:** `python -m pytest tests/ -q` → all green, including
  `test_import_tools_is_lazy` (which now also proves `ddgs` is not imported
  eagerly).

### T7 — Document the web tools in the README

- **Files:** `README.md`
- **Current behaviour:** README has no section about the web tools
  (`web_search`, etc. appear nowhere).
- **Expected behaviour:** add a `## Web search and page extraction (DDGS)`
  section (place it before `## Interactive chat input`), containing:
  - one-line description: metasearch via the
    [DDGS](https://github.com/deedy5/ddgs) library, no browser required;
  - a table or bullet list of the six tools with their key parameters and
    the DDGS method they wrap;
  - one example call per tool (same style as the other sections);
  - a note that image/video URLs can be saved with `download_file` and that
    `selenium`/`webdriver-manager` are no longer needed;
  - update the Setup section's package list if it enumerates packages
    (mention `ddgs`).
- **Validation:** `grep -n "DDGS" README.md` finds the new section;
  every tool name from §3 appears in the README.

### T8 — Final sweep and full validation

- **Files:** possibly none (verification only).
- **Expected behaviour:** repo-wide consistency check.
- **Validation (Definition of Done):**
  1. `grep -rn "selenium\|webdriver\|_create_chrome_driver\|webpage_headlines\|webpage_content" src/ tests/ bin/ config.toml README.md`
     → no matches (excluding `__pycache__`).
  2. `python -m pytest tests/ -q` → all tests pass.
  3. `python -c "from src.live_ai import _default_tools"` imports cleanly and
     includes the seven web tools.
  4. Network smoke test of all six DDGS-backed tools through the public API
     (`from src.tools import web_search, web_images, web_videos, web_news,
     web_books, extract_webpage`) — each returns either results or a
     well-formed `"Error: …"` string, never raises.
  5. Optional end-to-end: start the chat (`bin/chat`) and ask
     "search the web for python programming" and
     "extract the content of https://example.com" — the agent calls the new
     tools and answers from them.

---

## 5. Risks / notes for the implementing agent

- DDGS hits live search engines: engines may throttle or change. The
  `"Error: …"` convention plus the `_ddgs_client` seam keeps this testable
  without network.
- `DDGS(timeout=...)` accepts an int (seconds); clamp `timeout_s` with
  `max(1, int(timeout_s))`.
- `ddgs` requires Python ≥ 3.10 — fine for the pinned 3.12.
- Do not touch `src/document_loader.py` (it uses `bs4` independently) and do
  not remove `beautifulsoup4`/`requests` from the setup scripts.
- Keep `download_file` exactly as it is (including its tests).
- Commit after each task so a failing task can be rolled back independently.