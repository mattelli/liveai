# PLAN-TOOLS.md — Improve `src/tools.py`

Addendum to [`PLAN.md`](PLAN.md). Everything in `PLAN.md` §0 (rules for the executing
agent), §2 (test strategy and conventions) and §3 (design constraints) still applies.
Read those sections before starting any task here.

**Scope.** Correctness, safety and testability of `src/tools.py`, the file that backs the
agent's tool calls. No change to `Brain`/`Agent`, `ConversationSession`, audio, or the tool
registration list in `src/live_ai.py`.

**Target files.** `src/tools.py`, `tests/conftest.py`, new `tests/test_tools.py`.

**Hardware/network rule (PLAN.md §0.3).** No test may open a webcam, call the network,
launch Chrome, download a driver, or fetch market data. Every such call is either
monkeypatched (e.g. `tools.requests.get`, `tools.yf.download`) or reached through an
injectable seam added below.

**Baseline (current behaviour, for reference).** Line numbers are from the current file and
may shift as tasks land.

| # | Location | Issue |
|---|----------|-------|
| D1 | imports (lines 3-20) | `cv2`, `selenium`, `yfinance`, `matplotlib`, `bs4` are imported at module top, so importing `src.tools` needs the whole audio/browser/ML stack |
| D2 | `ValidYF.interval` (line 36) | `' 5m'` has a leading space, so the value is never a valid interval |
| D3 | `write_text_file` (line 91) | `dest_path.exists` is a **method object** (always truthy) instead of `dest_path.exists()`; no parent-dir creation; no overwrite option |
| D4 | `search_in_file` (line 326) | `f"Error: '{e}'"` is computed but not returned, so read errors are silently reported as "not found" |
| D5 | `replace_string_in_file` (line 393) | counts with `re.findall(old_string, …)` (regex) but replaces with `str.replace` (literal); a regex-special `old_string` miscounts |
| D6 | `run_command` (lines 118-128) | `cmd.split()` with `shell=True` mangles quoted arguments; `process.wait()` before `communicate()` can deadlock; no timeout; no `cwd` parameter |
| D7 | `exit_gracefully` (line 45) | uses the interactive-helper `exit(0)` instead of `raise SystemExit(0)` |
| D8 | `webpage_headlines` / `webpage_content` (lines 133-170) | duplicate fetch/parse code; the shared logic cannot be tested once |
| D9 | `web_search` (lines 172-202) | builds Chrome + downloads a driver inline, so the formatting/cleanup path can never be unit-tested |
| D10 | `fetch_financial_data` (lines 222-261) | assumes one column layout; `save_financial_chart` writes `f"{symbol}.jpg"` with unchecked symbols (`^GSPC`, `BRK/B`) |
| D11 | `search_pattern_in_files` (lines 344-345) | `"py"` never matches `.py` (leading dot not normalised); the default `exclude_glob=".*"` behaviour is undocumented |
| D12 | `capture_image` (lines 204-219) | `cv2.VideoCapture`/`cv2.imwrite` are hard-wired, so the success/failure branches cannot be tested |

---

## 0. Conventions for the executing agent

1. One task per commit, in the order of §1. Each task is self-contained.
2. Write the test **first**, watch it fail, then implement the change. A task is done only
   when its validation command passes.
3. Keep the public tool signatures compatible with how `src/live_ai.py` registers them.
   New parameters must have defaults; `fetch_financial_data`, `save_financial_chart`,
   `capture_image`, `run_command` and `web_search` keep their current callable form for the
   model.
4. Never call `sys.exit`/`exit` inside `src/`; raise `SystemExit` where an exit is the
   intended effect.
5. Every tool parameter that has a default **must** have a type annotation that LM Studio's
   schema builder accepts. `get_type_hints` drops unannotated parameters and the LM Studio
   SDK then raises `KeyError: '<param>'`; `str | Path | None` unions are rejected with a
   `TypeError`. Use `str | None` / `Any`, or keep test-only seams out of the signature
   entirely (module-level hooks such as `_VIDEO_CAPTURE_FACTORY`).
6. Tests must not import `selenium`, `cv2`, `yfinance`, or `matplotlib`; if a test needs a
   behaviour behind one of them, use the injectable seam created in the task.
7. `python -m pytest tests/ -q` must stay green at the end of every task.

---

## 1. Task index

| ID | Title | Depends on | Phase |
|----|-------|-----------|-------|
| T0 | Lazy heavy imports + test bootstrap | — | Foundations |
| T1 | Fix `ValidYF` + test the read-only tools | T0 | Correctness |
| T2 | Fix and extend `write_text_file` | T0 | Correctness |
| T3 | Fix `search_in_file` and `replace_string_in_file` | T0 | Correctness |
| T4 | Harden `run_command` and `exit_gracefully` | T0 | Correctness |
| T5 | Share and test the webpage helpers | T0 | Correctness |
| T6 | Make `web_search` testable (`_collect_results`, `format_search_results`) | T0 | Correctness |
| T7 | Financial helpers: `select_financial_columns`, `safe_filename` | T0 | Correctness |
| T8 | `normalize_extensions` + fix `search_pattern_in_files` | T0 | Correctness |
| T9 | `capture_image` seam + typing/cleanup/acceptance | T0 | Verification |

T0 is the only prerequisite; the rest are independent except T9.

---

## 2. Helpers to add (shared by several tasks)

```python
def normalize_extensions(value: str) -> list[str]:
    """Parse 'py, .MD' -> ['.py', '.md'] (lowercase, leading dot)."""

def extract_text(soup, tags, min_len: int = 20) -> list[str]:
    """Return stripped text of matching tags longer than min_len."""

def format_search_results(results: list[str], n_results: int = 5) -> str:
    """Render search results as the numbered block the model expects."""

def select_financial_columns(data, symbol: str):
    """Reduce a yfinance frame to ['Date','Open','Close'] or return an error string."""

def safe_filename(symbol: str) -> str:
    """Turn '^GSPC' / 'BRK/B' into a filesystem-safe filename stem."""
```

---

## T0 — Lazy heavy imports and test bootstrap

**Goal.** `import src.tools` must not require `cv2`, `selenium`, `yfinance`,
`matplotlib`, or `bs4`, so the module can be unit-tested without the browser/ML stack.

**Current behaviour (lines 3-20).** All of these are imported at module top; `src/live_ai.py`
imports `src.tools` at its own top, so starting the app pulls the entire stack even for a
text-only session.

**Expected behaviour.**

- Keep at module top only: `re`, `platform`, `subprocess`, `requests`, `pandas as pd`,
  `pathlib.Path`, `datetime/timezone`, `typing.Literal`, and the `dump_func_call` import
  (with the existing `try/except` fallback).
- Remove from module top: `cv2`, `yfinance`, `matplotlib`/`pyplot`/`mdates`,
  `BeautifulSoup`, all `selenium.*`, `ChromeDriverManager`.
- Move each import into the function that needs it:

  | Symbol | Import inside |
  |--------|---------------|
  | `cv2` | `capture_image` |
  | `yf` | `fetch_financial_data` |
  | `matplotlib`, `plt`, `mdates` | `save_financial_chart` (call `matplotlib.use("Agg")` there before importing `pyplot`) |
  | `BeautifulSoup` | `_fetch_soup` |
  | `selenium.*`, `ChromeDriverManager` | `_create_chrome_driver` / `_collect_results` |

- Add `__all__` listing the public tools (see T9).

**Files.** `src/tools.py`; new `tests/test_tools.py`.

**Tests.** `tests/test_tools.py` begins with:

```python
from pathlib import Path

import pandas as pd
import pytest

import src.tools as tools


def test_import_tools_is_lazy():
    import sys

    for heavy in ("cv2", "selenium", "yfinance", "matplotlib", "bs4"):
        assert heavy not in sys.modules, f"{heavy} imported at module load"


def test_public_tools_are_callable():
    for name in ("system_info", "read_text_file", "write_text_file", "run_command",
                 "web_search", "capture_image", "fetch_financial_data",
                 "save_financial_chart", "search_in_file", "search_pattern_in_files",
                 "replace_string_in_file"):
        assert callable(getattr(tools, name))
```

The lazy-import assertion is meaningful only if no earlier test imported the heavy modules.
Because pytest may run tests in one process, place `test_import_tools_is_lazy` in its own
file-level run and, if needed, drop modules first:

```python
def test_import_tools_is_lazy():
    import importlib
    import sys

    for heavy in ("cv2", "selenium", "yfinance", "matplotlib", "bs4"):
        sys.modules.pop(heavy, None)
    for mod in [m for m in sys.modules if m == "src.tools"]:
        del sys.modules[mod]
    importlib.import_module("src.tools")
    for heavy in ("cv2", "selenium", "yfinance", "matplotlib", "bs4"):
        assert heavy not in sys.modules, f"{heavy} imported at module load"
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** The lazy-import test passes and the existing suite stays green.

---

## T1 — Fix `ValidYF` and test the read-only tools

**Goal.** Correct the interval typo and lock down the trivial, already-working tools.

**Current behaviour (`ValidYF.interval`, line 36).**

```python
interval = Literal['1m', '2m', ' 5m', ...]   # ' 5m' has a leading space
```

**Expected behaviour.** Change `' 5m'` to `'5m'`; leave the rest. Also add a basic test
group for `system_info`, `current_date_time`, `get_working_directory` and
`read_text_file`.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
def test_validyf_interval_has_no_leading_space():
    assert " 5m" not in tools.ValidYF.interval.__args__
    assert "5m" in tools.ValidYF.interval.__args__


def test_system_info_mentions_platform():
    assert isinstance(tools.system_info(), str) and tools.system_info()


def test_current_date_time_is_a_string():
    assert isinstance(tools.current_date_time(), str)


def test_get_working_directory_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(tools.get_working_directory()) == tmp_path.resolve()


def test_read_text_file_roundtrip(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    assert tools.read_text_file(str(target)) == "hello"


def test_read_text_file_missing_returns_error(tmp_path):
    assert "Unable to read" in tools.read_text_file(str(tmp_path / "nope.txt"))
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** All tests pass.

---

## T2 — Fix and extend `write_text_file`

**Goal.** Make the existence check explicit, create parent directories, support overwrite.

**Current behaviour (`write_text_file`, line 84).**

```python
if dest_path.exists and dest_path.is_file():   # .exists is a bound method (always truthy)
    return "File already exists."
```

The check works only by accident (`is_file()` gates it), does not reject directories, does
not create missing parents, and has no way to overwrite on purpose.

**Expected behaviour.**

```python
def write_text_file(file: str, content: str, overwrite: bool = False) -> str:
    dest_path = Path(file).resolve()
    if dest_path.exists():
        if dest_path.is_dir():
            return f"Error: '{dest_path}' is a directory."
        if not overwrite:
            return "File already exists."
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    return f"File created at '{dest_path}'"
```

Default `overwrite=False` preserves today's refuse-to-overwrite behaviour.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
def test_write_text_file_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "dir" / "a.txt"
    out = tools.write_text_file(str(target), "body")
    assert target.read_text(encoding="utf-8") == "body"
    assert "File created" in out


def test_write_text_file_refuses_existing_by_default(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    assert tools.write_text_file(str(target), "new") == "File already exists."
    assert target.read_text(encoding="utf-8") == "old"


def test_write_text_file_overwrite_true(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    tools.write_text_file(str(target), "new", overwrite=True)
    assert target.read_text(encoding="utf-8") == "new"


def test_write_text_file_rejects_directory(tmp_path):
    assert "is a directory" in tools.write_text_file(str(tmp_path), "x")
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** All four tests pass.

---

## T3 — Fix `search_in_file` and `replace_string_in_file`

**Goal.** Read errors must surface, and replacement counts must match literal replacements.

**Current behaviour.**
- `search_in_file` (line 326): `f"Error: '{e}'"` is dropped; the function then reports
  "Did not find pattern".
- `replace_string_in_file` (line 393): `len(re.findall(old_string, content))` treats
  `old_string` as a regex while `content.replace` is literal. For `old_string="a.b"` on
  `"a.b aXb"` it counts 2 but replaces 1.

**Expected behaviour.**

- In `search_in_file`, return the error:

  ```python
  except Exception as e:
      return f"Error reading '{file_path}': {e}"
  ```

- In `replace_string_in_file`, count literally:

  ```python
  count = content.count(old_string)
  ```

  Keep the existing argument order `(path, old_string, new_string)` and the message format.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
def test_search_in_file_finds_lines(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nneedle\nbeta needle\n", encoding="utf-8")
    out = tools.search_in_file("needle", str(target))
    assert "2" in out and "3" in out


def test_search_in_file_missing(tmp_path):
    assert "does not exist" in tools.search_in_file("x", str(tmp_path / "nope.txt"))


def test_replace_string_counts_literally(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("a.b aXb", encoding="utf-8")
    out = tools.replace_string_in_file(str(target), "a.b", "Z")
    assert "1" in out
    assert target.read_text(encoding="utf-8") == "Z aXb"


def test_replace_string_missing_file(tmp_path):
    assert "not found" in tools.replace_string_in_file(
        str(tmp_path / "nope.txt"), "a", "b"
    )
```

If a genuine read error is hard to trigger portably, monkeypatch `Path.read_text` to raise
and assert the returned string starts with `"Error reading"`.

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** All tests pass.

---

## T4 — Harden `run_command` and `exit_gracefully`

**Goal.** Correct argument handling, no deadlock, a timeout, an optional `cwd`, and a
deterministic exit.

**Current behaviour (lines 108-131, 39-45).**

```python
cmd_list = cmd.split()
process = subprocess.Popen(cmd_list, ..., shell=True)   # split + shell conflict
rcode = process.wait()                                   # can deadlock on full pipes
stdout, stderr = process.communicate()
...
exit(0)                                                  # interactive helper
```

**Expected behaviour.**

```python
def run_command(cmd: str, timeout_s: float = 30.0, cwd: str | None = None) -> str:
    """Run a shell command, returning stdout, stderr, or a timeout message."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(cwd) if cwd is not None else Path.cwd(),
            text=True,
            shell=True,
        )
    except Exception as exc:
        return f"Error starting command: {exc}"
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return f"Command timed out after {timeout_s}s: {stderr}"
    if process.returncode != 0:
        return f"Return code [{process.returncode}]: {stderr}"
    return stdout


def exit_gracefully() -> None:
    """Exit the program gracefully (raises SystemExit(0))."""
    print("\nExiting gracefully. Bye! 👋")
    raise SystemExit(0)
```

Passing the command as a single string with `shell=True` preserves pipes/quotes that the
old `split()` broke.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
import subprocess


def test_run_command_returns_output():
    out = tools.run_command(f'{sys.executable} -c "print(42)"')
    assert out.strip() == "42"


def test_run_command_nonzero_returns_code():
    out = tools.run_command(f'{sys.executable} -c "raise SystemExit(3)"')
    assert "Return code [3]" in out


def test_run_command_timeout(monkeypatch):
    class FakeProcess:
        returncode = None
        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("cmd", timeout)
            return "", ""
        def kill(self):
            FakeProcess.killed = True

    FakeProcess.killed = False
    monkeypatch.setattr(tools.subprocess, "Popen", lambda *a, **k: FakeProcess())
    out = tools.run_command("sleep 100", timeout_s=0.5)
    assert "timed out" in out and FakeProcess.killed is True


def test_run_command_honours_cwd(tmp_path):
    out = tools.run_command(f'{sys.executable} -c "import os;print(os.getcwd())"', cwd=tmp_path)
    assert str(tmp_path) in out


def test_exit_gracefully_raises_system_exit():
    with pytest.raises(SystemExit) as exc:
        tools.exit_gracefully()
    assert exc.value.code == 0
```

`sys` and `pytest` must be imported at the top of `tests/test_tools.py`.

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** All tests pass; no `cmd.split(` remains in `run_command`.

---

## T5 — Share and test the webpage helpers

**Goal.** One fetch/parse path for `webpage_headlines` and `webpage_content`, unit-tested
against a fake HTTP response.

**Current behaviour (lines 133-170).** Both functions repeat the `requests.get` +
`raise_for_status` + `BeautifulSoup` + filter loop; the fetch error path is duplicated.

**Expected behaviour.** Add:

```python
def _fetch_soup(url: str, timeout_s: int = 10):
    """Fetch a URL and return a BeautifulSoup document, or an error string."""
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
    except Exception as exc:
        return f"Error fetching URL: {exc}"
    return BeautifulSoup(response.text, "html.parser")


def extract_text(soup, tags, min_len: int = 20) -> list[str]:
    """Return stripped text of matching tags longer than min_len."""
    return [
        text
        for tag in soup.find_all(tags)
        if len(text := tag.get_text(strip=True)) > min_len
    ]


def webpage_headlines(url="https://www.ansa.it/english/", relevant_text_length=20, timeout_s=10):
    soup = _fetch_soup(url, timeout_s)
    if isinstance(soup, str):
        return soup
    return f"\nFrom [{url}]\n" + "\n".join(extract_text(soup, ["h1", "h2"], relevant_text_length))


def webpage_content(url, relevant_text_length=20, timeout_s=10):
    soup = _fetch_soup(url, timeout_s)
    if isinstance(soup, str):
        return soup
    return f"\nFrom [{url}]\n" + "\n".join(extract_text(soup, ["p"], relevant_text_length))
```

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.** Monkeypatch `tools.requests.get`; never touch the network.

```python
from types import SimpleNamespace


class FakeResponse:
    def __init__(self, text, status=200):
        self.text = text
        self._status = status
    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


def test_webpage_headlines(monkeypatch):
    html = "<h1>Big headline number one</h1><h2>small</h2><h1>Another long headline here</h1>"
    monkeypatch.setattr(tools.requests, "get", lambda url, timeout: FakeResponse(html))
    out = tools.webpage_headlines("http://x", relevant_text_length=5)
    assert "Big headline number one" in out and "Another long headline here" in out
    assert "small" not in out


def test_webpage_content(monkeypatch):
    html = "<p>a paragraph long enough here</p><p>tiny</p>"
    monkeypatch.setattr(tools.requests, "get", lambda url, timeout: FakeResponse(html))
    out = tools.webpage_content("http://x", relevant_text_length=5)
    assert "a paragraph long enough here" in out and "tiny" not in out


def test_webpage_fetch_error(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda url, timeout: FakeResponse("", status=500))
    assert "Error fetching URL" in tools.webpage_content("http://x")
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** Both functions go through `_fetch_soup` and all tests pass without network.

---

## T6 — Make `web_search` testable

**Goal.** Separate browser automation from result formatting and allow an injected driver.

**Current behaviour (lines 172-202).** `web_search` creates Chrome and downloads the driver
inline; the result formatting is entangled with Selenium.

**Expected behaviour.**

```python
def _create_chrome_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--incognito")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(width=300, height=200)
    return driver


def _collect_results(driver, query: str, timeout_s: float) -> list[str]:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver.get("https://duckduckgo.com/")
    wait = WebDriverWait(driver, timeout_s)
    search_bar = wait.until(EC.element_to_be_clickable((By.NAME, "q")))
    search_bar.send_keys(query)
    search_bar.submit()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article")))
    return [el.text.strip() for el in driver.find_elements(By.CSS_SELECTOR, "article") if el.text.strip()]


def format_search_results(results: list[str], n_results: int = 5) -> str:
    out = ["\nSearch Results:\n"]
    for i, item in enumerate(results[:n_results], 1):
        out.append(f"{i}) {item.replace(' › ', '/')}")
        out.append("=" * 50)
    return "\n".join(out) + "\n"


def web_search(query: str, n_results: int = 5, timeout_s: float = 10.0) -> str:
    """Search the web. `_create_chrome_driver`/`_collect_results` are the test seams."""
    driver = _create_chrome_driver()
    try:
        results = _collect_results(driver, query, timeout_s)
        return format_search_results(results, n_results)
    except Exception as exc:
        return f"Error: {exc!r}"
    finally:
        driver.quit()
```

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
class FakeDriver:
    def __init__(self):
        self.quit_calls = 0
    def quit(self):
        self.quit_calls += 1


def test_format_search_results_limits_and_formats():
    out = tools.format_search_results(["a › b", "c", "d"], n_results=2)
    assert "1) a/b" in out and "2) c" in out and "3)" not in out


def test_web_search_formats_results(monkeypatch):
    monkeypatch.setattr(tools, "_create_chrome_driver", lambda: FakeDriver())
    monkeypatch.setattr(tools, "_collect_results", lambda driver, query, timeout: ["a › b"])
    out = tools.web_search("q", n_results=1)
    assert "a/b" in out


def test_web_search_quits_owned_driver(monkeypatch):
    fake = FakeDriver()
    monkeypatch.setattr(tools, "_create_chrome_driver", lambda: fake)
    monkeypatch.setattr(tools, "_collect_results", lambda driver, query, timeout: ["a"])
    assert "1) a" in tools.web_search("q")
    assert fake.quit_calls == 1


def test_web_search_reports_errors(monkeypatch):
    monkeypatch.setattr(tools, "_create_chrome_driver", lambda: FakeDriver())
    def boom(driver, query, timeout):
        raise RuntimeError("no chrome")
    monkeypatch.setattr(tools, "_collect_results", boom)
    assert "no chrome" in tools.web_search("q")
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** The formatting/lifecycle path is covered with no Selenium import during the
test.

---

## T7 — Financial helpers: `select_financial_columns`, `safe_filename`

**Goal.** Normalise price frames in a pure, tested function and make chart filenames safe.

**Current behaviour (lines 222-305).** `fetch_financial_data` builds
`target_cols = [(_date, '')] + [(rc, symbol) …]` and requires the MultiIndex layout;
`save_financial_chart` writes `f"{symbol}.jpg"` with no sanitising and no output directory
choice.

**Expected behaviour.**

```python
REQUIRED_FINANCIAL_COLUMNS = ("Date", "Open", "Close")


def select_financial_columns(data, symbol: str):
    """Return a frame with Date/Open/Close, or an error string."""
    if data is None or data.empty:
        return f"No data found for {symbol}."
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(level[0]) for level in frame.columns]
    date_col = "Date" if "Date" in frame.columns else (
        "Datetime" if "Datetime" in frame.columns else None
    )
    if date_col is None:
        return "Date is missing"
    frame = frame.rename(columns={date_col: "Date"})
    missing = [c for c in REQUIRED_FINANCIAL_COLUMNS if c not in frame.columns]
    if missing:
        return f"Not all the required columns {list(REQUIRED_FINANCIAL_COLUMNS)} are available."
    frame = frame[list(REQUIRED_FINANCIAL_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    if getattr(frame["Date"].dt, "tz", None) is not None:
        frame["Date"] = frame["Date"].dt.tz_localize(None)
    return frame


def safe_filename(symbol: str) -> str:
    """Turn '^GSPC' / 'BRK/B' into a filesystem-safe stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("._")
    return cleaned or "symbol"
```

- `fetch_financial_data` calls `yf.download(...)` (lazy import) then delegates to
  `select_financial_columns`.
- `save_financial_chart(symbol, interval="1mo", period="1y", out_dir=None)` uses
  `safe_filename(symbol)` and `Path(out_dir or Path.cwd()) / f"{stem}.jpg"`, creating
  `out_dir` if needed.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.** Build frames directly; never call `yfinance`.

```python
def test_select_flat_columns():
    frame = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Open": [1.0, 2.0],
        "Close": [1.5, 2.5],
        "High": [9, 9],
    })
    out = tools.select_financial_columns(frame, "AAPL")
    assert list(out.columns) == ["Date", "Open", "Close"]
    assert out["Date"].dt.tz is None


def test_select_multiindex_columns():
    cols = pd.MultiIndex.from_tuples(
        [("Date", ""), ("Open", "AAPL"), ("Close", "AAPL"), ("High", "AAPL")]
    )
    frame = pd.DataFrame(
        [["2024-01-01", 1.0, 1.5, 9]], columns=cols
    )
    out = tools.select_financial_columns(frame, "AAPL")
    assert list(out.columns) == ["Date", "Open", "Close"]


def test_select_empty_returns_message():
    assert "No data found" in tools.select_financial_columns(pd.DataFrame(), "AAPL")


def test_select_missing_columns_returns_message():
    frame = pd.DataFrame({"Date": ["2024-01-01"], "Open": [1.0]})
    assert "required columns" in tools.select_financial_columns(frame, "AAPL")


@pytest.mark.parametrize("symbol,expected", [
    ("^GSPC", "GSPC"),
    ("BRK/B", "BRK_B"),
    ("AAPL", "AAPL"),
])
def test_safe_filename(symbol, expected):
    assert tools.safe_filename(symbol) == expected
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** Both flat and MultiIndex frames reduce correctly and `safe_filename`
produces a filesystem-safe stem.

---

## T8 — `normalize_extensions` and `search_pattern_in_files`

**Goal.** Accept `py` and `.py` interchangeably and document the exclusion rules.

**Current behaviour (lines 344-345).** The parsed strings keep their leading dot, so
`include_extensions="py"` never matches `file_path.suffix == ".py"`.

**Expected behaviour.**

```python
def normalize_extensions(value: str) -> list[str]:
    """Parse 'py, .MD' -> ['.py', '.md'] (lowercase, leading dot)."""
    result = []
    for item in (value or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        result.append(item if item.startswith(".") else "." + item)
    return result
```

In `search_pattern_in_files`, replace the inline parsing with `normalize_extensions(...)`
and keep the rest. Document in the docstring that `exclude_glob=".*"` skips hidden entries.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
def test_normalize_extensions():
    assert tools.normalize_extensions("py, .MD ,txt") == [".py", ".md", ".txt"]
    assert tools.normalize_extensions("") == []


def test_search_pattern_in_files_filters_extensions(tmp_path):
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
    out = tools.search_pattern_in_files("needle", str(tmp_path), include_extensions="py")
    assert "a.py" in out and "b.txt" not in out


def test_search_pattern_in_files_excludes_extension(tmp_path):
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
    out = tools.search_pattern_in_files("needle", str(tmp_path), exclude_extensions=".txt")
    assert "a.py" in out and "b.txt" not in out


def test_search_pattern_in_files_missing_dir(tmp_path):
    assert "does not exist" in tools.search_pattern_in_files("x", str(tmp_path / "nope"))
```

**Validation.** `python -m pytest tests/test_tools.py -q`

**Done when.** All tests pass and both `"py"` and `".py"` select Python files.

---

## T9 — `capture_image` seam, typing/cleanup, acceptance

**Goal.** Make the camera path injectable, add `__all__`, tidy types/docstrings, and prove
the module works end to end with fakes.

**Current behaviour (`capture_image`, lines 204-219).** `cv2.VideoCapture` and `cv2.imwrite`
are hard-wired, so no branch can be tested without hardware.

**Expected behaviour.**

```python
def capture_image(out_file: str = CAPTURED_IMAGE, camera_index: int = 0) -> str:
    """Capture one frame and save it. `_VIDEO_CAPTURE_FACTORY`/`_IMAGE_WRITER` are the test seams."""
    capture_factory = _VIDEO_CAPTURE_FACTORY
    imwrite = _IMAGE_WRITER
    if capture_factory is None or imwrite is None:
        import cv2
        capture_factory = capture_factory or cv2.VideoCapture
        imwrite = imwrite or cv2.imwrite

    camera = capture_factory(camera_index)
    if not camera.isOpened():
        return f"Webcam '{camera_index}' not available"
    ok, frame = camera.read()
    camera.release()
    if not ok:
        return "Unable to capture frame"
    if not imwrite(out_file, frame):
        return f"Unable to save image '{out_file}'"
    return str(Path(out_file).resolve())
```

Add at module scope:

```python
__all__ = [
    "ValidYF", "exit_gracefully", "system_info", "current_date_time",
    "read_text_file", "write_text_file", "get_working_directory", "run_command",
    "webpage_headlines", "webpage_content", "web_search", "capture_image",
    "fetch_financial_data", "save_financial_chart", "search_in_file",
    "search_pattern_in_files", "replace_string_in_file",
]
```

Also add concise docstrings and return annotations where missing, and confirm no
`sys.exit(`/`exit(` remains.

**Files.** `src/tools.py`; `tests/test_tools.py`.

**Tests.**

```python
class FakeCamera:
    def __init__(self, opened=True, ok=True):
        self._opened, self._ok = opened, ok
        self.released = 0
    def isOpened(self):
        return self._opened
    def read(self):
        return self._ok, object()
    def release(self):
        self.released += 1


def test_capture_image_success(tmp_path, monkeypatch):
    out = tmp_path / "shot.jpg"
    camera = FakeCamera()
    monkeypatch.setattr(tools, "_VIDEO_CAPTURE_FACTORY", lambda i: camera)
    monkeypatch.setattr(tools, "_IMAGE_WRITER", lambda path, frame: True)
    result = tools.capture_image(out_file=str(out))
    assert result == str(out.resolve()) and camera.released == 1


def test_capture_image_no_webcam(monkeypatch):
    monkeypatch.setattr(tools, "_VIDEO_CAPTURE_FACTORY", lambda i: FakeCamera(opened=False))
    monkeypatch.setattr(tools, "_IMAGE_WRITER", lambda p, f: True)
    assert "not available" in tools.capture_image()


def test_capture_image_save_failure(monkeypatch):
    monkeypatch.setattr(tools, "_VIDEO_CAPTURE_FACTORY", lambda i: FakeCamera())
    monkeypatch.setattr(tools, "_IMAGE_WRITER", lambda p, f: False)
    assert "Unable to save" in tools.capture_image()


def test_module_exports_public_tools():
    for name in tools.__all__:
        assert hasattr(tools, name)


def test_source_has_no_bare_exit():
    import pathlib
    text = pathlib.Path(tools.__file__).read_text(encoding="utf-8")
    assert "exit(0)" not in text
```

**Validation.** `python -m pytest tests/ -q` (full suite).

**Done when.** Full suite green and `capture_image` covers success, no-camera and
save-failure branches with fakes.

---

## 3. Global acceptance criteria

Run from the repo root:

```
python -m pytest tests/ -q
```

Must pass. Additionally verify:

1. `python -c "import sys; import src.tools; print([m for m in ('cv2','selenium','yfinance','matplotlib','bs4') if m in sys.modules])"`
   prints `[]`.
2. `write_text_file` creates nested parent directories and refuses to
   overwrite unless `overwrite=True`.
3. `search_in_file` on an unreadable file returns an `Error reading` string, not
   "not found".
4. `replace_string_in_file("a.b", "Z")` on `"a.b aXb"` reports 1 replacement.
5. `run_command` preserves quoted arguments, honours `cwd`, and returns a timeout message
   instead of hanging.
6. `webpage_content` and `webpage_headlines` share `_fetch_soup` and return
   `Error fetching URL` on HTTP errors.
7. `web_search` with an injected driver never creates Chrome and never downloads a driver.
8. `select_financial_columns` handles both flat and MultiIndex frames and reports missing
   columns.
9. `search_pattern_in_files` accepts `py` and `.py`.
10. `capture_image` never imports `cv2` when factories are injected.
11. Every task above has at least one test that fails before its change and passes after.

---

## 4. Out of scope (do not implement under this plan)

- Replacing Selenium or yfinance with other libraries.
- Sandboxing or allow-listing shell commands (the agent is trusted in this project).
- New tools not already registered in `src/live_ai.py`'s tool list.
- Changing `dump_func_call` or the console output format.
- Async/concurrent tool execution or result caching.
- Any change to `Brain`/`Agent`, `ConversationSession`, audio capture/generation, or config
  keys other than none (tools are unaffected by config).
