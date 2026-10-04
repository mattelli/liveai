import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.tools as tools


# ------------------------------------------------------------------ T0 lazy imports


def test_import_tools_is_lazy():
    import importlib

    heavy = ("cv2", "selenium", "yfinance", "matplotlib", "bs4")
    for name in heavy:
        sys.modules.pop(name, None)
    sys.modules.pop("src.tools", None)
    importlib.import_module("src.tools")
    for name in heavy:
        assert name not in sys.modules, f"{name} imported at module load"


def test_public_tools_are_callable():
    for name in ("system_info", "read_text_file", "write_text_file", "run_command",
                 "web_search", "capture_image", "fetch_financial_data",
                 "save_financial_chart", "search_in_file", "search_pattern_in_files",
                 "replace_string_in_file"):
        assert callable(getattr(tools, name))


# ------------------------------------------------------------------ T1 read-only


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


# ------------------------------------------------------------------ T2 write_text_file


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


# ------------------------------------------------------------------ T3 search/replace


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


# ------------------------------------------------------------------ T4 run_command


def test_run_command_returns_output():
    out = tools.run_command(f'"{sys.executable}" -c "print(42)"')
    assert out.strip() == "42"


def test_run_command_nonzero_returns_code():
    out = tools.run_command(f'"{sys.executable}" -c "raise SystemExit(3)"')
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
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())
    out = tools.run_command("sleep 100", timeout_s=0.5)
    assert "timed out" in out and FakeProcess.killed is True


def test_run_command_honours_cwd(tmp_path):
    out = tools.run_command(
        f'"{sys.executable}" -c "import os;print(os.getcwd())"', cwd=str(tmp_path)
    )
    assert str(tmp_path) in out


def test_exit_gracefully_raises_system_exit():
    with pytest.raises(SystemExit) as exc:
        tools.exit_gracefully()
    assert exc.value.code == 0


# ------------------------------------------------------------------ T5 webpages


class FakeResponse:
    def __init__(self, text, status=200):
        self.text = text
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


def test_webpage_headlines(monkeypatch):
    html = (
        "<h1>Big headline number one</h1>"
        "<h2>small</h2>"
        "<h1>Another long headline here</h1>"
    )
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
    monkeypatch.setattr(
        tools.requests, "get", lambda url, timeout: FakeResponse("", status=500)
    )
    assert "Error fetching URL" in tools.webpage_content("http://x")


# ------------------------------------------------------------------ downloads


class FakeStreamResponse:
    """Streaming response double for ``requests.get(..., stream=True)``."""

    def __init__(self, chunks=(b"abc", b"def"), status=200, headers=None):
        self._chunks = list(chunks)
        self._status = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


def test_download_file_streams_content_from_url(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(chunks=(b"hello ", b"world")),
    )
    out = tools.download_file("https://host/pictures/pic.jpg", out_dir=str(tmp_path))
    target = tmp_path / "pic.jpg"
    assert target.read_bytes() == b"hello world"
    assert "pic.jpg" in out and "11 bytes" in out


def test_download_file_explicit_name_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(headers={"content-type": "video/mp4"}),
    )
    out = tools.download_file("https://host/film", out_dir=str(tmp_path), filename="movie.mp4")
    assert (tmp_path / "movie.mp4").exists() and "movie.mp4" in out

    again = tools.download_file("https://host/film", out_dir=str(tmp_path), filename="movie.mp4")
    assert again == "File already exists."

    replaced = tools.download_file(
        "https://host/film", out_dir=str(tmp_path), filename="movie.mp4", overwrite=True
    )
    assert "movie.mp4" in replaced


def test_download_file_content_disposition_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(
            headers={"content-disposition": 'attachment; filename="report.pdf"'}
        ),
    )
    tools.download_file("https://host/get?id=1", out_dir=str(tmp_path))
    assert (tmp_path / "report.pdf").exists()


def test_download_file_content_type_completes_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(headers={"content-type": "image/png"}),
    )
    tools.download_file("https://host/photo", out_dir=str(tmp_path))
    assert (tmp_path / "photo.png").exists()


def test_download_file_rfc5987_filename(tmp_path, monkeypatch):
    disposition = "attachment; filename*=UTF-8''na%C3%AFve.png"
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(
            headers={"content-disposition": disposition}
        ),
    )
    # RFC 5987 percent-escapes decode to "na\u00efve.png", then safe_filename
    # replaces the non-ASCII letter before the file is written.
    assert tools.web._suggested_name("https://h/x", {"content-disposition": disposition}) == "na\u00efve.png"
    tools.download_file("https://host/x", out_dir=str(tmp_path))
    assert (tmp_path / "na_ve.png").exists()


def test_download_file_sanitizes_traversal_names(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(
            headers={"content-disposition": 'attachment; filename="../../evil.exe"'}
        ),
    )
    out = tools.download_file("https://host/x", out_dir=str(tmp_path))
    saved = Path(out.split("'")[1])
    assert saved.parent == tmp_path.resolve()  # stayed inside out_dir
    assert saved.exists()


def test_download_file_http_error(monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get",
        lambda url, timeout, stream, headers=None: FakeStreamResponse(status=404),
    )
    assert "Error fetching URL" in tools.download_file("https://host/missing.png")


def test_download_file_network_error(monkeypatch):
    def boom(url, timeout, stream, headers=None):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(tools.requests, "get", boom)
    assert "Error fetching URL" in tools.download_file("https://host/x")


def test_suggested_name_variants():
    assert tools.web._suggested_name("https://h/dir/a%20b.png", {}) == "a b.png"
    assert tools.web._suggested_name("https://h/dir/file.bin", {"content-disposition": ""}) == "file.bin"
    assert tools.web._suggested_name("https://h/", {}) == ""  # caller falls back to timestamp
    assert tools.web._with_extension("photo", "image/jpeg;charset=utf8") == "photo.jpg"


# ------------------------------------------------------------------ archives


def make_zip(path, entries):
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def make_tar(path, entries, mode):
    import tarfile
    import io

    with tarfile.open(path, mode) as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            data = content.encode("utf-8")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_extract_archive_zip_with_nested_folders(tmp_path):
    source = tmp_path / "bundle.zip"
    make_zip(source, {"a.txt": "alpha", "sub/b.txt": "beta"})
    out = tools.extract_archive(str(source))
    dest = tmp_path / "bundle"
    assert dest.is_dir()
    assert (dest / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (dest / "sub" / "b.txt").read_text(encoding="utf-8") == "beta"
    assert f"to '{dest.resolve()}'" in out and "2 entries" in out


def test_extract_archive_zip_slip_is_rejected(tmp_path):
    import zipfile

    source = tmp_path / "evil.zip"
    with zipfile.ZipFile(source, "w") as archive:
        info = zipfile.ZipInfo("../evil.txt")
        archive.writestr(info, "boom")
    out = tools.extract_archive(str(source))
    assert "unsafe archive entry" in out
    assert not (tmp_path / "evil.txt").exists()


def test_extract_archive_tar_gz_and_plain_tar(tmp_path):
    tgz = tmp_path / "docs.tar.gz"
    make_tar(tgz, {"report.txt": "gz content"}, "w:gz")
    out = tools.extract_archive(str(tgz))
    assert (tmp_path / "docs" / "report.txt").read_text(encoding="utf-8") == "gz content"
    assert "1 entries" in out

    tar = tmp_path / "plain.tar"
    make_tar(tar, {"one.txt": "1", "two.txt": "2"}, "w")
    tools.extract_archive(str(tar))
    assert (tmp_path / "plain" / "two.txt").read_text(encoding="utf-8") == "2"


def test_extract_archive_tar_traversal_is_filtered(tmp_path):
    import tarfile
    import io

    source = tmp_path / "evil.tar"
    with tarfile.open(source, "w") as archive:
        info = tarfile.TarInfo("../evil.txt")
        data = b"boom"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    out = tools.extract_archive(str(source))
    assert "Error extracting" in out
    assert not (tmp_path / "evil.txt").exists()


def test_extract_archive_bare_gz(tmp_path):
    import gzip

    source = tmp_path / "notes.csv.gz"
    source.write_bytes(gzip.compress(b"col1,col2\n1,2\n"))
    # default: decompressed file lands beside the archive
    out = tools.extract_archive(str(source))
    assert (tmp_path / "notes.csv").read_text(encoding="utf-8") == "col1,col2\n1,2\n"
    assert "1 entries" in out

    again = tools.extract_archive(str(source))
    assert again == "File already exists."
    tools.extract_archive(str(source), overwrite=True)
    assert (tmp_path / "notes.csv").exists()


def test_extract_archive_bare_gz_out_dir(tmp_path):
    import gzip

    source = tmp_path / "notes.csv.gz"
    source.write_bytes(gzip.compress(b"data"))
    target_dir = tmp_path / "somewhere"
    tools.extract_archive(str(source), out_dir=str(target_dir))
    assert (target_dir / "notes.csv").read_text(encoding="utf-8") == "data"


def test_extract_archive_7z_requires_py7zr(tmp_path, monkeypatch):
    import builtins
    import sys

    source = tmp_path / "pack.7z"
    source.write_bytes(b"fake 7z")
    monkeypatch.setitem(sys.modules, "py7zr", None)

    real_import = builtins.__import__

    def no_py7zr(name, *args, **kwargs):
        if name == "py7zr":
            raise ImportError("no py7zr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_py7zr)
    out = tools.extract_archive(str(source))
    assert "pip install py7zr" in out


def test_extract_archive_unsupported_and_missing(tmp_path):
    missing = tmp_path / "missing.zip"
    assert "does not exist" in tools.extract_archive(str(missing))
    unsupported = tmp_path / "x.rar"
    unsupported.write_bytes(b"rar-ish")
    assert "unsupported archive type '.rar'" in tools.extract_archive(str(unsupported))
    (tmp_path / "folder.zip").mkdir()
    assert "is a directory" in tools.extract_archive(str(tmp_path / "folder.zip"))


def test_extract_archive_refuses_nonempty_destination(tmp_path):
    source = tmp_path / "again.zip"
    make_zip(source, {"a.txt": "alpha"})
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "existing.txt").write_text("keep me", encoding="utf-8")

    out = tools.extract_archive(str(source), out_dir=str(dest))
    assert "not empty" in out and "overwrite=True" in out
    assert (dest / "existing.txt").read_text(encoding="utf-8") == "keep me"

    out = tools.extract_archive(str(source), out_dir=str(dest), overwrite=True)
    assert (dest / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert "Extracted 1 entries" in out


def test_archive_stem_variants():
    assert tools.archives._archive_stem(Path("data.tar.gz")) == "data"
    assert tools.archives._archive_stem(Path("data.tgz")) == "data"
    assert tools.archives._archive_stem(Path("data.zip")) == "data"
    assert tools.archives._archive_stem(Path("logs.gz")) == "logs"
    assert tools.archives._archive_kind(Path("photos.tar.xz")) == "tar"
    assert tools.archives._archive_kind(Path("photos.rar")) == ""


# ------------------------------------------------------------------ zip_files


def test_zip_files_packs_folder_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("beta", encoding="utf-8")

    out = tools.zip_files(str(tmp_path), str(tmp_path / "bundle.zip"))
    archive = tmp_path / "bundle.zip"
    assert archive.exists()
    with __import__("zipfile").ZipFile(archive) as zf:
        names = zf.namelist()
        assert "a.txt" in names and "sub/b.txt" in names  # forward slashes
        assert zf.read("a.txt").decode("utf-8") == "alpha"
    assert f"to '{archive.resolve()}'" in out and "2 files" in out


def test_zip_files_extension_filters(tmp_path):
    txt_only = tmp_path / "txt_only"
    txt_only.mkdir()
    (txt_only / "keep.txt").write_text("t", encoding="utf-8")
    (txt_only / "skip.log").write_text("l", encoding="utf-8")
    (txt_only / "drop.bin").write_bytes(b"b")

    tools.zip_files(str(txt_only), str(tmp_path / "only_txt.zip"), include_extensions="txt")
    with __import__("zipfile").ZipFile(tmp_path / "only_txt.zip") as zf:
        assert zf.namelist() == ["keep.txt"]

    no_log = tmp_path / "no_log"
    no_log.mkdir()
    (no_log / "keep.txt").write_text("t", encoding="utf-8")
    (no_log / "skip.log").write_text("l", encoding="utf-8")
    (no_log / "drop.bin").write_bytes(b"b")

    tools.zip_files(str(no_log), str(tmp_path / "no_log.zip"), exclude_extensions="log, .bin")
    with __import__("zipfile").ZipFile(tmp_path / "no_log.zip") as zf:
        assert zf.namelist() == ["keep.txt"]


def test_zip_files_exclude_glob_skips_hidden(tmp_path):
    hidden = tmp_path / ".res"
    hidden.mkdir()
    (hidden / "cached.bin").write_bytes(b"x")
    (tmp_path / "visible.txt").write_text("v", encoding="utf-8")

    tools.zip_files(str(tmp_path), str(tmp_path / "default.zip"))
    with __import__("zipfile").ZipFile(tmp_path / "default.zip") as zf:
        assert zf.namelist() == ["visible.txt"]

    tools.zip_files(str(tmp_path), str(tmp_path / "all.zip"), exclude_glob=None)
    with __import__("zipfile").ZipFile(tmp_path / "all.zip") as zf:
        assert ".res/cached.bin" in zf.namelist()


def test_zip_files_overwrite_guard(tmp_path):
    (tmp_path / "one.txt").write_text("1", encoding="utf-8")
    archive = tmp_path / "out.zip"
    archive.write_bytes(b"stale")

    assert tools.zip_files(str(tmp_path), str(archive)) == "File already exists."
    assert archive.read_bytes() == b"stale"  # untouched

    out = tools.zip_files(str(tmp_path), str(archive), overwrite=True)
    assert "1 files" in out
    with __import__("zipfile").ZipFile(archive) as zf:
        assert zf.namelist() == ["one.txt"]


def test_zip_files_appends_zip_suffix(tmp_path):
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    tools.zip_files(str(tmp_path), str(tmp_path / "pack"))
    assert (tmp_path / "pack.zip").exists()


def test_zip_files_never_zips_itself(tmp_path):
    (tmp_path / "data.txt").write_text("d", encoding="utf-8")
    out = tools.zip_files(str(tmp_path), str(tmp_path / "self.zip"))  # archive inside source
    with __import__("zipfile").ZipFile(tmp_path / "self.zip") as zf:
        assert zf.namelist() == ["data.txt"]
    assert "1 files" in out


def test_zip_files_error_and_empty_cases(tmp_path):
    assert "does not exist" in tools.zip_files(str(tmp_path / "nope"), str(tmp_path / "a.zip"))
    single = tmp_path / "file.txt"
    single.write_text("x", encoding="utf-8")
    assert "is not a directory" in tools.zip_files(str(single), str(tmp_path / "b.zip"))

    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No files to zip" in tools.zip_files(str(empty), str(tmp_path / "c.zip"))


def test_zip_files_roundtrip_with_extract_archive(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.txt").write_text("first", encoding="utf-8")
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "two.txt").write_text("second", encoding="utf-8")

    tools.zip_files(str(tmp_path / "src"), str(tmp_path / "roundtrip"))
    out = tools.extract_archive(str(tmp_path / "roundtrip.zip"), out_dir=str(tmp_path / "restored"))
    assert "Extracted 2 entries" in out
    assert (tmp_path / "restored" / "one.txt").read_text(encoding="utf-8") == "first"
    assert (tmp_path / "restored" / "nested" / "two.txt").read_text(encoding="utf-8") == "second"


# ------------------------------------------------------------------ T6 web_search


class FakeDriver:
    def __init__(self):
        self.quit_calls = 0

    def quit(self):
        self.quit_calls += 1


def test_format_search_results_limits_and_formats():
    out = tools.format_search_results(["a › b", "c", "d"], n_results=2)
    assert "1) a/b" in out and "2) c" in out and "3)" not in out


def test_web_search_formats_results(monkeypatch):
    monkeypatch.setattr(tools.web, "_create_chrome_driver", lambda: FakeDriver())
    monkeypatch.setattr(tools.web, "_collect_results", lambda driver, query, timeout: ["a › b"])
    out = tools.web_search("q", n_results=1)
    assert "a/b" in out


def test_web_search_quits_owned_driver(monkeypatch):
    fake = FakeDriver()
    monkeypatch.setattr(tools.web, "_create_chrome_driver", lambda: fake)
    monkeypatch.setattr(tools.web, "_collect_results", lambda driver, query, timeout: ["a"])
    assert "1) a" in tools.web_search("q")
    assert fake.quit_calls == 1


def test_web_search_reports_errors(monkeypatch):
    monkeypatch.setattr(tools.web, "_create_chrome_driver", lambda: FakeDriver())

    def boom(driver, query, timeout):
        raise RuntimeError("no chrome")

    monkeypatch.setattr(tools.web, "_collect_results", boom)
    assert "no chrome" in tools.web_search("q")


# ------------------------------------------------------------------ T7 financial


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
    frame = pd.DataFrame([["2024-01-01", 1.0, 1.5, 9]], columns=cols)
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


# ------------------------------------------------------------------ T8 file search


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


# ------------------------------------------------------------------ T9 capture/cleanup


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
    monkeypatch.setattr(tools.camera, "_VIDEO_CAPTURE_FACTORY", lambda i: camera)
    monkeypatch.setattr(tools.camera, "_IMAGE_WRITER", lambda path, frame: True)
    result = tools.capture_image(out_file=str(out))
    assert result == str(out.resolve()) and camera.released == 1


def test_capture_image_no_webcam(monkeypatch):
    monkeypatch.setattr(tools.camera, "_VIDEO_CAPTURE_FACTORY", lambda i: FakeCamera(opened=False))
    monkeypatch.setattr(tools.camera, "_IMAGE_WRITER", lambda p, f: True)
    assert "not available" in tools.capture_image()


def test_capture_image_save_failure(monkeypatch):
    monkeypatch.setattr(tools.camera, "_VIDEO_CAPTURE_FACTORY", lambda i: FakeCamera())
    monkeypatch.setattr(tools.camera, "_IMAGE_WRITER", lambda p, f: False)
    assert "Unable to save" in tools.capture_image()


def test_module_exports_public_tools():
    for name in tools.__all__:
        assert hasattr(tools, name)


def test_source_has_no_bare_exit():
    text = Path(tools.__file__).read_text(encoding="utf-8")
    assert "exit(0)" not in text
