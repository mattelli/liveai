import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.tools as tools
from src.common import normalize_extensions, safe_filename


# ------------------------------------------------------------------ T0 lazy imports


def test_import_tools_is_lazy():
    import importlib

    heavy = ("cv2", "ddgs", "yfinance", "matplotlib", "bs4", "torch", "diffusers")
    for name in heavy:
        sys.modules.pop(name, None)
    sys.modules.pop("src.tools", None)
    importlib.import_module("src.tools")
    for name in heavy:
        assert name not in sys.modules, f"{name} imported at module load"


def test_public_tools_are_callable():
    for name in ("system_info", "read_text_file", "write_text_file", "run_command",
                 "web_search", "web_images", "web_videos", "web_news",
                 "web_books", "extract_webpage", "capture_image", "generate_image",
                 "unload_image_generator", "fetch_financial_data",
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


def test_read_text_file_window(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    out = tools.read_text_file(str(target), start_line=3, line_count=2)
    assert out.startswith("line3\nline4")
    assert "windowed" in out  # more lines remain -> marker guides the re-read


def test_read_text_file_window_exactly_to_end_has_no_marker(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    out = tools.read_text_file(str(target), start_line=9, line_count=2)
    assert out == "line9\nline10"  # window reaches EOF -> no marker


def test_read_text_file_window_to_end(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 6)), encoding="utf-8")
    out = tools.read_text_file(str(target), start_line=4)
    assert out == "line4\nline5"


def test_read_text_file_window_past_eof_clamps(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("one\ntwo", encoding="utf-8")
    assert tools.read_text_file(str(target), start_line=5, line_count=10) == ""


def test_read_text_file_window_marker_when_more_remain(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    out = tools.read_text_file(str(target), start_line=1, line_count=3)
    assert "line1\nline2\nline3" in out
    assert "windowed" in out and "10 total" in out


def test_read_text_file_number_lines(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    out = tools.read_text_file(str(target), number_lines=True)
    assert out == "1: alpha\n2: beta\n3: gamma"


def test_read_text_file_number_lines_with_window(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 13)), encoding="utf-8")
    out = tools.read_text_file(str(target), start_line=10, line_count=3,
                               number_lines=True)
    assert out == "10: line10\n11: line11\n12: line12"


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


def test_write_text_file_append_creates_missing_file(tmp_path):
    target = tmp_path / "a.txt"
    out = tools.write_text_file(str(target), "first", append=True)
    assert target.read_text(encoding="utf-8") == "first"
    assert "File created" in out


def test_write_text_file_append_adds_to_end(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    out = tools.write_text_file(str(target), "new", append=True)
    assert target.read_text(encoding="utf-8") == "oldnew"
    assert "Appended" in out


def test_write_text_file_append_twice(tmp_path):
    target = tmp_path / "a.txt"
    tools.write_text_file(str(target), "a\n", append=True)
    tools.write_text_file(str(target), "b\n", append=True)
    assert target.read_text(encoding="utf-8") == "a\nb\n"


def test_write_text_file_append_ignores_overwrite_flag(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    tools.write_text_file(str(target), "new", overwrite=True, append=True)
    assert target.read_text(encoding="utf-8") == "oldnew"


def test_write_text_file_readback_verification(tmp_path):
    """The tool verifies the write by reading the file back."""
    target = tmp_path / "a.txt"
    out = tools.write_text_file(str(target), "payload")
    assert "(verified)" in out
    assert target.read_text(encoding="utf-8") == "payload"


def test_write_text_file_readback_detects_mismatch(tmp_path, monkeypatch):
    """A damaged on-disk result is reported as an error, not success."""
    target = tmp_path / "a.txt"
    real_open = open

    def fake_open(file, mode="r", **kwargs):
        handle = real_open(file, mode, **kwargs)
        if "w" in mode and str(file) == str(target):
            class Damaged:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def write(self, text):
                    real_open(target, "w", encoding="utf-8").write(
                        text.replace("payload", "payl0ad"))
            return Damaged()
        return handle

    monkeypatch.setattr("builtins.open", fake_open)
    out = tools.write_text_file(str(target), "payload")
    assert "verification failed" in out


def test_read_text_file_checksum_off_by_default(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    assert tools.read_text_file(str(target)) == "hello"


def test_read_text_file_checksum_full_read(tmp_path):
    import hashlib
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    digest = hashlib.sha1(b"hello").hexdigest()[:12]
    out = tools.read_text_file(str(target), checksum=True)
    assert out == f"hello\n[sha1:{digest}]"


def test_read_text_file_checksum_windowed_read(tmp_path):
    import hashlib
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    digest = hashlib.sha1(b"alpha\nbeta").hexdigest()[:12]
    out = tools.read_text_file(str(target), start_line=1, line_count=2,
                               checksum=True)
    assert out.endswith(f"[sha1:{digest}]")


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


# ------------------------------------------------------------- spawn_process


def test_spawn_process_returns_pid():
    out = tools.spawn_process(f'"{sys.executable}" -c "import time; time.sleep(2)"')
    match = re.search(r"PID (\d+)", out)
    assert match, out
    assert int(match.group(1)) > 0


def test_spawn_process_returns_immediately():
    """The call returns while the spawned process is still running."""
    import time

    start = time.monotonic()
    out = tools.spawn_process(f'"{sys.executable}" -c "import time; time.sleep(5)"')
    elapsed = time.monotonic() - start
    assert "PID" in out
    assert elapsed < 2.0  # no wait for the 5s child


def test_spawn_process_error_message(monkeypatch):
    def exploding_popen(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "Popen", exploding_popen)
    out = tools.spawn_process("anything")
    assert "Error spawning process" in out and "cannot spawn" in out


def test_spawn_process_honours_cwd(tmp_path):
    """The child runs with the given working directory."""
    marker = tmp_path / "cwd.txt"
    out = tools.spawn_process(
        f'"{sys.executable}" -c "import os; open(r\'{marker}\', \'w\').write(os.getcwd())"',
        cwd=str(tmp_path),
    )
    assert "PID" in out
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists()
    assert marker.read_text(encoding="utf-8").strip() == str(tmp_path)


def test_spawn_process_is_exported_and_listed():
    assert callable(tools.spawn_process)
    import src.live_ai as live_ai
    from types import SimpleNamespace

    # registered in the agent's default tool set and the system topic
    offline = SimpleNamespace(model_info=lambda: "")
    tool_names = [getattr(t, "__name__", "") for t in live_ai._default_tools(offline)]
    assert "spawn_process" in tool_names
    assert "spawn_process" in live_ai._TOOL_TOPICS["system"]


def test_exit_gracefully_sets_quit_flag():
    from src.common import quit_requested, request_quit, reset_quit
    reset_quit()
    try:
        result = tools.exit_gracefully()
        assert quit_requested() is True
        assert "session will close" in result
    finally:
        reset_quit()  # do not leak the flag into other tests


def test_exit_gracefully_does_not_raise():
    from src.common import reset_quit
    reset_quit()
    try:
        tools.exit_gracefully()  # no SystemExit: the turn must complete
    finally:
        reset_quit()


# ------------------------------------------------------------------ T5 extract_webpage


def test_extract_webpage_formats_content(monkeypatch):
    fake = FakeDDGS(results="# Example Domain\n\nThis domain is for use in examples")
    _patch_ddgs(monkeypatch, fake)
    out = tools.web.extract_webpage("https://example.com")
    assert out.startswith("\nFrom [https://example.com]\n# Example Domain")
    assert fake.calls[0] == ("extract", {"url": "https://example.com", "fmt": "text_markdown"})


def test_extract_webpage_passes_fmt(monkeypatch):
    fake = FakeDDGS(results="plain text")
    _patch_ddgs(monkeypatch, fake)
    tools.web.extract_webpage("https://example.com", fmt="text_plain")
    assert fake.calls[0][1]["fmt"] == "text_plain"


def test_extract_webpage_truncates_long_content(monkeypatch):
    _patch_ddgs(monkeypatch, FakeDDGS(results="x" * 9000))
    out = tools.web.extract_webpage("https://example.com")
    assert len(out.split("]\n", 1)[1]) == 4000


def test_extract_webpage_rejects_raw_bytes_fmt(monkeypatch):
    fake = FakeDDGS(results=b"raw")
    _patch_ddgs(monkeypatch, fake)
    out = tools.web.extract_webpage("https://example.com", fmt="content")
    assert "Error" in out and fake.calls == []


def test_extract_webpage_reports_errors(monkeypatch):
    _patch_ddgs(monkeypatch, FakeDDGS(error=RuntimeError("dns fail")))
    assert "dns fail" in tools.web.extract_webpage("https://example.com")


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


class FakeDDGS:
    """Canned ``DDGS`` double: returns ``results`` from every search call."""

    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.calls = []

    def _respond(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if self.error:
            raise self.error
        return self.results

    def text(self, query, **kwargs):
        return self._respond("text", query=query, **kwargs)

    def images(self, query, **kwargs):
        return self._respond("images", query=query, **kwargs)

    def videos(self, query, **kwargs):
        return self._respond("videos", query=query, **kwargs)

    def news(self, query, **kwargs):
        return self._respond("news", query=query, **kwargs)

    def books(self, query, **kwargs):
        return self._respond("books", query=query, **kwargs)

    def extract(self, url, fmt="text_markdown"):
        self.calls.append(("extract", {"url": url, "fmt": fmt}))
        if self.error:
            raise self.error
        return {"url": url, "content": self.results}


def _patch_ddgs(monkeypatch, fake):
    """Patch the ``_ddgs_client`` seam in ``src.tools.web`` with ``fake``."""
    monkeypatch.setattr(tools.web, "_ddgs_client", lambda timeout_s=10.0: fake)


def test_format_search_results_limits_and_formats():
    results = [
        {"title": "a", "href": "b", "body": "c"},
        {"title": "d"},
        {"title": "e"},
        {"title": "f"},
    ]
    out = tools.format_search_results(results, n_results=2)
    assert "1) title: a | href: b | body: c" in out
    assert "2) title: d" in out
    assert "3)" not in out


def test_web_search_formats_results(monkeypatch):
    fake = FakeDDGS(results=[{"title": "t", "href": "h", "body": "b"}])
    _patch_ddgs(monkeypatch, fake)
    out = tools.web_search("q", n_results=1)
    assert "1) title: t | href: h | body: b" in out


def test_web_search_passes_parameters_to_ddgs(monkeypatch):
    fake = FakeDDGS(results=[{"title": "t"}])
    _patch_ddgs(monkeypatch, fake)
    tools.web_search("q", n_results=2, region="uk-en", safesearch="off", timelimit="m")
    method, kwargs = fake.calls[0]
    assert method == "text"
    assert kwargs["max_results"] == 2
    assert kwargs["region"] == "uk-en"
    assert kwargs["safesearch"] == "off" and kwargs["timelimit"] == "m"


def test_web_search_empty_results(monkeypatch):
    _patch_ddgs(monkeypatch, FakeDDGS(results=[]))
    assert "No results" in tools.web_search("q")


def test_web_search_reports_errors(monkeypatch):
    _patch_ddgs(monkeypatch, FakeDDGS(error=RuntimeError("no network")))
    assert "no network" in tools.web_search("q")


def test_web_images_formats_results(monkeypatch):
    fake = FakeDDGS(results=[{
        "title": "sun", "image": "http://i/x.jpg", "thumbnail": "http://t",
        "width": 10, "height": 20, "source": "Bing",
    }])
    _patch_ddgs(monkeypatch, fake)
    out = tools.web.web_images("sun", n_results=1)
    assert "1) title: sun | image: http://i/x.jpg | thumbnail: http://t" in out
    assert "width: 10 | height: 20 | source: Bing" in out


def test_web_images_passes_filters(monkeypatch):
    fake = FakeDDGS(results=[{"title": "t"}])
    _patch_ddgs(monkeypatch, fake)
    tools.web.web_images("q", size="Large", color="Red", type_image="photo",
                     layout="Wide", license_image="any")
    method, kwargs = fake.calls[0]
    assert method == "images"
    assert kwargs["size"] == "Large" and kwargs["color"] == "Red"
    assert kwargs["type_image"] == "photo" and kwargs["layout"] == "Wide"
    assert kwargs["license_image"] == "any"


def test_web_videos_formats_results(monkeypatch):
    fake = FakeDDGS(results=[{
        "title": "v", "content": "http://yt/1", "duration": "8:22",
        "publisher": "YouTube", "uploader": "Sun TV",
    }])
    _patch_ddgs(monkeypatch, fake)
    out = tools.web.web_videos("v", n_results=1)
    assert "1) title: v | content: http://yt/1 | duration: 8:22" in out
    assert "publisher: YouTube | uploader: Sun TV" in out


def test_web_news_formats_results(monkeypatch):
    fake = FakeDDGS(results=[{
        "date": "2026-01-02", "title": "n", "body": "b",
        "url": "http://u", "source": "s",
    }])
    _patch_ddgs(monkeypatch, fake)
    out = tools.web.web_news("n", n_results=1)
    assert "1) date: 2026-01-02 | title: n | body: b" in out


def test_web_books_formats_results(monkeypatch):
    fake = FakeDDGS(results=[{
        "title": "The Sea-Wolf", "author": "Jack London", "publisher": "DigiCat",
        "info": "English [en], .epub", "url": "http://u",
    }])
    _patch_ddgs(monkeypatch, fake)
    out = tools.web.web_books("sea wolf", n_results=1)
    assert "1) title: The Sea-Wolf | author: Jack London" in out


def test_web_news_empty_and_errors(monkeypatch):
    _patch_ddgs(monkeypatch, FakeDDGS(results=[]))
    assert "No results" in tools.web.web_news("q")
    _patch_ddgs(monkeypatch, FakeDDGS(error=RuntimeError("throttled")))
    assert "throttled" in tools.web.web_news("q")


def test_remaining_web_kinds_report_errors(monkeypatch):
    for tool_call in (tools.web.web_images, tools.web.web_videos, tools.web.web_books):
        _patch_ddgs(monkeypatch, FakeDDGS(error=RuntimeError("boom")))
        assert "boom" in tool_call("q")


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


# ------------------------------------------------------------------ T10 image generation


class FakeImageGen:
    """Minimal stand-in for src.image_generator.ImageGenerator."""

    def __init__(self, fail_load=False, fail_save=False):
        self.fail_load, self.fail_save = fail_load, fail_save
        self.loaded = False
        self.calls = []
        self.unloaded = 0
        self.last_seed = None

    def is_t2i_pipe_loaded(self):
        return self.loaded

    def load_t2i_pipe(self):
        if self.fail_load:
            raise RuntimeError("no model")
        self.loaded = True
        return True

    def text2image(self, prompt, negative_prompt=None, width=None, height=None, seed=None, steps=None):
        self.calls.append({"prompt": prompt, "seed": seed, "width": width, "height": height})
        self.last_seed = seed if seed is not None and seed >= 0 else 424242
        return f"image-{len(self.calls)}"

    def save(self, image, img_path):
        if self.fail_save:
            return False
        path = Path(img_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # like ImageGenerator.save
        path.write_bytes(b"png")
        return True

    def unload_t2i_pipe(self):
        self.unloaded += 1
        was = self.loaded
        self.loaded = False
        return was


@pytest.fixture
def image_gen_seam(monkeypatch):
    """Reset the module-level seams around each image_gen test."""
    monkeypatch.setattr(tools.image_gen, "_generator", None)
    monkeypatch.setattr(tools.image_gen, "_GENERATOR_FACTORY", None)
    monkeypatch.setattr(tools.image_gen, "_config_cache", None)
    return tools.image_gen


def test_generate_image_success(tmp_path, image_gen_seam):
    fake = FakeImageGen()
    image_gen_seam._GENERATOR_FACTORY = lambda: fake
    out = tmp_path / "art.png"
    result = image_gen_seam.generate_image("a cat", seed=7, out_file=str(out))
    assert result == f"{out.resolve()} (seed=7)"
    assert out.read_bytes() == b"png"
    call = fake.calls[0]
    assert call["prompt"] == "a cat" and call["seed"] == 7
    assert call["width"] is None and call["height"] is None  # defaults resolved by the generator


def test_generate_image_reuses_cached_generator(tmp_path, image_gen_seam):
    fake = FakeImageGen()
    builds = []

    def factory():
        builds.append(1)
        return fake

    image_gen_seam._GENERATOR_FACTORY = factory
    image_gen_seam.generate_image("a cat", out_file=str(tmp_path / "a.png"))
    image_gen_seam.generate_image("a dog", out_file=str(tmp_path / "b.png"))
    assert len(builds) == 1 and len(fake.calls) == 2  # one model load, two generations


def test_generate_image_random_seed(tmp_path, image_gen_seam):
    fake = FakeImageGen()
    image_gen_seam._GENERATOR_FACTORY = lambda: fake
    result = image_gen_seam.generate_image("a cat", seed=-1, out_file=str(tmp_path / "a.png"))
    seed_used = int(result.rsplit("(seed=", 1)[1].rstrip(")"))
    assert seed_used >= 0


def test_generate_image_appends_png_extension(tmp_path, image_gen_seam):
    fake = FakeImageGen()
    image_gen_seam._GENERATOR_FACTORY = lambda: fake
    result = image_gen_seam.generate_image("a cat", out_file=str(tmp_path / "art"))
    assert result.startswith(str((tmp_path / "art.png").resolve()))


def test_generate_image_empty_prompt(image_gen_seam):
    called = []
    image_gen_seam._GENERATOR_FACTORY = lambda: called.append(1)
    assert "prompt is required" in image_gen_seam.generate_image("   ")
    assert not called  # factory never invoked


def test_generate_image_rejects_bad_size(image_gen_seam):
    assert "multiple of 8" in image_gen_seam.generate_image("a cat", width=500)
    assert "multiple of 8" in image_gen_seam.generate_image("a cat", height=-512)


def test_generate_image_rejects_directory(tmp_path, image_gen_seam):
    out_dir = tmp_path / "dir.png"
    out_dir.mkdir()
    assert "directory" in image_gen_seam.generate_image("a cat", out_file=str(out_dir))


def test_generate_image_load_failure(tmp_path, image_gen_seam):
    fake = FakeImageGen(fail_load=True)
    image_gen_seam._GENERATOR_FACTORY = lambda: fake
    assert "Error:" in image_gen_seam.generate_image("a cat", out_file=str(tmp_path / "a.png"))


def test_generate_image_missing_diffusers_hint(tmp_path, image_gen_seam, monkeypatch):
    class NoDiffusers(FakeImageGen):
        def load_t2i_pipe(self):
            raise ImportError("No module named 'diffusers'")

    image_gen_seam._GENERATOR_FACTORY = lambda: NoDiffusers()
    result = image_gen_seam.generate_image("a cat", out_file=str(tmp_path / "a.png"))
    assert "diffusers" in result and "pip install" in result


def test_generate_image_save_failure(tmp_path, image_gen_seam):
    fake = FakeImageGen(fail_save=True)
    image_gen_seam._GENERATOR_FACTORY = lambda: fake
    assert "Unable to save" in image_gen_seam.generate_image("a cat", out_file=str(tmp_path / "a.png"))


def test_unload_image_generator(image_gen_seam):
    fake = FakeImageGen()
    fake.loaded = True  # a loaded model is required for unload to do anything
    image_gen_seam._generator = fake
    assert "unloaded" in image_gen_seam.unload_image_generator()
    assert fake.unloaded == 1 and image_gen_seam._generator is None
    assert "No image generation model" in image_gen_seam.unload_image_generator()


def test_generate_image_default_out_file_from_config(tmp_path, image_gen_seam):
    image_gen_seam.configure(SimpleNamespace(image_gen=SimpleNamespace(out_dir=tmp_path / "images")))
    fake = FakeImageGen()
    image_gen_seam._GENERATOR_FACTORY = lambda: fake
    result = image_gen_seam.generate_image("a cat")
    out = Path(result.split(" (seed=", 1)[0])
    assert out.parent == (tmp_path / "images").resolve()
    assert out.name.startswith("generated_") and out.suffix == ".png"


def test_configure_uses_image_gen_section_and_resets_generator(image_gen_seam):
    fake = FakeImageGen()
    image_gen_seam._generator = fake
    image_gen_seam.configure(SimpleNamespace(
        image_gen=SimpleNamespace(t2i_model="fake/model", width=640, steps=20),
        brain=SimpleNamespace(model="qwen"),  # other sections are ignored
    ))
    assert image_gen_seam._generator is None  # config change drops the cached generator
    gen = image_gen_seam._default_generator_factory()
    assert gen.t2i_model == "fake/model"
    assert gen.defaults["width"] == 640 and gen.defaults["steps"] == 20
    assert gen.defaults["seed"] == 42  # keys missing from the section use built-ins


def test_default_generator_factory_without_section(image_gen_seam):
    image_gen_seam.configure(SimpleNamespace())  # config without [image_gen]
    gen = image_gen_seam._default_generator_factory()
    assert gen.defaults["width"] == 512  # built-in defaults kick in


def test_module_exports_public_tools():
    for name in tools.__all__:
        assert hasattr(tools, name)


def test_source_has_no_bare_exit():
    text = Path(tools.__file__).read_text(encoding="utf-8")
    assert "exit(0)" not in text


# ------------------------------------------------------------------ T11 summarize (Brain tool moved to module)


@pytest.fixture
def summarize_seam(monkeypatch):
    monkeypatch.setattr(tools.summarize, "_summarizer", None)
    monkeypatch.setattr(tools.summarize, "_SUMMARIZER_FACTORY", None)
    monkeypatch.setattr(tools.summarize, "_config_cache", None)
    return tools.summarize


def test_summarize_doc_reads_file_and_caches(tmp_path, summarize_seam):
    seen = {"calls": [], "count": 0}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def summarize_file(self, path, max_sentences=None):
            seen["calls"].append((str(path), max_sentences))
            return "SUMMARY"

    summarize_seam._SUMMARIZER_FACTORY = Recorder
    summarize_seam.configure(SimpleNamespace(summarization=SimpleNamespace(
        model_path="m", device="cpu", ratio=0.2, threshold=0.4, max_sentences=3)))

    first = tmp_path / "first.txt"
    first.write_text("long text", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("more text", encoding="utf-8")

    assert summarize_seam.summarize_doc(str(first)) == "SUMMARY"
    assert summarize_seam.summarize_doc(str(second)) == "SUMMARY"

    assert seen["count"] == 1  # summarizer created once and cached
    assert seen["init"]["model_path"] == "m"
    assert seen["init"]["ratio"] == 0.2
    assert seen["init"]["threshold"] == 0.4
    assert seen["init"]["max_sentences"] == 3
    assert seen["calls"] == [(str(first), None), (str(second), None)]


def test_summarize_doc_missing_file_returns_message(tmp_path, summarize_seam):
    summarize_seam.configure(SimpleNamespace(summarization=SimpleNamespace(model_path="m")))
    out = summarize_seam.summarize_doc(str(tmp_path / "nope.txt"))
    assert "Unable to read" in out


def test_summarize_doc_forwards_max_sentences(tmp_path, summarize_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize_file(self, path, max_sentences=None):
            seen["max"] = max_sentences
            return "S"

    summarize_seam._SUMMARIZER_FACTORY = Recorder
    summarize_seam.configure(SimpleNamespace(summarization=SimpleNamespace(model_path="m")))
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    assert summarize_seam.summarize_doc(str(target), max_sentences=2) == "S"
    assert seen["max"] == 2


def test_summarize_doc_reports_errors(tmp_path, summarize_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def summarize_file(self, path, max_sentences=None):
            raise RuntimeError("boom")

    summarize_seam._SUMMARIZER_FACTORY = Recorder
    summarize_seam.configure(SimpleNamespace(summarization=SimpleNamespace(model_path="m")))
    target = tmp_path / "a.txt"
    target.write_text("t", encoding="utf-8")
    out = summarize_seam.summarize_doc(str(target))
    assert out.startswith("Error:") and "boom" in out


# ------------------------------------------------------------------ T12 docquery


@pytest.fixture
def docquery_seam(monkeypatch):
    monkeypatch.setattr(tools.docquery, "_doc_query", None)
    monkeypatch.setattr(tools.docquery, "_DOCUMENT_QUERY_FACTORY", None)
    monkeypatch.setattr(tools.docquery, "_config_cache", None)
    return tools.docquery


def test_query_document_reads_config_and_caches(tmp_path, docquery_seam):
    seen = {"init": {}, "count": 0, "calls": []}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def query(self, path, query, top_k=None):
            seen["calls"].append((str(path), query, top_k))
            return "HITS"

    docquery_seam._DOCUMENT_QUERY_FACTORY = Recorder
    docquery_seam.configure(SimpleNamespace(document_query=SimpleNamespace(
        model_path="ce-model", device="cpu", chunk_size=4, top_k=7)))
    target = tmp_path / "doc.txt"
    target.write_text("text", encoding="utf-8")

    assert docquery_seam.query_document(str(target), "find x") == "HITS"
    assert docquery_seam.query_document(str(target), "find y", top_k=2) == "HITS"

    assert seen["count"] == 1  # created once and cached
    assert seen["init"]["model_path"] == "ce-model"
    assert seen["init"]["chunk_size"] == 4
    assert seen["init"]["top_k"] == 7
    assert seen["calls"] == [(str(target), "find x", None), (str(target), "find y", 2)]


def test_query_document_requires_query(docquery_seam):
    class Recorder:
        def __init__(self, **kwargs):
            raise AssertionError("must not be constructed")

    docquery_seam._DOCUMENT_QUERY_FACTORY = Recorder
    docquery_seam.configure(SimpleNamespace(document_query=SimpleNamespace(model_path="m")))
    assert docquery_seam.query_document("a.txt", "   ") == "Error: a query is required."


def test_query_document_reports_errors(tmp_path, docquery_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, path, query, top_k=None):
            raise RuntimeError("boom")

    docquery_seam._DOCUMENT_QUERY_FACTORY = Recorder
    docquery_seam.configure(SimpleNamespace(document_query=SimpleNamespace(model_path="m")))
    out = docquery_seam.query_document(str(tmp_path / "a.txt"), "q")
    assert out.startswith("Error:") and "boom" in out


# ------------------------------------------------------------------ T13 imgsearch


@pytest.fixture
def imgsearch_seam(monkeypatch):
    monkeypatch.setattr(tools.imgsearch, "_img_search", None)
    monkeypatch.setattr(tools.imgsearch, "_IMAGE_SEARCH_FACTORY", None)
    monkeypatch.setattr(tools.imgsearch, "_config_cache", None)
    return tools.imgsearch


def test_image_search_reads_config_and_caches(tmp_path, imgsearch_seam):
    seen = {"init": {}, "count": 0, "calls": []}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            seen["count"] += 1

        def search(self, directory, query, top_k=None):
            seen["calls"].append((directory, query, top_k))
            return [(0.912, str(tmp_path / "hit1.png")), (0.034, str(tmp_path / "hit2.png"))]

    imgsearch_seam._IMAGE_SEARCH_FACTORY = Recorder
    imgsearch_seam.configure(SimpleNamespace(image_search=SimpleNamespace(
        model_path="clip-model", device="cpu", top_k=4)))
    (tmp_path / "imgs").mkdir()

    out = imgsearch_seam.image_search(str(tmp_path / "imgs"), "a dog in snow")
    assert out == (
        f"1) [0.912] {tmp_path / 'hit1.png'}\n"
        f"2) [0.034] {tmp_path / 'hit2.png'}"
    )

    imgsearch_seam.image_search(str(tmp_path / "imgs"), "a cat", top_k=2)
    assert seen["count"] == 1  # created once and cached
    assert seen["init"]["model_path"] == "clip-model"
    assert seen["init"]["top_k"] == 4
    assert seen["calls"][0] == (str(tmp_path / "imgs"), "a dog in snow", None)
    assert seen["calls"][1] == (str(tmp_path / "imgs"), "a cat", 2)


def test_image_search_requires_query(imgsearch_seam):
    class Recorder:
        def __init__(self, **kwargs):
            raise AssertionError("must not be constructed")

    imgsearch_seam._IMAGE_SEARCH_FACTORY = Recorder
    imgsearch_seam.configure(SimpleNamespace(image_search=SimpleNamespace(model_path="m")))
    assert imgsearch_seam.image_search("some/folder", "  ") == "Error: a query is required."


def test_image_search_reports_errors(tmp_path, imgsearch_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def search(self, directory, query, top_k=None):
            raise RuntimeError("boom")

    imgsearch_seam._IMAGE_SEARCH_FACTORY = Recorder
    imgsearch_seam.configure(SimpleNamespace(image_search=SimpleNamespace(model_path="m")))
    out = imgsearch_seam.image_search(str(tmp_path), "q")
    assert out.startswith("Error:") and "boom" in out


def test_image_search_no_images_message(tmp_path, imgsearch_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def search(self, directory, query, top_k=None):
            return []

    imgsearch_seam._IMAGE_SEARCH_FACTORY = Recorder
    imgsearch_seam.configure(SimpleNamespace(image_search=SimpleNamespace(model_path="m")))
    assert imgsearch_seam.image_search(str(tmp_path), "q") == f"No readable images found in '{tmp_path}'."


# ------------------------------------------------------------------ T14 corpora


@pytest.fixture
def corpora_seam(monkeypatch):
    monkeypatch.setattr(tools.corpora, "_emb_corpus", None)
    monkeypatch.setattr(tools.corpora, "_EMBEDDING_CORPUS_FACTORY", None)
    monkeypatch.setattr(tools.corpora, "_config_cache", None)
    return tools.corpora


def _configure_corpora(seam, tmp_path, **overrides):
    cfg = dict(model_path="m", device="cpu", extensions="txt, md",
               chunk_size=3, top_k=5, corpora_dir=str(tmp_path))
    cfg.update(overrides)
    seam.configure(SimpleNamespace(embedding=SimpleNamespace(**cfg)))


def test_store_corpus_requires_name(corpora_seam):
    import inspect

    params = inspect.signature(tools.corpora.store_corpus).parameters
    assert params["corpus_name"].default is inspect.Parameter.empty

    seen = {"store": 0}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, *args, **kwargs):
            seen["store"] += 1
            return "STORED"

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, Path("/res/corpora"))
    out = corpora_seam.store_corpus("/docs", "")
    assert out.startswith("Error: a corpus name is required")
    assert seen["store"] == 0


def test_query_corpus_requires_name(corpora_seam):
    import inspect

    params = inspect.signature(tools.corpora.query_corpus).parameters
    assert params["corpus_name"].default is inspect.Parameter.empty

    seen = {"query": 0}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, *args, **kwargs):
            seen["query"] += 1
            return "RESULTS"

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, Path("/res/corpora"))
    out = corpora_seam.query_corpus("hello", "")
    assert out.startswith("Error: a corpus name is required")
    assert seen["query"] == 0


def test_store_corpus_uses_named_corpus(tmp_path, corpora_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, corpus_dir, overwrite=False):
            seen["store"] = (directory, str(corpus_dir), overwrite)
            return "STORED"

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path)
    assert corpora_seam.store_corpus("/docs", "autosar", overwrite=True) == "STORED"
    assert seen["store"] == ("/docs", str(tmp_path / "autosar"), True)


def test_store_corpus_invalid_name_returns_error(tmp_path, corpora_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, *args, **kwargs):
            raise AssertionError("store must not be called")

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path)
    out = corpora_seam.store_corpus("/docs", "a/b")
    assert out.startswith("Error:") and "corpus name" in out.lower()


def test_query_corpus_resolves_name_and_forwards(tmp_path, corpora_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_dir, query, top_k=None, kind=None):
            seen["query"] = (str(corpus_dir), query, top_k, kind)
            return "RESULTS"

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path)
    assert corpora_seam.query_corpus("hello", "autosar", kind="image", top_k=2) == "RESULTS"
    assert seen["query"] == (str(tmp_path / "autosar"), "hello", 2, "image")


def test_list_corpora_formats_entries(tmp_path, corpora_seam, monkeypatch):
    entries = [
        {"name": "autosar", "mode": "multimodal", "records": 10, "text_records": 8,
         "image_records": 2, "sources": 3, "built_at": "2026-01-01 00:00:00 UTC"},
        {"name": "quora", "mode": "text", "records": 5, "text_records": 5,
         "image_records": 0, "sources": 1, "built_at": "2026-01-02 00:00:00 UTC"},
    ]
    monkeypatch.setattr(tools.corpora, "list_corpus_entries", lambda corpora_dir: entries)
    _configure_corpora(corpora_seam, tmp_path)
    out = corpora_seam.list_corpora()
    assert "Available corpora (2)" in out
    assert "autosar: mode=multimodal" in out and "quora: mode=text" in out


def test_list_corpora_empty_returns_message(tmp_path, corpora_seam, monkeypatch):
    monkeypatch.setattr(tools.corpora, "list_corpus_entries", lambda corpora_dir: [])
    _configure_corpora(corpora_seam, tmp_path)
    assert corpora_seam.list_corpora() == "No corpora available."


def test_store_corpus_reports_errors(tmp_path, corpora_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def store(self, directory, corpus_dir, overwrite=False):
            raise RuntimeError("boom")

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path)
    out = corpora_seam.store_corpus("/docs", "autosar")
    assert out.startswith("Error:") and "boom" in out


def test_query_corpus_reports_errors(tmp_path, corpora_seam):
    class Recorder:
        def __init__(self, **kwargs):
            pass

        def query(self, corpus_dir, query, top_k=None, kind=None):
            raise RuntimeError("boom")

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path)
    out = corpora_seam.query_corpus("hi", "autosar")
    assert out.startswith("Error:") and "boom" in out


def test_store_corpus_passes_multimodal_config(tmp_path, corpora_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def store(self, directory, corpus_dir, overwrite=False):
            return "STORED"

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path,
                       image_model_path="clip", multimodal=True, image_extensions="png, jpg")
    assert corpora_seam.store_corpus("/docs", "autosar") == "STORED"
    assert seen["init"]["image_model_path"] == "clip"
    assert seen["init"]["multimodal"] is True
    assert seen["init"]["image_extensions"] == "png, jpg"
    assert "processed_name" not in seen["init"]


def test_embedding_corpus_defaults_when_config_is_minimal(tmp_path, corpora_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def query(self, corpus_dir, query, top_k=None, kind=None):
            return "RESULTS"

    corpora_seam._EMBEDDING_CORPUS_FACTORY = Recorder
    _configure_corpora(corpora_seam, tmp_path)
    corpora_seam.query_corpus("hi", "autosar")
    assert seen["init"]["multimodal"] is False
    assert seen["init"]["image_model_path"] is None
    assert "processed_name" not in seen["init"]


# ------------------------------------------------------------------ T15 memory


@pytest.fixture
def memory_seam(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.memory, "_history_provider", None)
    monkeypatch.setattr(tools.memory, "_config_cache", None)
    monkeypatch.setattr(tools.memory._default, "_summarizer", None)
    tools.memory.configure(SimpleNamespace(
        brain=SimpleNamespace(memory_path=str(tmp_path / "history.txt"),
                              system_prompt="be helpful"),
        agent=SimpleNamespace(bootstrap_prompt="bootstrap"),
    ))
    return tools.memory


def test_recall_memory_missing_file(memory_seam):
    assert memory_seam.recall_memory() == "No personal data found!"


def test_store_memory_appends_personal_messages(memory_seam, tmp_path):
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "bootstrap"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "my name is Max, remember this personal note"},
        {"role": "user", "content": "what time is it?"},
    ]
    memory_seam.bind_history(lambda: messages)

    result = memory_seam.store_memory()

    history = tmp_path / "history.txt"
    assert result == f"Conversation written to {history.resolve()}"
    text = history.read_text(encoding="utf-8")
    assert "personal note" in text and "what time is it?" not in text
    first_line = text.splitlines()[0]
    assert re.match(r"At \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} user wrote:", first_line), first_line


def test_store_memory_falls_back_to_last_user_message(memory_seam, tmp_path):
    """No message contains the word 'personal': the latest user message wins."""
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "remember that I like dogs"},
    ]
    memory_seam.bind_history(lambda: messages)

    result = memory_seam.store_memory()  # used to raise IndexError

    history = tmp_path / "history.txt"
    assert result == f"Conversation written to {history.resolve()}"
    assert "I like dogs" in history.read_text(encoding="utf-8")


def test_store_memory_prefers_explicit_content(memory_seam, tmp_path):
    """Explicit content wins over the history scan and needs no binding."""
    messages = [{"role": "user", "content": "unrelated chatter"}]
    memory_seam.bind_history(lambda: messages)

    result = memory_seam.store_memory(content="Birthday: May 5th")

    history = tmp_path / "history.txt"
    assert result == f"Conversation written to {history.resolve()}"
    text = history.read_text(encoding="utf-8")
    assert "Birthday: May 5th" in text and "unrelated chatter" not in text


def test_store_memory_without_user_message_returns_error(memory_seam, tmp_path):
    memory_seam.bind_history(lambda: [{"role": "assistant", "content": "hi"}])
    out = memory_seam.store_memory()
    assert out == "Error: no user message found to store."
    assert not (tmp_path / "history.txt").exists()


def test_recall_memory_returns_file_content(memory_seam, tmp_path):
    (tmp_path / "history.txt").write_text("At 01/01/2026 user wrote: personal", encoding="utf-8")
    assert "personal" in memory_seam.recall_memory()


def test_clear_chat_resets_history_in_place(memory_seam):
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "bootstrap"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "last question"},
    ]
    memory_seam.bind_history(lambda: messages)

    assert memory_seam.clear_chat() == "Chat cleared successfully"
    assert messages == [  # same list object, reset in place
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "No personal data found!"},
        {"role": "user", "content": "last question"},
    ]


def test_clear_chat_reseeds_with_recalled_personal_data(memory_seam, tmp_path):
    (tmp_path / "history.txt").write_text("stored personal data", encoding="utf-8")
    messages = [{"role": "user", "content": "hi"}]
    memory_seam.bind_history(lambda: messages)

    memory_seam.clear_chat()

    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1] == {"role": "user", "content": "stored personal data"}
    assert messages[2] == {"role": "user", "content": "hi"}


def test_clear_chat_without_bound_session(memory_seam, monkeypatch):
    monkeypatch.setattr(tools.memory, "_history_provider", None)
    assert memory_seam.clear_chat().startswith("Error:")


# ------------------------------------------------------------- T15b compaction


def test_compact_chat_without_bound_session(memory_seam):
    assert memory_seam.compact_chat().startswith("Error:")


def test_compact_chat_already_short(memory_seam):
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    memory_seam.bind_history(lambda: messages)
    snapshot = list(messages)

    assert memory_seam.compact_chat() == "Context is already compact; nothing to do."
    assert messages == snapshot  # untouched


def test_compact_chat_keeps_recent_turns_and_summary(memory_seam):
    messages = (
        [{"role": "system", "content": "be helpful"}]
        + [{"role": "user", "content": f"old question {i}"} for i in range(6)]
        + [{"role": "assistant", "content": f"old answer {i}"} for i in range(6)]
        + [{"role": "user", "content": "recent question"},
           {"role": "assistant", "content": "recent answer"}]
    )
    memory_seam.bind_history(lambda: messages)
    memory_seam.bind_summarizer(lambda old: "SUMMARY")

    result = memory_seam.compact_chat(keep_last=2)

    assert result == "Chat compacted: 13 messages summarized into 1, 2 kept."
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1] == {"role": "user", "content": "No personal data found!"}
    assert messages[2]["role"] == "user"
    assert messages[2]["content"].startswith("[Compacted context on ")
    assert "SUMMARY" in messages[2]["content"]
    assert messages[3:] == [
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    assert len(messages) == 5


def test_compact_chat_preserves_list_object(memory_seam):
    messages = [
        {"role": "user", "content": f"m{i}"} for i in range(10)
    ]
    memory_seam.bind_history(lambda: messages)
    same = messages

    memory_seam.compact_chat()

    assert messages is same  # in place: the session keeps the same list object


def test_compact_chat_uses_bound_summarizer(memory_seam):
    messages = (
        [{"role": "user", "content": f"old {i}"} for i in range(6)]
        + [{"role": "user", "content": "recent"}]
    )
    memory_seam.bind_history(lambda: messages)
    seen = []

    def summarizer(old):
        seen.append(list(old))
        return "S"

    memory_seam.bind_summarizer(summarizer)
    memory_seam.compact_chat(keep_last=1)  # clamped to minimum 2

    # the summarizer received exactly the non-recent messages, not the head copy
    assert seen == [[{"role": "user", "content": f"old {i}"} for i in range(5)]]


def test_compact_chat_reseeds_personal_data(memory_seam, tmp_path):
    (tmp_path / "history.txt").write_text("stored personal data", encoding="utf-8")
    messages = (
        [{"role": "system", "content": "be helpful"}]
        + [{"role": "user", "content": f"old {i}"} for i in range(6)]
        + [{"role": "user", "content": "recent"}]
    )
    memory_seam.bind_history(lambda: messages)

    memory_seam.compact_chat()

    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1] == {"role": "user", "content": "stored personal data"}


def test_compact_chat_fallback_summary_without_summarizer(memory_seam):
    messages = (
        [{"role": "user", "content": f"old {i}"} for i in range(6)]
        + [{"role": "user", "content": "recent"}]
    )
    memory_seam.bind_history(lambda: messages)

    memory_seam.compact_chat()

    # no system message in the original history: head = config system + recall
    assert "old 0" in messages[2]["content"]  # fallback extractive summary used


def test_compact_chat_invalid_keep_last_uses_default(memory_seam):
    messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    memory_seam.bind_history(lambda: messages)
    memory_seam.bind_summarizer(lambda old: "S")

    memory_seam.compact_chat(keep_last="abc")  # invalid -> default 4

    assert messages[-4:] == [{"role": "user", "content": f"m{i}"} for i in (6, 7, 8, 9)]


def test_module_default_compact_chat(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.memory, "_history_provider", None)
    monkeypatch.setattr(tools.memory, "_config_cache", None)
    tools.memory.configure(SimpleNamespace(
        brain=SimpleNamespace(memory_path=str(tmp_path / "history.txt"),
                              system_prompt="be helpful"),
        agent=SimpleNamespace(bootstrap_prompt="bootstrap"),
    ))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    tools.memory.bind_history(lambda: messages)
    tools.memory.bind_summarizer(lambda old: "MODSUM")

    result = tools.memory.compact_chat()

    assert result == "Chat compacted: 4 messages summarized into 1, 4 kept."
    assert "MODSUM" in messages[2]["content"]
    assert messages[1] == {"role": "user", "content": "No personal data found!"}
    assert len(messages) == 7


def test_compact_chat_falls_back_when_llm_summarizer_fails(memory_seam):
    """An LLM summarizer returning '' (failed call) falls back to extractive."""
    messages = (
        [{"role": "user", "content": f"old {i}"} for i in range(6)]
        + [{"role": "user", "content": "recent"}]
    )
    memory_seam.bind_history(lambda: messages)
    memory_seam.bind_summarizer(lambda old: "")

    memory_seam.compact_chat()

    assert "old 0" in messages[2]["content"]
    assert "[Compacted context on " in messages[2]["content"]


# ------------------------------------------------------------------ T16 speech


@pytest.fixture
def speech_seam(monkeypatch):
    monkeypatch.setattr(tools.speech, "_speech_generator", None)
    monkeypatch.setattr(tools.speech, "_SPEECH_FACTORY", None)
    monkeypatch.setattr(tools.speech, "_config_cache", None)
    return tools.speech


def test_allow_speech_forwards_config_and_closes(speech_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)
            self.closed = 0

        def close(self):
            self.closed += 1

    speech_seam._SPEECH_FACTORY = Recorder
    speech_seam.configure(SimpleNamespace(speech_gen=SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.3, speech_speed=1.5,
        speaker_id=2, noise_scale=0.4, noise_w_scale=0.7, normalize_audio=True,
    )))

    assert speech_seam.allow_speech(True) == "Speech Generator initialized"
    speech = speech_seam._speech_generator
    assert seen["speaker_id"] == 2 and seen["noise_scale"] == 0.4
    assert seen["noise_w_scale"] == 0.7 and seen["normalize_audio"] is True
    assert seen["speech_speed"] == 1.5

    assert speech_seam.allow_speech(False) == "Speech Generator released"
    assert speech.closed == 1
    assert speech_seam._speech_generator is None


def test_allow_speech_without_optional_keys(speech_seam):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)

        def close(self):
            pass

    speech_seam._SPEECH_FACTORY = Recorder
    speech_seam.configure(SimpleNamespace(speech_gen=SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.5, speech_speed=1.0)))
    speech_seam.allow_speech(True)
    assert seen["speaker_id"] is None
    assert seen["noise_scale"] == 1.0
    assert seen["noise_w_scale"] == 1.0
    assert seen["normalize_audio"] is False


def test_speak_if_enabled_only_speaks_when_enabled(speech_seam):
    spoken = []

    class Recorder:
        def speak(self, text):
            spoken.append(text)

        def close(self):
            pass

    speech_seam.allow_speech(False)  # nothing enabled yet
    speech_seam.speak_if_enabled("ignored")
    assert spoken == []

    speech_seam._speech_generator = Recorder()
    speech_seam.speak_if_enabled("hello")
    assert spoken == ["hello"]


def test_stop_speech_noop_without_generator(speech_seam):
    speech_seam.allow_speech(False)  # nothing enabled
    speech_seam.stop_speech()  # must not raise


def test_speak_if_enabled_installs_and_restores_sigint_handler(speech_seam):
    import signal

    captured = {}

    class Recorder:
        def speak(self, text):
            captured["during"] = signal.getsignal(signal.SIGINT)

        def request_stop(self):
            captured["stopped"] = True

        def close(self):
            pass

    previous = signal.getsignal(signal.SIGINT)
    try:
        speech_seam._speech_generator = Recorder()
        speech_seam.speak_if_enabled("hello")
        assert captured["during"] is not previous  # speech handler wins
        assert signal.getsignal(signal.SIGINT) is previous  # restored after
    finally:
        signal.signal(signal.SIGINT, previous)


def test_speak_if_enabled_routes_sigint_to_stop_speech(speech_seam, capsys):
    """Ctrl+C while audio plays stops the speech, never unwinds the turn."""
    import signal

    seen = {}

    class Recorder:
        def speak(self, text):
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)  # simulate Ctrl+C during playback
            seen["reached_end"] = True  # playback body completed normally

        def request_stop(self):
            seen["stopped"] = True

        def close(self):
            pass

    speech_seam._speech_generator = Recorder()
    speech_seam.speak_if_enabled("hello")  # must not raise
    assert seen["stopped"] is True and seen["reached_end"] is True
    assert speech_seam._speech_stopped is False  # flag consumed after printing
    assert "speech stopped" in capsys.readouterr().out.lower()


def test_ctrl_c_during_playback_stops_audio_only(speech_seam, tmp_path, capsys):
    """End-to-end: real SpeechGenerator, Ctrl+C mid-playback drops the audio."""
    pytest.importorskip("piper", reason="requires piper")
    pytest.importorskip("sounddevice", reason="requires sounddevice")
    import signal

    import src.audio_generator as ag
    from tests.conftest import FakeOutputStream, FakePiperVoice

    class SigintSecondWriteStream(FakeOutputStream):
        """FakeOutputStream simulating Ctrl+C on the second write."""

        def write(self, data):
            super().write(data)
            if len(self.written) == 2:
                handler = signal.getsignal(signal.SIGINT)
                assert handler is not signal.default_int_handler
                handler(signal.SIGINT, None)

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not-a-real-model")
    gen = ag.SpeechGenerator(
        model_path=model,
        voice=FakePiperVoice(chunks=(b"\x01\x02",) * 4),
        stream=SigintSecondWriteStream(),
    )
    speech_seam._speech_generator = gen
    previous = signal.getsignal(signal.SIGINT)
    try:
        speech_seam.speak_if_enabled("hello")  # must not raise
        assert gen.stream.written == [b"\x01\x02", b"\x01\x02"]  # 2 chunks played
        assert gen.stream.aborted == 1 and gen.stream.stopped == 0
        assert gen._stop_event.is_set() is False  # next round may speak again
        assert "speech stopped" in capsys.readouterr().out.lower()
        assert signal.getsignal(signal.SIGINT) is previous  # generation guard back
    finally:
        signal.signal(signal.SIGINT, previous)


# ------------------------------------------------------------------ T17 vision


@pytest.fixture
def vision_seam(monkeypatch):
    monkeypatch.setattr(tools.vision, "_session", None)
    return tools.vision


class FakeVisionSession:
    """Session double exposing the read-only attributes bind_session expects."""

    def __init__(self, vision=True, vision_source="server", fail=False):
        self.vision = vision
        self.vision_source = vision_source
        self.model_name = "fake-model"
        self.temperature = 0.3
        self.messages = [{"role": "system", "content": "sys"}]
        self.fail = fail

        class Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                if self._outer.fail:
                    raise RuntimeError("boom")
                self._outer.kwargs = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="it shows a cat"))])

        self.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions(self)))
        self.kwargs = None


def test_image_response_describes_image(vision_seam, tmp_path):
    session = FakeVisionSession()
    vision_seam.bind_session(session)
    image = tmp_path / "x.png"
    image.write_bytes(b"fake-image-bytes")
    assert vision_seam.image_response(str(image)) == "it shows a cat"
    assert session.kwargs["model"] == "fake-model"
    assert session.kwargs["temperature"] == 0.3
    assert session.messages[-1] == {"role": "assistant", "content": "it shows a cat"}


def test_image_response_without_vision(vision_seam, tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"fake-image-bytes")
    vision_seam.bind_session(FakeVisionSession(vision=False, vision_source="server"))
    assert "no 'vision' capability" in vision_seam.image_response(str(image))

    vision_seam.bind_session(FakeVisionSession(vision=False, vision_source="config"))
    assert "[brain] vision = true override" in vision_seam.image_response(str(image))


def test_image_response_without_session(vision_seam, tmp_path):
    assert vision_seam.image_response("x.png").startswith("Error:")


def test_image_response_provider_error_pops_image_turn(vision_seam, tmp_path):
    session = FakeVisionSession(fail=True)
    vision_seam.bind_session(session)
    image = tmp_path / "x.png"
    image.write_bytes(b"fake-image-bytes")
    before = len(session.messages)
    assert vision_seam.image_response(str(image)).startswith("Error:")
    assert len(session.messages) == before  # the image turn was dropped


# ------------------------------------------------------------------ T18 tools.configure fan-out


def test_tools_configure_fans_out_to_all_modules(monkeypatch):
    config = SimpleNamespace(
        image_gen=SimpleNamespace(), summarization=SimpleNamespace(),
        document_query=SimpleNamespace(), image_search=SimpleNamespace(),
        embedding=SimpleNamespace(), brain=SimpleNamespace(),
        agent=SimpleNamespace(), speech_gen=SimpleNamespace(),
    )
    seen = []
    for name in ("corpora", "docquery", "image_gen", "imgsearch",
                 "memory", "speech", "summarize"):
        module = getattr(tools, name)
        original = module.configure
        monkeypatch.setattr(
            module, "configure",
            lambda cfg, _n=name, _orig=original: (seen.append(_n), _orig(cfg)),
        )

    tools.configure(config)

    assert sorted(seen) == ["corpora", "docquery", "image_gen", "imgsearch",
                            "memory", "speech", "summarize"]
