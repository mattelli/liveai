"""Archive tools: extract zip/7z/tar/gz archives and zip folder contents."""

from pathlib import Path

from src.common import dump_func_call
from .files import normalize_extensions

__all__ = ["extract_archive", "zip_files"]

_ZIP_SUFFIXES = (".zip",)
_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
_SUPPORTED_ARCHIVES = (
    "zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and gz"
)


def _archive_stem(path: Path) -> str:
    """``foo.tar.gz`` -> ``foo``; ``foo.zip`` -> ``foo``."""
    name = path.name.lower()
    for suffix in _TAR_SUFFIXES + _ZIP_SUFFIXES + (".7z", ".gz"):
        if name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _archive_kind(path: Path) -> str:
    """Map an archive path to its extractor: 'zip', 'tar', '7z', 'gz' or ''."""
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in _ZIP_SUFFIXES):
        return "zip"
    if any(name.endswith(suffix) for suffix in _TAR_SUFFIXES):
        return "tar"
    if name.endswith(".7z"):
        return "7z"
    if name.endswith(".gz"):
        return "gz"
    return ""


def _extract_zip(path: Path, dest: Path) -> int:
    """Extract a zip archive, rejecting entries that escape ``dest`` (zip slip)."""
    import zipfile

    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        root = dest.resolve()
        for member in members:
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive entry '{member.filename}'")
        for member in members:
            archive.extract(member, dest)
        return len(members)


def _extract_tar(path: Path, dest: Path) -> int:
    """Extract a tar archive; ``filter='data'`` blocks path traversal (PEP 706)."""
    import tarfile

    with tarfile.open(path) as archive:
        members = archive.getmembers()
        archive.extractall(dest, filter="data")
        return len(members)


def _extract_7z(path: Path, dest: Path) -> int:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "py7zr is required for .7z archives (install it with: pip install py7zr)"
        ) from exc
    with py7zr.SevenZipFile(path) as archive:
        entries = len(archive.list())
        archive.extractall(path=dest)
        return entries


def _extract_gz(path: Path, dest: Path) -> int:
    """Decompress a bare ``.gz`` file (not a tar) into ``dest/<stem>``."""
    import gzip
    import shutil

    target = dest / _archive_stem(path)
    with gzip.open(path, "rb") as source, open(target, "wb") as output:
        shutil.copyfileobj(source, output)
    return 1


@dump_func_call
def extract_archive(archive: str, out_dir: str | None = None, overwrite: bool = False) -> str:
    """
    Unzip / unpack a compressed archive and save its content into a folder.
    Useful when user asks to extract, unzip, unpack or decompress an archive.
    Supported formats: zip, 7z, tar, tar.gz, tgz, tar.bz2, tbz2, tar.xz, txz and
    single-file gz. By default the content is extracted into a new folder named
    after the archive (e.g. ``data.zip`` -> ``data/``); ``out_dir`` overrides it.
    A non-empty destination is only written when ``overwrite`` is True. Returns
    the destination folder and the number of extracted entries.
    """
    archive_path = Path(archive).resolve()
    if not archive_path.exists():
        return f"Error: '{archive_path}' does not exist!"
    if archive_path.is_dir():
        return f"Error: '{archive_path}' is a directory!"
    kind = _archive_kind(archive_path)
    if not kind:
        return f"Error: unsupported archive type '{archive_path.suffix}'. Supported: {_SUPPORTED_ARCHIVES}."

    if kind == "gz":  # single decompressed file, extracted beside the archive
        dest = Path(out_dir) if out_dir is not None else archive_path.parent
        if (dest / _archive_stem(archive_path)).exists() and not overwrite:
            return "File already exists."
    else:
        dest = Path(out_dir) if out_dir is not None else archive_path.parent / _archive_stem(archive_path)
        if dest.exists() and any(dest.iterdir()) and not overwrite:
            return f"Destination '{dest}' is not empty. Use overwrite=True to extract into it."
    try:
        dest.mkdir(parents=True, exist_ok=True)
        extractors = {
            "zip": _extract_zip, "tar": _extract_tar,
            "7z": _extract_7z, "gz": _extract_gz,
        }
        entries = extractors[kind](archive_path, dest)
    except Exception as exc:
        return f"Error extracting '{archive_path.name}': {exc}"
    return f"Extracted {entries} entries from '{archive_path.name}' to '{dest.resolve()}'."


def _collect_zip_files(
        directory: Path,
        include_extensions: list[str],
        exclude_extensions: list[str],
        excluded_items: list[str],
        skip_path,
    ) -> list[Path]:
    """Every file under ``directory`` (recursively, sorted) to pack into a zip.

    Mirrors ``search_pattern_in_files``: hidden entries matched by the exclude
    glob are skipped, extension filters are optional comma lists.
    """
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if skip_path is not None and path.resolve() == skip_path:
            continue
        if any(item in path.parts for item in excluded_items):
            continue
        ext = path.suffix.lower()
        if include_extensions and ext not in include_extensions:
            continue
        if exclude_extensions and ext in exclude_extensions:
            continue
        files.append(path)
    return files


@dump_func_call
def zip_files(
        directory: str,
        archive: str,
        exclude_glob: str | None = ".*",
        include_extensions: str = "",
        exclude_extensions: str = "",
        overwrite: bool = False,
    ) -> str:
    """
    Zip the files of a folder into a zip archive (recursively, keeping the
    relative paths). Useful when user asks to compress, zip or pack a folder or
    its files. ``include_extensions``/``exclude_extensions`` accept ``py`` or
    ``.py`` (empty = all files); ``exclude_glob`` (default ``".*"``) skips
    hidden entries. A ``.zip`` suffix is appended to ``archive`` when missing,
    and existing archives are only replaced when ``overwrite`` is True. Returns
    the archive path and the number of files zipped.
    """
    search_dir = Path(directory).resolve()
    if not search_dir.exists():
        return f"'{search_dir}' does not exist!"
    if not search_dir.is_dir():
        return f"'{search_dir}' is not a directory!"

    archive_path = Path(archive)
    if archive_path.suffix.lower() != ".zip":
        archive_path = archive_path.with_suffix(archive_path.suffix + ".zip")
    if archive_path.exists() and not overwrite:
        return "File already exists."

    excluded_items = [f.name for f in search_dir.glob(exclude_glob)] if exclude_glob else []
    files = _collect_zip_files(
        search_dir,
        normalize_extensions(include_extensions),
        normalize_extensions(exclude_extensions),
        excluded_items,
        archive_path.resolve(),  # never zip the archive into itself
    )
    if not files:
        return f"No files to zip in '{search_dir}'."

    total_bytes = 0
    try:
        import zipfile

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, arcname=path.relative_to(search_dir))
                total_bytes += path.stat().st_size
    except Exception as exc:
        return f"Error zipping: {exc}"
    return (
        f"Zipped {len(files)} files ({total_bytes:,} bytes) from '{search_dir}' "
        f"to '{archive_path.resolve()}'."
    )