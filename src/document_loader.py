"""
Document loading for the semantic-search corpus.

Extracts plain text (and, where available, images) from the common document
types so ``EmbeddingCorpus.store`` can index them:

- text-like: ``.txt`` ``.md`` ``.markdown`` ``.tex`` ``.rst`` ``.log`` ``.csv``
- markup:    ``.html`` ``.htm``
- office:    ``.pdf`` ``.docx`` ``.xlsx``
- images:    ``.png`` ``.jpg`` ``.jpeg`` ``.webp`` ``.bmp`` ``.tif`` ``.tiff``

Heavy dependencies (``pymupdf``, ``docx``, ``openpyxl``, ``bs4``, ``PIL``) are
imported lazily inside the extractor functions so this module is import-safe and
unit-testable with fakes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "TEXT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "Document",
    "strip_tex",
    "strip_dot_leaders",
    "extract_text",
    "extract_html",
    "extract_pdf",
    "extract_docx",
    "extract_xlsx",
    "extract_image",
    "load_document",
    "iter_documents",
]

TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".tex", ".rst", ".log", ".csv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".html", ".htm")
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS + DOCUMENT_EXTENSIONS + IMAGE_EXTENSIONS

# Embedded PDF images smaller than this (either side) or flatter than this
# (grayscale stddev) are icons, masks or blank fills -- not figures/diagrams.
MIN_IMAGE_DIM = 64
MIN_IMAGE_STD = 6.0

_TEX_COMMENT = re.compile(r"(?m)(?<!\\)%.*$")
_TEX_COMMAND = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?")
_TEX_MATH = re.compile(r"\$[^$]*\$")
_TEX_BRACES = re.compile(r"[{}]")
_DOT_LEADER = re.compile(r"(?:\.\s*){3,}")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Document:
    """One input file: its extracted text and/or image paths.

    ``sections`` carries ``(page_number, text)`` pairs for paginated documents
    (PDF). For single-section documents it is empty and the caller falls back to
    ``(pages, text)``.
    """

    source: str
    kind: str  # "text" | "image"
    text: str = ""
    images: list = field(default_factory=list)
    pages: Optional[int] = None
    error: str = ""
    sections: list = field(default_factory=list)


# --------------------------------------------------------------------- helpers


def _normalize_extensions(value) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else list(value)
    result: list[str] = []
    for item in items:
        item = str(item).strip().lower()
        if not item:
            continue
        result.append(item if item.startswith(".") else "." + item)
    return tuple(result)


def _safe_name(value) -> str:
    cleaned = _SAFE_NAME.sub("_", str(value)).strip("._")
    return cleaned or "file"


def _relative_source(path: Path, root) -> str:
    if root is None:
        return path.name
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _collapse_blank_lines(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def strip_tex(text: str) -> str:
    """Remove LaTeX comments, math and command noise while keeping words."""
    text = _TEX_COMMENT.sub("", text)
    text = _TEX_MATH.sub(" ", text)
    text = _TEX_COMMAND.sub(" ", text)
    text = _TEX_BRACES.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def strip_dot_leaders(text: str) -> str:
    """Collapse table-of-contents dot leaders (``. . . .``) into single spaces.

    PDF text extraction keeps the leader dots that connect a TOC entry to its
    page number; left in place they are misread as sentence boundaries and
    produce meaningless ``. . .`` passages.
    """
    return _DOT_LEADER.sub(" ", text or "")


# ------------------------------------------------------------------ openers (seams)


def _open_pdf(path):
    import pymupdf

    return pymupdf.open(path)


def _open_docx(path):
    import docx

    return docx.Document(path)


def _open_xlsx(path):
    import openpyxl

    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def _open_image(path):
    from PIL import Image

    return Image.open(path)


# ------------------------------------------------------------------ extractors


def extract_text(path) -> Document:
    """Plain text file; ``.tex`` is cleaned with :func:`strip_tex`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".tex":
        text = strip_tex(text)
    return Document(source=path.name, kind="text", text=text)


def extract_html(path) -> Document:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    return Document(source=path.name, kind="text", text=_collapse_blank_lines(soup.get_text("\n")))


def _image_is_useful(data: bytes) -> bool:
    """True when ``data`` decodes to an image with visible detail (not blank)."""
    import io

    from PIL import Image, ImageStat

    try:
        with Image.open(io.BytesIO(data)) as image:
            stddev = ImageStat.Stat(image.convert("L")).stddev[0]
    except Exception:
        return False
    return stddev >= MIN_IMAGE_STD


def _extract_pdf_images(document, page, page_index: int, image_dir, stem: str) -> list:
    """Save useful embedded page images under ``image_dir`` and return their paths.

    Skips soft-mask xrefs, duplicate xrefs, tiny icons and (near-)blank images.
    """
    try:
        infos = page.get_images(full=True)
    except Exception:
        return []

    mask_xrefs = {info[1] for info in infos if len(info) > 1 and info[1]}
    directory = Path(image_dir)
    paths: list[str] = []
    seen_xrefs: set = set()
    for info in infos:
        xref = info[0] if isinstance(info, (list, tuple)) else info
        if xref in seen_xrefs or xref in mask_xrefs:
            continue
        seen_xrefs.add(xref)

        width = info[2] if len(info) > 2 else None
        height = info[3] if len(info) > 3 else None
        if width and height and (width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM):
            continue

        try:
            image = document.extract_image(xref)
        except Exception:
            continue
        data = image.get("image") if isinstance(image, dict) else None
        if not data or not _image_is_useful(data):
            continue
        ext = (image.get("ext") or "png") if isinstance(image, dict) else "png"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{_safe_name(stem)}_p{page_index + 1}_{xref}.{ext}"
        target.write_bytes(data)
        paths.append(str(target.resolve()))
    return paths


def extract_pdf(path, image_dir=None, source: str = "") -> Document:
    path = Path(path)
    document = _open_pdf(str(path))
    pages = int(getattr(document, "page_count", 0) or 0)
    sections: list = []
    images: list[str] = []
    stem = Path(source).stem if source else path.stem
    try:
        for page_index in range(pages):
            page = document[page_index]
            page_text = strip_dot_leaders(page.get_text("text") or "")
            sections.append((page_index + 1, page_text))
            if image_dir is not None:
                images.extend(_extract_pdf_images(document, page, page_index, image_dir, stem))
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()
    text = "\n".join(section_text for _, section_text in sections)
    return Document(
        source=path.name,
        kind="text",
        text=text,
        images=images,
        pages=pages,
        sections=sections,
    )


def extract_docx(path) -> Document:
    path = Path(path)
    document = _open_docx(str(path))
    parts = [paragraph.text for paragraph in getattr(document, "paragraphs", [])]
    for table in getattr(document, "tables", []):
        for row in getattr(table, "rows", []):
            cells = [cell.text for cell in getattr(row, "cells", [])]
            if any(cell.strip() for cell in cells):
                parts.append(" | ".join(cells))
    return Document(source=path.name, kind="text", text="\n".join(parts))


def extract_xlsx(path) -> Document:
    path = Path(path)
    workbook = _open_xlsx(str(path))
    parts: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(value) for value in row if value is not None and str(value).strip()]
                if values:
                    rows.append(" | ".join(values))
            if rows:
                parts.append(f"# Sheet: {sheet.title}\n" + "\n".join(rows))
    finally:
        close = getattr(workbook, "close", None)
        if callable(close):
            close()
    return Document(source=path.name, kind="text", text="\n".join(parts))


def extract_image(path) -> Document:
    path = Path(path)
    text = path.name
    try:
        with _open_image(str(path)) as image:
            size = getattr(image, "size", None)
        if size:
            text = f"{path.name} ({size[0]}x{size[1]})"
    except Exception:
        pass
    return Document(source=path.name, kind="image", text=text, images=[str(path.resolve())])


def load_document(path, root=None, image_dir=None) -> Document:
    """Load one file into a :class:`Document`; errors are captured, not raised."""
    path = Path(path)
    source = _relative_source(path, root)
    suffix = path.suffix.lower()
    try:
        if suffix not in SUPPORTED_EXTENSIONS:
            return Document(source=source, kind="text", error=f"Unsupported file type: '{suffix}'")
        if suffix in IMAGE_EXTENSIONS:
            document = extract_image(path)
        elif suffix == ".pdf":
            document = extract_pdf(path, image_dir=image_dir, source=source)
        elif suffix == ".docx":
            document = extract_docx(path)
        elif suffix == ".xlsx":
            document = extract_xlsx(path)
        elif suffix in (".html", ".htm"):
            document = extract_html(path)
        else:  # text-like, including .tex
            document = extract_text(path)
    except Exception as exc:
        return Document(source=source, kind="text", error=f"{type(exc).__name__}: {exc}")
    document.source = source
    return document


def iter_documents(directory, extensions=None, image_dir=None) -> list[Document]:
    """Load every supported file under ``directory`` (recursively, sorted)."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: '{root}'")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: '{root}'")

    suffix_set = _normalize_extensions(extensions or SUPPORTED_EXTENSIONS)
    documents: list[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if suffix_set and path.suffix.lower() not in suffix_set:
            continue
        documents.append(load_document(path, root=root, image_dir=image_dir))
    return documents
