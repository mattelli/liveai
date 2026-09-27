import importlib
import sys

import pytest

import src.document_loader as loader
from src.document_loader import Document, extract_html, extract_pdf, load_document


class FakePdfPage:
    def __init__(self, text, images=()):
        self._text = text
        self._images = list(images)

    def get_text(self, kind="text"):
        return self._text

    def get_images(self, full=True):
        return [(xref, 0, 0, 0, 0, "", 0, 0, 0, 0) for xref in self._images]


class FakePdfDocument:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)
        self.closed = False

    def __getitem__(self, index):
        return self._pages[index]

    def extract_image(self, xref):
        return {"image": f"IMG{xref}".encode(), "ext": "png"}

    def close(self):
        self.closed = True


class FakeParagraph:
    def __init__(self, text):
        self.text = text


class FakeCell:
    def __init__(self, text):
        self.text = text


class FakeRow:
    def __init__(self, cells):
        self.cells = [FakeCell(cell) for cell in cells]


class FakeTable:
    def __init__(self, rows):
        self.rows = [FakeRow(row) for row in rows]


class FakeDocx:
    def __init__(self, paragraphs=(), tables=()):
        self.paragraphs = [FakeParagraph(p) for p in paragraphs]
        self.tables = [FakeTable(t) for t in tables]


class FakeSheet:
    def __init__(self, title, rows):
        self.title = title
        self._rows = rows

    def iter_rows(self, values_only=True):
        return iter(self._rows)


class FakeWorkbook:
    def __init__(self, sheets):
        self.worksheets = sheets
        self.closed = False

    def close(self):
        self.closed = True


# ------------------------------------------------------------- lazy imports


def test_import_is_lazy():
    for name in ("pymupdf", "docx", "openpyxl", "bs4", "PIL", "pandas"):
        sys.modules.pop(name, None)
    sys.modules.pop("src.document_loader", None)
    importlib.import_module("src.document_loader")
    for name in ("pymupdf", "docx", "openpyxl", "bs4", "PIL", "pandas"):
        assert name not in sys.modules, f"{name} imported at module load"


# ---------------------------------------------------------------- text/markup


def test_extract_text_plain(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    document = loader.extract_text(target)
    assert document.kind == "text" and document.text == "hello world"


def test_extract_tex_strips_commands(tmp_path):
    target = tmp_path / "a.tex"
    target.write_text(r"\section{Hello} % comment" "\n" r"World $x^2$", encoding="utf-8")
    document = load_document(target)
    assert "Hello" in document.text and "World" in document.text
    assert "%" not in document.text and "\\section" not in document.text


def test_extract_html_uses_get_text(tmp_path):
    target = tmp_path / "a.html"
    target.write_text(
        "<html><body><h1>Title</h1><p>Body text</p></body></html>", encoding="utf-8"
    )
    document = extract_html(target)
    assert "Title" in document.text and "Body text" in document.text
    assert "<h1>" not in document.text


# ---------------------------------------------------------------------- pdf


def test_extract_pdf_monkeypatched(tmp_path, monkeypatch):
    pages = [
        FakePdfPage("page one text", images=[11]),
        FakePdfPage("page two text"),
    ]
    fake = FakePdfDocument(pages)
    monkeypatch.setattr(loader, "_open_pdf", lambda path: fake)

    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    image_dir = tmp_path / "imgs"

    document = extract_pdf(source, image_dir=image_dir, source="report.pdf")

    assert "page one text" in document.text and "page two text" in document.text
    assert document.pages == 2
    assert len(document.images) == 1
    assert (image_dir / "report_p1_11.png").exists()
    assert fake.closed is True


# -------------------------------------------------------------------- office


def test_extract_docx_monkeypatched(tmp_path, monkeypatch):
    fake = FakeDocx(paragraphs=["First paragraph", "Second"], tables=[[["A", "B"]]])
    monkeypatch.setattr(loader, "_open_docx", lambda path: fake)
    target = tmp_path / "a.docx"
    target.write_bytes(b"x")

    document = load_document(target)
    assert "First paragraph" in document.text
    assert "A | B" in document.text


def test_extract_xlsx_monkeypatched(tmp_path, monkeypatch):
    fake = FakeWorkbook([FakeSheet("Sheet1", [("A", 1), (None, "B")])])
    monkeypatch.setattr(loader, "_open_xlsx", lambda path: fake)
    target = tmp_path / "a.xlsx"
    target.write_bytes(b"x")

    document = load_document(target)
    assert "# Sheet: Sheet1" in document.text
    assert "A | 1" in document.text and "B" in document.text
    assert fake.closed is True


# -------------------------------------------------------------------- images


def test_extract_image_returns_image_record(tmp_path):
    from PIL import Image

    target = tmp_path / "pic.png"
    Image.new("RGB", (3, 2), color="red").save(target)

    document = load_document(target)
    assert document.kind == "image"
    assert document.images == [str(target.resolve())]
    assert "3x2" in document.text


# --------------------------------------------------------------- robustness


def test_load_document_error_is_captured(tmp_path, monkeypatch):
    def boom(path):
        raise RuntimeError("cannot parse")

    monkeypatch.setattr(loader, "_open_docx", boom)
    target = tmp_path / "a.docx"
    target.write_bytes(b"x")

    document = load_document(target)
    assert document.error and "RuntimeError" in document.error


def test_iter_documents_filters_and_sorts(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "c.bin").write_text("nope", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.md").write_text("beta", encoding="utf-8")

    documents = loader.iter_documents(tmp_path)
    assert [d.source for d in documents] == ["a.txt", "sub/b.md"]


def test_iter_documents_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.iter_documents(tmp_path / "nope")
