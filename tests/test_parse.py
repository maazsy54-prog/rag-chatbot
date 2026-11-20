"""Unit tests for file-type detection and text extraction."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from docx import Document

from app.ingestion.errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    IngestionError,
    UnsupportedFileTypeError,
)
from app.ingestion.parse import clean_text, parse_file, sniff_file_type


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """A two-page PDF with a heading and body text on each page."""
    document = pymupdf.open()
    for number in (1, 2):
        page = document.new_page()
        page.insert_text((72, 80), f"Chapter {number}", fontsize=22)
        page.insert_textbox(pymupdf.Rect(72, 110, 520, 400), "Body text about refunds. " * 15, fontsize=11)
    path = tmp_path / "sample.pdf"
    document.save(path)
    return path


@pytest.fixture()
def docx_file(tmp_path: Path) -> Path:
    """A Word file with a heading, a paragraph and a small table."""
    document = Document()
    document.add_heading("Refund Policy", level=1)
    document.add_paragraph("Refunds are paid within 14 days.")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Plan", "Days"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Basic", "14"
    path = tmp_path / "policy.docx"
    document.save(path)
    return path


def test_pdf_pages_keep_their_numbers(pdf_file: Path) -> None:
    pages = parse_file(pdf_file)
    assert [p.page for p in pages] == [1, 2]
    assert "refunds" in pages[0].text


def test_docx_keeps_headings_and_tables(docx_file: Path) -> None:
    (page,) = parse_file(docx_file)
    assert page.page is None
    assert "# Refund Policy" in page.text
    assert "| Plan | Days |" in page.text and "| Basic | 14 |" in page.text


def test_text_and_markdown_files_are_supported(tmp_path: Path) -> None:
    for name in ("notes.txt", "notes.md"):
        path = tmp_path / name
        path.write_text("# Notes\n\nHello world.", encoding="utf-8")
        assert parse_file(path)[0].text.startswith("# Notes")


def test_type_comes_from_content_not_extension(pdf_file: Path, tmp_path: Path) -> None:
    disguised = tmp_path / "report.docx"
    disguised.write_bytes(pdf_file.read_bytes())
    assert sniff_file_type(disguised) == "pdf"


def test_unsupported_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "program.exe"
    path.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")
    with pytest.raises(UnsupportedFileTypeError):
        parse_file(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        parse_file(path)


def test_blank_pdf_has_no_extractable_text(tmp_path: Path) -> None:
    document = pymupdf.open()
    document.new_page()
    path = tmp_path / "blank.pdf"
    document.save(path)
    with pytest.raises(EmptyDocumentError):
        parse_file(path)


def test_damaged_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a real pdf")
    with pytest.raises((CorruptDocumentError, EmptyDocumentError)) as info:
        parse_file(path)
    assert isinstance(info.value, IngestionError)


def test_damaged_docx_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"PK\x03\x04 not really a zip file")
    with pytest.raises(CorruptDocumentError):
        parse_file(path)


def test_clean_text_fixes_ligatures_and_blank_lines() -> None:
    assert clean_text("\ufb01nance\r\n\n\n\nreport  ") == "finance\n\nreport"
