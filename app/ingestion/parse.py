"""Text extraction: turn an uploaded file into clean Markdown, page by page.

The file type is detected from the file's *content*, not its extension, so a
renamed or disguised file cannot slip through. Markdown keeps headings, lists
and tables, which the chunker later uses as natural split points.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.ingestion.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)

# PDFs often store "fi", "fl" etc. as single ligature characters, which would
# otherwise break keyword search (a ligature "fi" would not match "fi").
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_HEADING_STYLE = re.compile(r"^heading\s+(\d)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPage:
    """Text of one page.

    Attributes:
        page: 1-based page number, or ``None`` for formats without fixed pages
            (Word, text and Markdown files), where headings identify location.
        text: The page content as Markdown.
    """

    page: int | None
    text: str


def clean_text(text: str) -> str:
    """Normalise line endings, ligatures and blank lines in extracted text."""
    for ligature, plain in _LIGATURES.items():
        text = text.replace(ligature, plain)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def sniff_file_type(path: Path) -> str:
    """Identify a file's real type from its bytes.

    Returns:
        One of ``"pdf"``, ``"docx"``, ``"txt"`` or ``"md"``.

    Raises:
        UnsupportedFileTypeError: If the content is not a supported type.
        CorruptDocumentError: If a Word file is a broken ZIP archive.
    """
    with path.open("rb") as handle:
        head = handle.read(4096)

    if head.startswith(b"%PDF-"):
        return "pdf"

    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                if "word/document.xml" in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile as exc:
            raise CorruptDocumentError("The file is a damaged ZIP archive.") from exc
        raise UnsupportedFileTypeError("The file is a ZIP archive but not a Word document.")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"} and b"\x00" not in head:
        return suffix.lstrip(".")

    raise UnsupportedFileTypeError(
        f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}."
    )


def _parse_pdf(path: Path) -> list[ParsedPage]:
    """Extract Markdown from each PDF page, keeping page numbers."""
    import pymupdf4llm  # imported lazily: the library is heavy to load

    try:
        page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception as exc:  # corrupt, encrypted or malformed PDF
        raise CorruptDocumentError(f"The PDF could not be read: {exc}") from exc

    if len(page_chunks) > settings.max_pages:
        raise DocumentTooLargeError(
            f"The document has {len(page_chunks)} pages; the limit is {settings.max_pages}."
        )
    return [
        ParsedPage(page=number, text=clean_text(chunk["text"]))
        for number, chunk in enumerate(page_chunks, start=1)
    ]


def _table_to_markdown(table) -> str:  # type: ignore[no-untyped-def]
    """Render a python-docx table as a Markdown table (first row = header)."""
    rows = [
        [cell.text.strip().replace("\n", " ").replace("|", "/") for cell in row.cells]
        for row in table.rows
    ]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _parse_docx(path: Path) -> list[ParsedPage]:
    """Extract headings, paragraphs and tables from a Word file in reading order."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(str(path))
    except Exception as exc:
        raise CorruptDocumentError(f"The Word file could not be read: {exc}") from exc

    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            match = _HEADING_STYLE.match(style)
            if match:
                blocks.append(f"{'#' * int(match.group(1))} {text}")
            elif style.lower() == "title":
                blocks.append(f"# {text}")
            else:
                blocks.append(text)
        elif tag == "tbl":
            table_text = _table_to_markdown(Table(child, document))
            if table_text:
                blocks.append(table_text)

    # Word has no fixed pages, so the whole file is one "page" with page=None.
    return [ParsedPage(page=None, text=clean_text("\n\n".join(blocks)))]


def _parse_text(path: Path) -> list[ParsedPage]:
    """Read a plain-text or Markdown file as UTF-8."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [ParsedPage(page=None, text=clean_text(raw))]


def parse_file(path: Path) -> list[ParsedPage]:
    """Extract clean Markdown pages from a supported file.

    Raises:
        UnsupportedFileTypeError: Unsupported content type.
        CorruptDocumentError: The file cannot be read.
        DocumentTooLargeError: More pages than ``settings.max_pages``.
        EmptyDocumentError: No extractable text (empty or scanned document).
    """
    parsers = {"pdf": _parse_pdf, "docx": _parse_docx, "txt": _parse_text, "md": _parse_text}
    pages = parsers[sniff_file_type(path)](path)
    if not any(page.text.strip() for page in pages):
        raise EmptyDocumentError(
            "No extractable text was found. The file may be empty or a scanned image."
        )
    return pages
