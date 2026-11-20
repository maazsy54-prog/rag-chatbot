"""Structure-aware chunking.

Markdown pages are split on headings, then paragraphs, then sentences, then
words (in that order of preference) into chunks of about ``target_tokens``
tokens. Small tables stay whole; large ones are split by rows with the header
repeated. Each chunk remembers its heading path and page range, which become
the citation shown to the user.

Token counts use the ``cl100k_base`` tokenizer as a close approximation (the
embedding model has its own tokenizer). If it cannot be loaded, e.g. when
offline, characters / 4 is used instead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterator

from app.config import settings
from app.ingestion.parse import ParsedPage

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Split after sentence-ending punctuation (Latin, Arabic question mark, Urdu full stop)
# or at line breaks, so list items are separate units.
_SENTENCE_RE = re.compile(r"(?<=[.!?\u061f\u06d4])\s+|\n+")
_MIN_CHUNK_CHARS = 15  # chunks shorter than this (page numbers, stray symbols) are dropped


@lru_cache(maxsize=1)
def _encoder():  # type: ignore[no-untyped-def]
    """Load the tokenizer once; return ``None`` if it is unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline, or the package is missing
        logger.warning("tiktoken unavailable; estimating tokens as characters / 4")
        return None


def count_tokens(text: str) -> int:
    """Return the (approximate) number of tokens in ``text``."""
    encoder = _encoder()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a document.

    Attributes:
        index: Position of the chunk within its document (0-based).
        text: The chunk content shown to the user and the LLM.
        heading: Heading path such as ``"Contract > Termination"``, if any.
        page_start: First page the chunk covers (``None`` for page-less formats).
        page_end: Last page the chunk covers.
        token_count: Approximate token count of ``text``.
    """

    index: int
    text: str
    heading: str | None
    page_start: int | None
    page_end: int | None
    token_count: int

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: the heading path plus the content.

        Prefixing the heading restores context a chunk would otherwise lose
        (e.g. which section a bare "30 days" belongs to).
        """
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text


@dataclass(frozen=True)
class _Block:
    """A paragraph or table with the location it came from."""

    kind: str  # "text" or "table"
    text: str
    page: int | None
    heading: str | None  # full heading path
    title: str | None  # innermost heading title


def _split_markdown(text: str) -> list[tuple[str, str, int]]:
    """Split Markdown into ``(kind, text, level)`` items.

    ``kind`` is ``"heading"`` (with its level 1-6), ``"table"`` or ``"text"``.
    """
    items: list[tuple[str, str, int]] = []
    buffer: list[str] = []
    buffer_kind = "text"

    def flush() -> None:
        nonlocal buffer
        if buffer:
            items.append((buffer_kind, "\n".join(buffer), 0))
            buffer = []

    for line in text.split("\n"):
        if not line.strip():
            flush()
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            title = heading.group(2).strip("* ").strip()
            if title:
                items.append(("heading", title, len(heading.group(1))))
            continue
        kind = "table" if line.lstrip().startswith("|") else "text"
        if kind != buffer_kind:
            flush()
        buffer_kind = kind
        buffer.append(line)
    flush()
    return items


def _iter_blocks(pages: list[ParsedPage]) -> Iterator[_Block]:
    """Yield content blocks, each tagged with its page and heading path.

    The heading stack carries over from page to page, so a section that
    continues on the next page keeps its heading.
    """
    stack: list[tuple[int, str]] = []
    for page in pages:
        for kind, text, level in _split_markdown(page.text):
            if kind == "heading":
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                continue
            path = " > ".join(title for _, title in stack) or None
            title = stack[-1][1] if stack else None
            yield _Block(kind, text, page.page, path, title)


def _split_long_sentence(sentence: str, limit: int) -> list[str]:
    """Cut a single over-long sentence into pieces of about ``limit`` tokens."""
    tokens = count_tokens(sentence)
    if tokens <= limit:
        return [sentence]
    words = sentence.split()
    if len(words) > 1:
        step = max(1, int(len(words) * limit / tokens))
        return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
    size = max(1, int(len(sentence) * limit / tokens))  # no spaces (e.g. CJK text)
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def _split_prose(text: str, limit: int) -> list[str]:
    """Group sentences into parts of at most about ``limit`` tokens."""
    parts: list[str] = []
    current: list[str] = []
    used = 0
    for sentence in (s for s in _SENTENCE_RE.split(text) if s and s.strip()):
        for piece in _split_long_sentence(sentence, limit):
            size = count_tokens(piece)
            if current and used + size > limit:
                parts.append(" ".join(current))
                current, used = [], 0
            current.append(piece)
            used += size
    if current:
        parts.append(" ".join(current))
    return parts


def _split_table(text: str, limit: int) -> list[str]:
    """Split a Markdown table by rows, repeating the header in every part."""
    lines = text.split("\n")
    header_len = 2 if len(lines) > 1 and "---" in lines[1] else 1
    header, rows = lines[:header_len], lines[header_len:]
    header_tokens = count_tokens("\n".join(header))
    parts: list[str] = []
    current: list[str] = []
    used = header_tokens
    for row in rows:
        size = count_tokens(row)
        if current and used + size > limit:
            parts.append("\n".join(header + current))
            current, used = [], header_tokens
        current.append(row)
        used += size
    if current:
        parts.append("\n".join(header + current))
    return parts or [text]


def _split_block(block: _Block, max_tokens: int, target_tokens: int) -> list[_Block]:
    """Split a block only if it exceeds ``max_tokens``; pieces aim for the target size."""
    if count_tokens(block.text) <= max_tokens:
        return [block]
    splitter = _split_table if block.kind == "table" else _split_prose
    return [replace(block, text=part) for part in splitter(block.text, target_tokens)]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return roughly the last ``overlap_tokens`` tokens of ``text`` (whole words)."""
    if overlap_tokens <= 0:
        return ""
    tail: list[str] = []
    for word in reversed(text.split()):
        tail.append(word)
        if count_tokens(" ".join(tail)) >= overlap_tokens:
            break
    return " ".join(reversed(tail))


def chunk_pages(
    pages: list[ParsedPage],
    *,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int | None = None,
) -> list[Chunk]:
    """Split parsed pages into chunks.

    Rules:
        * A chunk never mixes sections unless the earlier section is smaller
          than ``min_tokens`` (tiny sections are merged to avoid weak chunks).
        * When one section has to be cut for size, the next chunk starts with
          about ``overlap_tokens`` of the previous one so no sentence is lost
          at the boundary. Overlap is never carried across a new section or
          out of a table (tables repeat their header row instead).
        * Blocks larger than ``max_tokens`` are split by sentence (prose) or
          by row with the header repeated (tables).

    Args:
        pages: Output of :func:`app.ingestion.parse.parse_file`.
        target_tokens: Desired chunk size. Defaults to the configured value.
        max_tokens: Hard ceiling for a single block. Defaults to the configured value.
        overlap_tokens: Overlap between size-split chunks.
        min_tokens: Minimum size before a new section starts a new chunk.

    Returns:
        Chunks in document order with contiguous ``index`` values.
    """
    target = target_tokens or settings.chunk_target_tokens
    ceiling = max_tokens or settings.chunk_max_tokens
    overlap_size = settings.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens
    minimum = settings.chunk_min_tokens if min_tokens is None else min_tokens

    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_tokens = 0
    overlap = ""

    def emit() -> str:
        """Turn the blocks gathered so far into a chunk and return its body text."""
        nonlocal current, current_tokens
        body = "\n\n".join(block.text for block in current)
        text = f"{overlap}\n\n{body}" if overlap else body
        pages_seen = [block.page for block in current if block.page is not None]
        if len(text.strip()) >= _MIN_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    heading=current[0].heading,
                    page_start=min(pages_seen) if pages_seen else None,
                    page_end=max(pages_seen) if pages_seen else None,
                    token_count=count_tokens(text),
                )
            )
        current, current_tokens = [], 0
        return body

    for block in _iter_blocks(pages):
        for piece in _split_block(block, ceiling, target):
            size = count_tokens(piece.text)
            if current:
                new_section = piece.heading != current[-1].heading
                if new_section and current_tokens >= minimum:
                    emit()
                    overlap = ""
                elif current_tokens + size > target:
                    last_kind = current[-1].kind
                    body = emit()
                    # Tables repeat their header row instead of using text overlap.
                    carry_over = not new_section and last_kind != "table"
                    overlap = _overlap_tail(body, overlap_size) if carry_over else ""
                elif new_section and piece.title:
                    # Merging a tiny section: keep its title so the text stays labelled.
                    piece = replace(piece, text=f"{piece.title}\n\n{piece.text}")
                    size = count_tokens(piece.text)
            current.append(piece)
            current_tokens += size
    if current:
        emit()

    logger.info("Created %d chunks", len(chunks))
    return chunks
