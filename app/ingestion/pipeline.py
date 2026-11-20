"""The ingestion pipeline: file -> text -> chunks -> vectors -> database.

Order matters for cost and safety: the cheap checks (size, duplicate hash,
parsing) run before the paid embedding call, and nothing is written to the
database until embedding has succeeded.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.config import settings
from app.ingestion import store
from app.ingestion.chunk import chunk_pages
from app.ingestion.embed import embed_texts
from app.ingestion.errors import DocumentTooLargeError, DuplicateDocumentError, EmptyDocumentError
from app.ingestion.parse import parse_file

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str], str], Awaitable[list[list[float]]]]


def sha256_of_file(path: Path) -> str:
    """Hash a file's content in 1 MB pieces, so large files do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def ingest_file(
    path: Path,
    filename: str,
    user_id: int,
    *,
    embedder: Embedder | None = None,
) -> store.DocumentRecord:
    """Ingest one uploaded file for a user.

    Args:
        path: Location of the saved upload.
        filename: Original file name, shown in the UI and in citations.
        user_id: Owner of the document.
        embedder: Replacement for the real embedding call (used by tests).

    Raises:
        DocumentTooLargeError: File or page count over the limit.
        DuplicateDocumentError: This user already uploaded the same content.
        UnsupportedFileTypeError / CorruptDocumentError / EmptyDocumentError:
            The file cannot be read (see ``app.ingestion.parse``).
        EmbeddingServiceError / DatabaseUnavailableError: A dependency failed.
    """
    started = time.perf_counter()
    embed = embedder or embed_texts

    if path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
        raise DocumentTooLargeError(f"The file is larger than {settings.max_upload_mb} MB.")

    # Blocking work (file reads, parsing, database) runs in worker threads so the
    # event loop stays free to serve other requests, e.g. simultaneous uploads.
    digest = await asyncio.to_thread(sha256_of_file, path)
    existing_id = await asyncio.to_thread(store.find_document_by_hash, user_id, digest)
    if existing_id is not None:
        raise DuplicateDocumentError(existing_id)

    pages = await asyncio.to_thread(parse_file, path)
    chunks = await asyncio.to_thread(chunk_pages, pages)
    if not chunks:
        raise EmptyDocumentError("No usable text was found in the document.")

    vectors = await embed([chunk.embed_text for chunk in chunks], "document")

    page_count = len(pages) if pages[0].page is not None else None
    record = await asyncio.to_thread(
        store.save_document, user_id, filename, digest, page_count, chunks, vectors
    )
    logger.info(
        "Ingested document id=%s chunks=%d in %.1fs",
        record.id,
        record.chunk_count,
        time.perf_counter() - started,
    )
    return record
