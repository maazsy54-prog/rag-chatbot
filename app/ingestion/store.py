"""Database storage for documents and chunks (PostgreSQL + pgvector).

All functions take a ``user_id`` and filter by it, so one user can never read,
duplicate-check against or delete another user's data. Deleting a document
removes its chunks and vectors in the same statement (ON DELETE CASCADE), so
deleted content can never appear in search results again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.db import connection
from app.ingestion.chunk import Chunk
from app.ingestion.errors import DuplicateDocumentError

_INSERT_CHUNK = """
    INSERT INTO chunks
        (document_id, user_id, chunk_index, page_start, page_end,
         heading, content, token_count, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
"""

_SELECT_DOCUMENTS = """
    SELECT d.id, d.filename, d.sha256, d.page_count, count(c.id), d.created_at
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    WHERE d.user_id = %s {extra}
    GROUP BY d.id
    ORDER BY d.created_at DESC
"""


@dataclass(frozen=True)
class DocumentRecord:
    """A stored document, as shown in the document list."""

    id: int
    filename: str
    sha256: str
    page_count: int | None
    chunk_count: int
    created_at: datetime


def to_vector_literal(vector: list[float]) -> str:
    """Format a vector as the text pgvector accepts, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(f"{value:.8g}" for value in vector) + "]"


def find_document_by_hash(user_id: int, sha256: str) -> int | None:
    """Return the id of this user's document with the given file hash, if any."""
    with connection() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE user_id = %s AND sha256 = %s", (user_id, sha256)
        ).fetchone()
    return row[0] if row else None


def save_document(
    user_id: int,
    filename: str,
    sha256: str,
    page_count: int | None,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> DocumentRecord:
    """Store a document and all its chunks in one transaction.

    Either everything is saved or nothing is. The unique ``(user_id, sha256)``
    constraint makes duplicate detection safe even when two identical uploads
    arrive at the same moment.

    Raises:
        DuplicateDocumentError: This user already stored a file with the same hash.
        ValueError: ``chunks`` and ``vectors`` have different lengths.
    """
    if len(chunks) != len(vectors):
        raise ValueError("Every chunk needs exactly one vector.")

    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (user_id, filename, sha256, status, page_count)
            VALUES (%s, %s, %s, 'ready', %s)
            ON CONFLICT (user_id, sha256) DO NOTHING
            RETURNING id, created_at
            """,
            (user_id, filename, sha256, page_count),
        ).fetchone()
        if row is None:
            existing = conn.execute(
                "SELECT id FROM documents WHERE user_id = %s AND sha256 = %s", (user_id, sha256)
            ).fetchone()
            raise DuplicateDocumentError(existing[0] if existing else -1)

        document_id, created_at = row
        conn.cursor().executemany(
            _INSERT_CHUNK,
            [
                (
                    document_id,
                    user_id,
                    chunk.index,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.heading,
                    chunk.text,
                    chunk.token_count,
                    to_vector_literal(vector),
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )

    return DocumentRecord(document_id, filename, sha256, page_count, len(chunks), created_at)


def list_documents(user_id: int) -> list[DocumentRecord]:
    """Return all of a user's documents, newest first."""
    with connection() as conn:
        rows = conn.execute(_SELECT_DOCUMENTS.format(extra=""), (user_id,)).fetchall()
    return [DocumentRecord(*row) for row in rows]


def get_document(user_id: int, document_id: int) -> DocumentRecord | None:
    """Return one of the user's documents, or ``None`` if it does not exist."""
    with connection() as conn:
        row = conn.execute(
            _SELECT_DOCUMENTS.format(extra="AND d.id = %s"), (user_id, document_id)
        ).fetchone()
    return DocumentRecord(*row) if row else None


def delete_document(user_id: int, document_id: int) -> bool:
    """Delete a document with its chunks and vectors.

    Returns:
        ``True`` if a document was deleted, ``False`` if it did not exist or
        belongs to another user.
    """
    with connection() as conn:
        deleted = conn.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s", (document_id, user_id)
        ).rowcount
    return deleted == 1
