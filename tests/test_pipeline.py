"""Integration tests for ingestion, duplicate detection and deletion.

These use the real PostgreSQL + pgvector database (start it with
``docker compose up -d``) but a fake embedder, so no API calls are made.
They are skipped automatically when the database is not running.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from app.db import connection
from app.ingestion import pipeline, store
from app.ingestion.errors import (
    DuplicateDocumentError,
    EmbeddingServiceError,
    UnsupportedFileTypeError,
)
from tests.conftest import fake_embedder


@pytest.fixture(autouse=True)
def _use_fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the pipeline use the fake embedder instead of calling Voyage."""
    monkeypatch.setattr(pipeline, "embed_texts", fake_embedder)


def _search(user_id: int, vector: str) -> list[tuple]:
    """Nearest chunks for a user, the way retrieval will do it."""
    with connection() as conn:
        return conn.execute(
            "SELECT id, 1 - (embedding <=> %s::vector) FROM chunks WHERE user_id = %s "
            "ORDER BY embedding <=> %s::vector LIMIT 5",
            (vector, user_id, vector),
        ).fetchall()


def test_ingest_stores_chunks_with_metadata(db_user: int, make_docx: Callable[[str], Path]) -> None:
    record = asyncio.run(pipeline.ingest_file(make_docx("alpha"), "alpha.docx", db_user))
    assert record.chunk_count >= 2 and record.page_count is None
    assert [d.id for d in store.list_documents(db_user)] == [record.id]
    with connection() as conn:
        rows = conn.execute(
            "SELECT chunk_index, heading, token_count, embedding IS NOT NULL "
            "FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (record.id,),
        ).fetchall()
    assert [row[0] for row in rows] == list(range(record.chunk_count))
    assert all(row[1] and row[2] > 0 and row[3] for row in rows)
    assert any("Refunds" in row[1] for row in rows)


def test_duplicate_upload_is_rejected_and_not_stored_twice(
    db_user: int, make_docx: Callable[[str], Path]
) -> None:
    path = make_docx("beta")
    first = asyncio.run(pipeline.ingest_file(path, "beta.docx", db_user))
    with pytest.raises(DuplicateDocumentError) as info:
        asyncio.run(pipeline.ingest_file(path, "renamed-copy.docx", db_user))
    assert info.value.document_id == first.id
    assert len(store.list_documents(db_user)) == 1


def test_two_simultaneous_identical_uploads_store_one(
    db_user: int, make_docx: Callable[[str], Path]
) -> None:
    path = make_docx("gamma")

    async def upload_twice() -> list:
        return await asyncio.gather(
            pipeline.ingest_file(path, "a.docx", db_user),
            pipeline.ingest_file(path, "b.docx", db_user),
            return_exceptions=True,
        )

    results = asyncio.run(upload_twice())
    assert sum(isinstance(r, store.DocumentRecord) for r in results) == 1
    assert sum(isinstance(r, DuplicateDocumentError) for r in results) == 1
    assert len(store.list_documents(db_user)) == 1


def test_same_file_is_allowed_for_different_users(
    user_factory: Callable[[], int], make_docx: Callable[[str], Path]
) -> None:
    path = make_docx("delta")
    first, second = user_factory(), user_factory()
    asyncio.run(pipeline.ingest_file(path, "d.docx", first))
    asyncio.run(pipeline.ingest_file(path, "d.docx", second))
    assert len(store.list_documents(first)) == len(store.list_documents(second)) == 1


def test_deleted_content_can_never_be_found_again(
    db_user: int, make_docx: Callable[[str], Path]
) -> None:
    record = asyncio.run(pipeline.ingest_file(make_docx("epsilon"), "e.docx", db_user))
    with connection() as conn:
        vector = conn.execute(
            "SELECT embedding::text FROM chunks WHERE document_id = %s LIMIT 1", (record.id,)
        ).fetchone()[0]

    before = _search(db_user, vector)
    assert before and before[0][1] > 0.999  # the chunk finds itself

    assert store.delete_document(db_user, record.id) is True
    assert _search(db_user, vector) == []
    assert store.get_document(db_user, record.id) is None
    assert store.delete_document(db_user, record.id) is False  # already gone


def test_a_user_cannot_delete_or_see_anothers_document(
    user_factory: Callable[[], int], make_docx: Callable[[str], Path]
) -> None:
    owner, intruder = user_factory(), user_factory()
    record = asyncio.run(pipeline.ingest_file(make_docx("zeta"), "z.docx", owner))
    assert store.delete_document(intruder, record.id) is False
    assert store.get_document(intruder, record.id) is None
    assert store.list_documents(intruder) == []
    assert store.get_document(owner, record.id) is not None


def test_unsupported_file_stores_nothing(db_user: int, tmp_path: Path) -> None:
    bad = tmp_path / "program.exe"
    bad.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")
    with pytest.raises(UnsupportedFileTypeError):
        asyncio.run(pipeline.ingest_file(bad, "program.exe", db_user))
    assert store.list_documents(db_user) == []


def test_embedding_failure_stores_nothing(
    db_user: int, make_docx: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(texts, input_type="document"):  # noqa: ANN001
        raise EmbeddingServiceError("simulated outage")

    monkeypatch.setattr(pipeline, "embed_texts", broken)
    with pytest.raises(EmbeddingServiceError):
        asyncio.run(pipeline.ingest_file(make_docx("eta"), "eta.docx", db_user))
    assert store.list_documents(db_user) == []
