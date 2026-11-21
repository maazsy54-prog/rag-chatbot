"""Shared pytest fixtures: throwaway database users, fake embeddings, sample files."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from docx import Document

import asyncio

from app import retrieval
from app.config import settings
from app.db import connection
from app.ingestion import pipeline
from app.ingestion.errors import DatabaseUnavailableError


def fake_vector(text: str) -> list[float]:
    """A deterministic stand-in embedding: the same text always gives the same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
    base = [(byte - 127.5) / 127.5 for byte in digest]
    return (base * (settings.embedding_dimensions // len(base) + 1))[: settings.embedding_dimensions]


async def fake_embedder(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Replacement for the Voyage call, so tests need no API key or network."""
    return [fake_vector(text) for text in texts]


_TOPICS = ("refund", "leave", "security", "travel")


def topic_vector(text: str) -> list[float]:
    """A tiny hand-made 'embedding': one dimension per topic word, plus a constant.

    Texts about the same topic get a cosine similarity near 1, unrelated texts
    about 0.1, so tests can predict exactly what retrieval should return.
    """
    lowered = text.lower()
    vector = [0.0] * settings.embedding_dimensions
    for position, topic in enumerate(_TOPICS):
        if topic in lowered:
            vector[position] = 1.0
    vector[100] = 0.1  # keeps every vector non-zero
    return vector


async def topic_embedder(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Topic-based stand-in for the Voyage call used when ingesting documents."""
    return [topic_vector(text) for text in texts]


async def topic_query_embedder(question: str, **_: object) -> list[float]:
    """Topic-based stand-in for embedding a question."""
    return topic_vector(question)


@pytest.fixture()
def user_factory() -> Iterator[Callable[[], int]]:
    """Create throwaway users; they (and all their data) are deleted afterwards."""
    created: list[int] = []

    def make() -> int:
        email = f"pytest-{uuid.uuid4().hex[:10]}@local"
        try:
            with connection() as conn:
                row = conn.execute(
                    "INSERT INTO users (email) VALUES (%s) RETURNING id", (email,)
                ).fetchone()
        except DatabaseUnavailableError:
            pytest.skip("Postgres is not running (start it with: docker compose up -d)")
        created.append(row[0])
        return row[0]

    yield make
    if created:
        with connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ANY(%s)", (created,))


@pytest.fixture()
def db_user(user_factory: Callable[[], int]) -> int:
    """One throwaway user id."""
    return user_factory()


@pytest.fixture()
def make_docx(tmp_path: Path) -> Callable[[str], Path]:
    """Build a Word file whose content is unique per ``marker`` (so hashes differ)."""

    def build(marker: str) -> Path:
        document = Document()
        document.add_heading(f"Policy {marker}", level=1)
        document.add_heading("Refunds", level=2)
        document.add_paragraph(f"Refunds for {marker} are paid within fourteen days. " * 20)
        document.add_heading("Leave", level=2)
        document.add_paragraph(f"Staff of {marker} receive twenty days of annual leave. " * 20)
        path = tmp_path / f"{marker}.docx"
        document.save(path)
        return path

    return build


def make_word_doc(tmp_path: Path, name: str, sections: dict[str, str]) -> Path:
    """Create a Word file with one level-2 heading and one paragraph per section."""
    document = Document()
    document.add_heading(name, level=1)
    for heading, body in sections.items():
        document.add_heading(heading, level=2)
        document.add_paragraph(body)
    path = tmp_path / f"{name}.docx"
    document.save(path)
    return path


def ingest_sync(user_id: int, path: Path) -> int:
    """Ingest a file for a user and return the new document id."""
    return asyncio.run(pipeline.ingest_file(path, path.name, user_id)).id


@pytest.fixture()
def topic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use topic-based fake embeddings for ingestion and for questions."""
    monkeypatch.setattr(pipeline, "embed_texts", topic_embedder)
    monkeypatch.setattr(retrieval, "embed_query", topic_query_embedder)


@pytest.fixture()
def handbook(db_user: int, tmp_path: Path, topic_embeddings: None) -> tuple[int, int]:
    """A stored document with refund, travel and security sections; returns (user, doc id)."""
    path = make_word_doc(
        tmp_path,
        "handbook",
        {
            "Refunds": "Refund requests are processed within fourteen days of the claim. " * 12,
            "Travel": "Travel expenses need manager approval before any booking is made. " * 12
            + "Reference code ZX-4417 applies to hotel bookings.",
            "Security": "Security badges must be worn visibly inside every building. " * 12,
        },
    )
    return db_user, ingest_sync(db_user, path)
