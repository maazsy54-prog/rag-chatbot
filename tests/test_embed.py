"""Tests for the embedding wrapper, using a fake client (no network, no API key)."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from app.config import settings
from app.ingestion.embed import BATCH_SIZE, embed_query, embed_texts
from app.ingestion.errors import EmbeddingServiceError


class FakeClient:
    """Mimics ``voyageai.AsyncClient.embed``; each vector starts with the text's number."""

    def __init__(self, *, dimension: int | None = None, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.dimension = dimension or settings.embedding_dimensions
        self.fail = fail

    async def embed(self, texts, model, input_type, output_dimension):  # noqa: ANN001
        self.calls.append({"size": len(texts), "input_type": input_type, "model": model})
        await asyncio.sleep(random.random() * 0.01)  # finish out of order on purpose
        if self.fail:
            raise TimeoutError("simulated timeout")
        vectors = [[float(int(text))] + [0.0] * (self.dimension - 1) for text in texts]
        return SimpleNamespace(embeddings=vectors)


def test_empty_input_makes_no_calls() -> None:
    client = FakeClient()
    assert asyncio.run(embed_texts([], client=client)) == []
    assert client.calls == []


def test_texts_are_batched_and_order_is_preserved() -> None:
    texts = [str(i) for i in range(BATCH_SIZE * 2 + 5)]
    client = FakeClient()
    vectors = asyncio.run(embed_texts(texts, client=client))
    assert len(client.calls) == 3
    assert [int(vector[0]) for vector in vectors] == list(range(len(texts)))


def test_input_type_and_model_are_passed_through() -> None:
    client = FakeClient()
    asyncio.run(embed_texts(["1"], "query", client=client))
    assert client.calls[0]["input_type"] == "query"
    assert client.calls[0]["model"] == settings.embedding_model


def test_embed_query_returns_one_vector() -> None:
    vector = asyncio.run(embed_query("7", client=FakeClient()))
    assert len(vector) == settings.embedding_dimensions and vector[0] == 7.0


def test_api_failure_becomes_embedding_service_error() -> None:
    with pytest.raises(EmbeddingServiceError):
        asyncio.run(embed_texts(["1", "2"], client=FakeClient(fail=True)))


def test_wrong_vector_size_is_rejected() -> None:
    with pytest.raises(EmbeddingServiceError):
        asyncio.run(embed_texts(["1"], client=FakeClient(dimension=8)))
