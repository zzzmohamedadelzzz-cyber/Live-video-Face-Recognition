import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")

EMBED_DIM = 512  # ArcFace


@pytest.fixture()
def store():
    with patch("core.qdrant_store.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Pretend collection already exists with the correct vector size
        existing = MagicMock()
        existing.name = "DEBI_FACE_RECO"
        mock_client.get_collections.return_value = MagicMock(collections=[existing])

        info = MagicMock()
        info.config.params.vectors.size = EMBED_DIM
        mock_client.get_collection.return_value = info

        from core.qdrant_store import QdrantStore

        s = QdrantStore()
        s.client = mock_client
        return s


def test_upsert_calls_client(store):
    store.upsert([0.1] * EMBED_DIM, "Alice")
    store.client.upsert.assert_called_once()
    assert store.client.upsert.call_args.kwargs["collection_name"] == "DEBI_FACE_RECO"


def test_search_returns_best_match(store):
    hit = MagicMock()
    hit.payload = {"name": "Bob"}
    hit.score = 0.93
    store.client.search.return_value = [hit]

    name, score = store.search([0.5] * EMBED_DIM)
    assert name == "Bob"
    assert score == 0.93


def test_search_empty_returns_unknown(store):
    store.client.search.return_value = []
    name, score = store.search([0.5] * EMBED_DIM)
    assert name == "Unknown"
    assert score == 0.0


def test_list_persons_deduplicates(store):
    def make_record(n):
        r = MagicMock()
        r.payload = {"name": n}
        return r

    store.client.scroll.return_value = (
        [make_record("Alice"), make_record("Bob"), make_record("Alice")],
        None,
    )
    persons = store.list_persons()
    assert persons == ["Alice", "Bob"]


def test_delete_person_calls_client(store):
    store.delete_person("Carol")
    store.client.delete.assert_called_once()
    assert store.client.delete.call_args.kwargs["collection_name"] == "DEBI_FACE_RECO"
