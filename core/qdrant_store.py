import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "DEBI_FACE_RECO")
VECTOR_SIZE = 512  # ArcFace embedding dimension


class QdrantStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}

        if COLLECTION in existing:
            # Recreate if vector size changed (e.g. 128-d → 512-d migration)
            info = self.client.get_collection(COLLECTION)
            current_size = info.config.params.vectors.size
            if current_size != VECTOR_SIZE:
                self.client.delete_collection(COLLECTION)
                existing.discard(COLLECTION)

        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

        # Qdrant Cloud requires a payload index to filter by a field.
        # This is idempotent — safe to call on every startup.
        self.client.create_payload_index(
            collection_name=COLLECTION,
            field_name="name",
            field_schema="keyword",
        )

    def upsert(self, encoding: list[float], name: str) -> None:
        self.client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=encoding,
                    payload={"name": name},
                )
            ],
        )

    def search(self, encoding: list[float], limit: int = 1) -> tuple[str, float]:
        results = self.client.search(
            collection_name=COLLECTION,
            query_vector=encoding,
            limit=limit,
        )
        if not results:
            return "Unknown", 0.0
        best = results[0]
        return best.payload["name"], best.score

    def list_persons(self) -> list[str]:
        records, _ = self.client.scroll(
            collection_name=COLLECTION,
            limit=500,
            with_payload=True,
        )
        return sorted({r.payload["name"] for r in records})

    def delete_person(self, name: str) -> None:
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="name", match=MatchValue(value=name))]
                )
            ),
        )
