from functools import lru_cache
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config.settings import settings


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=10.0
    )


def ensure_collection(vector_size: Optional[int] = None) -> None:
    client = get_qdrant()
    size = vector_size or settings.QDRANT_VECTOR_SIZE
    try:
        exists = client.collection_exists(settings.QDRANT_COLLECTION)
    except Exception:
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=size, distance=qm.Distance.COSINE),
        )


def upsert_embedding(point_id: int, embedding: List[float], user_id: int) -> None:
    client = get_qdrant()
    ensure_collection(vector_size=len(embedding))
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            qm.PointStruct(
                id=point_id,
                vector=embedding,
                payload={"user_id": user_id},
            )
        ],
        wait=True,
    )


def delete_embedding(point_id: int) -> None:
    """Delete an embedding from Qdrant collection by point ID."""
    client = get_qdrant()
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=qm.PointIdsList(points=[point_id]),
            wait=True,
        )
    except Exception:
        pass


def delete_embeddings_by_user(user_id: int) -> None:
    """Delete all embeddings for a specific user from Qdrant collection."""
    client = get_qdrant()
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="user_id", match=qm.MatchValue(value=user_id)
                        )
                    ]
                )
            ),
            wait=True,
        )
    except Exception:
        pass
