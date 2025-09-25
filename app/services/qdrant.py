from functools import lru_cache
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config.settings import settings
from app.utils.logger import log


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30.0
    )


def ensure_collection(vector_size: Optional[int] = None) -> None:
    client = get_qdrant()
    size = vector_size or settings.QDRANT_VECTOR_SIZE
    try:
        exists = client.collection_exists(settings.QDRANT_COLLECTION)
    except Exception as e:
        log.exception(
            e, f"checking collection existence for {settings.QDRANT_COLLECTION}"
        )
        exists = False
    if not exists:
        try:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=qm.VectorParams(size=size, distance=qm.Distance.COSINE),
            )
            log.info(
                f"Created Qdrant collection {settings.QDRANT_COLLECTION} with vector size {size}"
            )
        except Exception as e:
            log.exception(e, f"creating collection {settings.QDRANT_COLLECTION}")
            raise


def upsert_embedding(point_id: int, embedding: List[float], user_id: int) -> None:
    client = get_qdrant()
    ensure_collection(vector_size=len(embedding))
    try:
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
    except Exception as e:
        log.exception(e, f"upserting embedding for user {user_id}")
        raise


def delete_embedding(point_id: int) -> None:
    """Delete an embedding from Qdrant collection by point ID."""
    client = get_qdrant()
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=qm.PointIdsList(points=[point_id]),
            wait=True,
        )
    except Exception as e:
        log.exception(e, f"deleting embedding with point_id {point_id}")


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
    except Exception as e:
        log.exception(e, f"deleting embeddings for user {user_id}")


def search_similar_embeddings(
    query_embedding: List[float], limit: int = 10, score_threshold: float = 0.0
) -> List[qm.ScoredPoint]:
    """Search for similar embeddings in Qdrant collection.

    Args:
        query_embedding: The embedding vector to search for
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score threshold

    Returns:
        List of ScoredPoint objects with similar embeddings
    """
    client = get_qdrant()
    try:
        results = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        ).points
        return results
    except Exception as e:
        log.exception(e, f"searching similar embeddings")
        raise
