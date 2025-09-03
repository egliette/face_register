import numpy as np
from fastapi import status


class TestFaceEmbeddingDeletion:
    """Test cases for face embedding deletion functionality."""

    def test_delete_face_embedding_success(self, client, clean_db):
        """Test successful deletion of a face embedding."""

        user_payload = {"name": "Test User", "phone": "1234567890"}
        user_resp = client.post("/api/users/", json=user_payload)
        assert user_resp.status_code == status.HTTP_201_CREATED
        user_id = user_resp.json()["id"]

        from app.config.settings import settings
        from app.crud.face_embedding import create_face_embedding
        from app.schema.face_embedding import FaceEmbeddingCreate
        from app.services.qdrant import upsert_embedding

        vector_size = settings.QDRANT_VECTOR_SIZE
        embedding = np.random.rand(vector_size).astype(np.float32)
        record = create_face_embedding(
            clean_db, FaceEmbeddingCreate(user_id=user_id, embedding=embedding.tolist())
        )
        upsert_embedding(
            point_id=record.id, embedding=embedding.tolist(), user_id=user_id
        )

        resp = client.delete(f"/api/face-embeddings/{record.id}")
        assert resp.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_face_embedding_not_found(self, client, clean_db):
        """Test deletion of a non-existent face embedding (real call, empty DB)."""
        resp = client.delete("/api/face-embeddings/999")
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert resp.json()["detail"] == "Face embedding not found"


class TestUserDeletionWithEmbeddings:
    """Test cases for user deletion that also removes embeddings."""

    def test_delete_user_removes_embeddings(self, client, clean_db):
        """Test that deleting a user also removes their face embeddings."""

        user_payload = {"name": "Test User", "phone": "1234567890"}
        user_resp = client.post("/api/users/", json=user_payload)
        assert user_resp.status_code == status.HTTP_201_CREATED
        user_id = user_resp.json()["id"]

        from app.config.settings import settings
        from app.crud.face_embedding import (
            create_face_embedding,
            get_face_embeddings_by_user,
        )
        from app.schema.face_embedding import FaceEmbeddingCreate
        from app.services.qdrant import upsert_embedding

        vector_size = settings.QDRANT_VECTOR_SIZE
        for _ in range(2):
            vec = np.random.rand(vector_size).astype(np.float32)
            rec = create_face_embedding(
                clean_db,
                FaceEmbeddingCreate(user_id=user_id, embedding=vec.tolist()),
            )
            upsert_embedding(point_id=rec.id, embedding=vec.tolist(), user_id=user_id)

        resp = client.delete(f"/api/users/{user_id}")
        assert resp.status_code == status.HTTP_204_NO_CONTENT

        remaining = get_face_embeddings_by_user(clean_db, user_id)
        assert remaining == []


class TestFaceEmbeddingCRUD:
    """Test cases for face embedding CRUD operations."""

    def test_get_face_embeddings_by_user(self, client, clean_db):
        """Test retrieving face embeddings for a user."""

        user_payload = {"name": "Test User", "phone": "1234567890"}
        user_resp = client.post("/api/users/", json=user_payload)
        assert user_resp.status_code == status.HTTP_201_CREATED
        user_id = user_resp.json()["id"]

        from app.config.settings import settings
        from app.crud.face_embedding import create_face_embedding
        from app.schema.face_embedding import FaceEmbeddingCreate

        vec = np.random.rand(settings.QDRANT_VECTOR_SIZE).astype(np.float32)
        create_face_embedding(
            clean_db,
            FaceEmbeddingCreate(user_id=user_id, embedding=vec.tolist()),
        )

        resp = client.get(f"/api/users/{user_id}/face-embeddings")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert len(data) == 1
        assert data[0]["user_id"] == user_id

    def test_get_face_embeddings_user_not_found(self, client, clean_db):
        """Test retrieving face embeddings for a non-existent user."""
        resp = client.get("/api/users/999/face-embeddings")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert len(data) == 0
