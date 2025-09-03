from typing import Dict, List

from fastapi import status


def test_create_user(client):
    payload = {"name": "Alice", "phone": "1234567890"}
    resp = client.post("/api/users/", json=payload)
    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    assert data["name"] == payload["name"]
    assert data["phone"] == payload["phone"]
    assert "id" in data


def test_list_users(client):
    resp = client.get("/api/users/")
    assert resp.status_code == status.HTTP_200_OK
    users: List[Dict] = resp.json()
    assert isinstance(users, list)


def test_get_user_found(client):
    create = client.post("/api/users/", json={"name": "Bob", "phone": "999"})
    user_id = create.json()["id"]

    resp = client.get(f"/api/users/{user_id}")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["id"] == user_id


def test_get_user_not_found(client):
    resp = client.get("/api/users/999999")
    assert resp.status_code == status.HTTP_404_NOT_FOUND


def test_update_user(client):
    create = client.post("/api/users/", json={"name": "Carol", "phone": "888"})
    user_id = create.json()["id"]

    resp = client.put(f"/api/users/{user_id}", json={"name": "Carolyn"})
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["name"] == "Carolyn"


def test_delete_user(client):
    create = client.post("/api/users/", json={"name": "Dave", "phone": "777"})
    user_id = create.json()["id"]

    resp = client.delete(f"/api/users/{user_id}")
    assert resp.status_code == status.HTTP_204_NO_CONTENT

    # ensure now 404
    resp2 = client.get(f"/api/users/{user_id}")
    assert resp2.status_code == status.HTTP_404_NOT_FOUND


def test_delete_user_removes_embeddings(client, clean_db):
    """Test that deleting a user also removes their face embeddings."""

    create = client.post("/api/users/", json={"name": "Eve", "phone": "666"})
    user_id = create.json()["id"]

    # Create real embeddings and upsert to Qdrant
    import numpy as np

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

    # Delete the user and assert success
    resp = client.delete(f"/api/users/{user_id}")
    assert resp.status_code == status.HTTP_204_NO_CONTENT

    # Verify user is gone
    resp2 = client.get(f"/api/users/{user_id}")
    assert resp2.status_code == status.HTTP_404_NOT_FOUND

    # Verify DB embeddings are removed
    remaining = get_face_embeddings_by_user(clean_db, user_id)
    assert remaining == []
