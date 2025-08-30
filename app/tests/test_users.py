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
    # This assertion will always fail to demonstrate exit code propagation
    assert False, "This test is intentionally failing to show exit code behavior"


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
