from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.model_dependent
def test_face_comparison_success(clean_db, test_user, client):
    """Test successful face comparison with enrolled user"""

    face_image_path = Path("assets/images/face.png")
    assert face_image_path.exists(), f"face.png image must exist in {face_image_path}"

    with open(face_image_path, "rb") as f:
        enroll_resp = client.post(
            f"/api/users/{test_user.id}/enroll",
            files={"image": ("face.png", f, "image/png")},
        )
    assert enroll_resp.status_code == 201

    with open(face_image_path, "rb") as f:
        response = client.post(
            "/api/face-comparison",
            files={"image": ("face.png", f, "image/png")},
            params={"threshold": 0.6, "limit": 10},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    match = data[0]

    assert "user" in match
    assert "face_embedding_id" in match
    assert "image_path" in match
    assert "similarity_score" in match
    assert "created_at" in match

    user = match["user"]
    assert "id" in user
    assert "name" in user
    assert "phone" in user
    assert user["id"] == test_user.id
    assert user["name"] == test_user.name
    assert user["phone"] == test_user.phone

    assert 0.0 <= match["similarity_score"] <= 1.0
    assert match["similarity_score"] >= 0.6


@pytest.mark.model_dependent
def test_face_comparison_no_matches(clean_db, client):
    face_image_path = Path("assets/images/face.png")
    assert face_image_path.exists(), f"face.png image must exist in {face_image_path}"

    with open(face_image_path, "rb") as f:
        response = client.post(
            "/api/face-comparison",
            files={"image": ("face.png", f, "image/png")},
            params={"threshold": 0.6, "limit": 10},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_face_comparison_invalid_threshold_low(client):
    """Test face comparison with threshold below valid range"""
    face_image_path = Path("assets/images/face.png")
    if not face_image_path.exists():
        pytest.skip("face.png image not available")

    with open(face_image_path, "rb") as f:
        response = client.post(
            "/api/face-comparison",
            files={"image": ("face.png", f, "image/png")},
            params={"threshold": -0.1, "limit": 10},
        )

    assert response.status_code == 422


def test_face_comparison_invalid_limit_low(client):
    """Test face comparison with limit below valid range"""
    face_image_path = Path("assets/images/face.png")
    if not face_image_path.exists():
        pytest.skip("face.png image not available")

    with open(face_image_path, "rb") as f:
        response = client.post(
            "/api/face-comparison",
            files={"image": ("face.png", f, "image/png")},
            params={"threshold": 0.6, "limit": 0},
        )

    assert response.status_code == 422


@pytest.mark.model_dependent
def test_face_comparison_no_face_detected(client):
    """Test face comparison when no face is detected in image"""

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode(".png", blank_img)

    response = client.post(
        "/api/face-comparison",
        files={"image": ("blank.png", img_bytes.tobytes(), "image/png")},
        params={"threshold": 0.6, "limit": 10},
    )

    assert response.status_code == 422


def test_face_comparison_empty_image(client):
    """Test face comparison with empty image"""
    response = client.post(
        "/api/face-comparison",
        files={"image": ("empty.png", b"", "image/png")},
        params={"threshold": 0.6, "limit": 10},
    )

    assert response.status_code == 422


def test_face_comparison_invalid_file_type(client):
    """Test face comparison with non-image file"""
    response = client.post(
        "/api/face-comparison",
        files={"image": ("test.txt", b"not an image", "text/plain")},
        params={"threshold": 0.6, "limit": 10},
    )

    assert response.status_code == 415
