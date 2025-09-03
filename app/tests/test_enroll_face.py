import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from minio import Minio


@pytest.mark.model_dependent
def test_face_detection_and_embedding():
    """Test face detection and embedding extraction with real models and face image"""
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes", "y")

    scrfd_bucket = os.getenv("SCRFD_BUCKET")
    scrfd_object = os.getenv("SCRFD_MODEL_OBJECT")
    arcface_bucket = os.getenv("ARCFACE_BUCKET")
    arcface_object = os.getenv("ARCFACE_MODEL_OBJECT")

    assert (
        endpoint and access_key and secret_key
    ), "MinIO credentials must be set in .env"
    assert scrfd_bucket and scrfd_object, "SCRFD bucket/object must be set in .env"
    assert (
        arcface_bucket and arcface_object
    ), "ArcFace bucket/object must be set in .env"

    client_minio = Minio(
        endpoint, access_key=access_key, secret_key=secret_key, secure=secure
    )

    scrfd_models_dir = Path("assets/models/scrfd")
    scrfd_models_dir.mkdir(parents=True, exist_ok=True)
    scrfd_local_path = scrfd_models_dir / Path(scrfd_object).name
    if not scrfd_local_path.exists():
        client_minio.fget_object(scrfd_bucket, scrfd_object, str(scrfd_local_path))

    arcface_models_dir = Path("assets/models/arcface")
    arcface_models_dir.mkdir(parents=True, exist_ok=True)
    arcface_local_path = arcface_models_dir / Path(arcface_object).name
    if not arcface_local_path.exists():
        client_minio.fget_object(
            arcface_bucket, arcface_object, str(arcface_local_path)
        )

    face_image_path = Path("assets/images/face.png")
    assert face_image_path.exists(), f"face.png image must exist in {face_image_path}"

    import cv2
    import numpy as np

    img = cv2.imread(str(face_image_path))
    assert img is not None, "Failed to load face image"

    from app.core.scrfd import SCRFD

    scrfd = SCRFD(model_file=str(scrfd_local_path))
    faces = scrfd.detect(img, max_num=1)

    assert len(faces) == 1, f"Expected 1 face, got {len(faces)}"
    face = faces[0]
    assert hasattr(face, "keypoint"), "Face should have landmarks"
    assert face.keypoint is not None, "Landmarks should not be None"
    assert face.keypoint.shape == (
        5,
        2,
    ), f"Expected landmarks shape (5, 2), got {face.keypoint.shape}"

    from app.core.arcface import ArcFace

    arc = ArcFace(model_file=str(arcface_local_path))
    embedding = arc.detect(img, landmarks=face.keypoint)

    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.ndim == 1, f"Expected 1D embedding, got {embedding.ndim}D"
    assert embedding.shape[0] > 0, "Embedding should have positive length"


@pytest.mark.model_dependent
def test_enroll_face_api_success(clean_db, test_user, client):
    """Test successful face enrollment through API endpoint"""
    face_image_path = Path("assets/images/face.png")
    assert face_image_path.exists(), f"face.png image must exist in {face_image_path}"

    with open(face_image_path, "rb") as f:
        response = client.post(
            f"/api/users/{test_user.id}/enroll",
            files={"image": ("face.png", f, "image/png")},
        )

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "user_id" in data
    assert "created_at" in data
    assert data["user_id"] == test_user.id


@pytest.mark.model_dependent
def test_enroll_face_api_no_face_detected(clean_db, test_user, client):
    """Test API error when no face is detected in image"""
    import cv2
    import numpy as np

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode(".png", blank_img)

    response = client.post(
        f"/api/users/{test_user.id}/enroll",
        files={"image": ("blank.png", img_bytes.tobytes(), "image/png")},
    )

    assert response.status_code == 422
    assert "No face detected" in response.json()["detail"]


@pytest.mark.model_dependent
def test_enroll_face_api_empty_image(clean_db, test_user, client):
    """Test API error when empty image is uploaded"""

    response = client.post(
        f"/api/users/{test_user.id}/enroll",
        files={"image": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "Empty image upload" in response.json()["detail"]
