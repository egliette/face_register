import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from minio import Minio


@pytest.mark.model_dependent
def test_scrfd_detect_returns_faces():
    # Load environment from repo root .env (parent of tests folder)
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes", "y")

    bucket = os.getenv("SCRFD_BUCKET")
    object_name = os.getenv("SCRFD_MODEL_OBJECT")

    assert (
        endpoint and access_key and secret_key
    ), "MinIO credentials (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY) must be set in .env"
    assert (
        bucket and object_name
    ), "SCRFD bucket/object (SCRFD_BUCKET and SCRFD_MODEL_OBJECT) must be set in .env"

    client = Minio(
        endpoint, access_key=access_key, secret_key=secret_key, secure=secure
    )

    # Download model to assets/models/scrfd
    models_dir = repo_root / "assets" / "models" / "scrfd"
    models_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = models_dir / Path(object_name).name
    if not local_model_path.exists():
        client.fget_object(bucket, object_name, str(local_model_path))

    # Import SCRFD and run detection on a blank image
    from src.core.scrfd import SCRFD, Face

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    scrfd = SCRFD(model_file=str(local_model_path))
    faces = scrfd.detect(img)

    # Type assertions only (do not require a face to be present)
    assert isinstance(faces, list)
    assert all(isinstance(f, Face) for f in faces)
