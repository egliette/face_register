import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from minio import Minio


def test_arcface_detect_returns_embedding():
    # Load environment from repo root .env (parent of tests folder)
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes", "y")

    bucket = os.getenv("ARCFACE_BUCKET")
    object_name = os.getenv("ARCFACE_MODEL_OBJECT")

    assert (
        endpoint and access_key and secret_key
    ), "MinIO credentials (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY) must be set in .env"
    assert (
        bucket and object_name
    ), "ArcFace bucket/object (ARCFACE_BUCKET and ARCFACE_MODEL_OBJECT) must be set in .env"

    client = Minio(
        endpoint, access_key=access_key, secret_key=secret_key, secure=secure
    )

    # Download model to assets/models/arcface
    models_dir = repo_root / "assets" / "models" / "arcface"
    models_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = models_dir / Path(object_name).name
    if not local_model_path.exists():
        client.fget_object(bucket, object_name, str(local_model_path))

    # Import ArcFace and run detection on a dummy image with dummy landmarks
    from src.core.arcface import ArcFace

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    landmarks = np.array(
        [
            [w * 0.35, h * 0.35],
            [w * 0.65, h * 0.35],
            [w * 0.50, h * 0.55],
            [w * 0.40, h * 0.75],
            [w * 0.60, h * 0.75],
        ],
        dtype=np.float32,
    )

    arc = ArcFace(model_file=str(local_model_path))
    emb = arc.detect(img, landmarks)

    # Type assertions only (do not require a specific embedding size)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
