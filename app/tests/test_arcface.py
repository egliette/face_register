import numpy as np
import pytest


@pytest.mark.model_dependent
def test_arcface_detect_returns_embedding(arcface_model):
    # Get model via helper and run detection on a dummy image with dummy landmarks

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

    arc = arcface_model
    embeddings = arc.detect(img, landmarks)

    # Type assertions only (do not require a specific embedding size)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    emb = embeddings[0]
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1


@pytest.mark.model_dependent
def test_arcface_detect_dynamic_batching_returns_embeddings_lists(arcface_model):
    # Get model via helper and run detection on a batch of dummy images with dummy landmarks

    imgs = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((720, 1280, 3), dtype=np.uint8),
        np.zeros((360, 360, 3), dtype=np.uint8),
    ]

    # Create dummy landmarks for each image
    landmarks_batch = []
    for img in imgs:
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
        landmarks_batch.append(landmarks)

    arc = arcface_model
    batch_embeddings = arc.detect(imgs, landmarks_batch)

    # Assertions on batched output structure and types
    assert isinstance(batch_embeddings, list)
    assert len(batch_embeddings) == len(imgs)
    # ArcFace returns a flat list of embeddings, not nested lists
    for emb in batch_embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1

    # Also verify tuple input works identically
    batch_embeddings_tuple = arc.detect(tuple(imgs), tuple(landmarks_batch))
    assert isinstance(batch_embeddings_tuple, list)
    assert len(batch_embeddings_tuple) == len(imgs)
