import numpy as np
import pytest


@pytest.mark.model_dependent
def test_scrfd_detect_returns_faces(scrfd_model):
    # Get model via helper and run detection on a blank image
    from app.core.models.scrfd import Face

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    scrfd = scrfd_model
    faces = scrfd.detect(img)

    # Type assertions only (do not require a face to be present)
    assert isinstance(faces, list)
    assert len(faces) == 1
    assert isinstance(faces[0], list)
    assert all(isinstance(f, Face) for f in faces[0])


@pytest.mark.model_dependent
def test_scrfd_detect_dynamic_batching_returns_faces_lists(scrfd_model):
    # Get model via helper and run detection on a batch of blank images of different sizes
    from app.core.models.scrfd import Face

    imgs = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((720, 1280, 3), dtype=np.uint8),
        np.zeros((360, 360, 3), dtype=np.uint8),
    ]

    scrfd = scrfd_model
    batch_faces = scrfd.detect(imgs)

    # Assertions on batched output structure and types
    assert isinstance(batch_faces, list)
    assert len(batch_faces) == len(imgs)
    assert all(isinstance(per_img, list) for per_img in batch_faces)
    for per_img in batch_faces:
        assert all(isinstance(f, Face) for f in per_img)

    # Also verify tuple input works identically
    batch_faces_tuple = scrfd.detect(tuple(imgs))
    assert isinstance(batch_faces_tuple, list)
    assert len(batch_faces_tuple) == len(imgs)
