import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.runtime import get_arcface, get_scrfd
from app.utils.logger import log


def validate_image_upload(
    image: UploadFile,
) -> tuple[bytes, np.ndarray]:
    """Validate and decode an uploaded image to a numpy array.

    Enforces exact content type membership for supported image formats.
    """
    try:
        allowed_content_types = {"image/jpeg", "image/png", "image/webp"}
        if image.content_type not in allowed_content_types:
            log.warn(f"Unsupported file type: {image.content_type}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Allowed: image/jpeg, image/png, image/webp",
            )

        image_data = image.file.read()
        if len(image_data) == 0:
            log.warn("Empty image upload")
            raise HTTPException(status_code=422, detail="Empty image file")

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            log.warn("Invalid image data")
            raise HTTPException(status_code=422, detail="Invalid image format")

        log.bug(f"Image validation successful, size: {img.shape}")
        return image_data, img
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(exc, "processing uploaded image")
        raise HTTPException(status_code=422, detail="Failed to process image")


def detect_face_in_image(img: np.ndarray):
    """Detect exactly one face and ensure landmarks exist."""
    try:
        scrfd = get_scrfd()
        faces = scrfd.detect(img, max_num=1)
        log.bug(f"Face detection completed, found {len(faces)} faces")

        if len(faces) == 0:
            log.warn("No face detected in image")
            raise HTTPException(status_code=422, detail="No face detected")
        if len(faces) > 1:
            log.warn("Multiple faces detected in image")
            raise HTTPException(status_code=422, detail="Multiple faces detected")

        face = faces[0]
        if getattr(face, "keypoint", None) is None:
            log.warn("No landmarks available")
            raise HTTPException(
                status_code=422, detail="No landmarks available for alignment"
            )

        log.bug(f"Face detection successful, score: {getattr(face, 'score', None)}")
        return face
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(exc, "face detection")
        raise HTTPException(status_code=500, detail="Face detection failed")


def extract_face_embedding(img: np.ndarray, face) -> np.ndarray:
    """Extract face embedding given the face landmarks."""
    try:
        arc = get_arcface()
        embedding = arc.detect(img, landmarks=face.keypoint)
        log.bug(f"Face embedding extracted, dimension: {len(embedding)}")
        return embedding
    except Exception as exc:
        log.exception(exc, "face embedding extraction")
        raise HTTPException(status_code=500, detail="Face embedding extraction failed")
