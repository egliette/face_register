import time
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.runtime import get_arcface, get_scrfd
from app.crud.face_embedding import create_face_embedding
from app.schema.face_embedding import (
    FaceEmbedding,
    FaceEmbeddingCreate,
    FaceEmbeddingPublic,
)
from app.services.minio import minio_service
from app.services.qdrant import upsert_embedding
from app.utils.logger import log


class FaceEnrollmentService:
    """Service for handling face enrollment operations."""

    def __init__(self):
        self.allowed_content_types = {"image/jpeg", "image/png", "image/webp"}

    def validate_image_upload(
        self, image: UploadFile, user_id: int
    ) -> tuple[bytes, np.ndarray]:
        """Validate and process the uploaded image file."""
        if image.content_type not in self.allowed_content_types:
            log.warn(f"Unsupported file type for user {user_id}: {image.content_type}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Allowed: image/jpeg, image/png, image/webp",
            )

        try:
            data = image.file.read()
            if not data:
                log.warn(f"Empty image upload for user {user_id}")
                raise HTTPException(status_code=400, detail="Empty image upload")

            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                log.warn(f"Invalid image data for user {user_id}")
                raise HTTPException(status_code=400, detail="Invalid image data")

            log.bug(f"Image loaded successfully for user {user_id}, size: {img.shape}")
            return data, img
        except Exception as e:
            log.exception(e, f"processing image for user {user_id}")
            raise

    def detect_face_in_image(self, img: np.ndarray, user_id: int):
        """Detect and validate face in the image."""
        try:
            scrfd = get_scrfd()
            faces = scrfd.detect(img, max_num=1)
            log.bug(
                f"Face detection completed for user {user_id}, found {len(faces)} faces"
            )

            if len(faces) == 0:
                log.warn(f"No face detected in image for user {user_id}")
                raise HTTPException(status_code=422, detail="No face detected")
            if len(faces) > 1:
                log.warn(f"Multiple faces detected in image for user {user_id}")
                raise HTTPException(status_code=422, detail="Multiple faces detected")

            face = faces[0]
            if getattr(face, "keypoint", None) is None:
                log.warn(f"No landmarks available for user {user_id}")
                raise HTTPException(
                    status_code=422, detail="No landmarks available for alignment"
                )

            log.bug(
                f"Face detection successful for user {user_id}, score: {face.score}"
            )
            return face
        except Exception as e:
            log.exception(e, f"face detection for user {user_id}")
            raise

    def extract_face_embedding(self, img: np.ndarray, face, user_id: int) -> np.ndarray:
        """Extract face embedding from the detected face."""
        try:
            arc = get_arcface()
            embedding = arc.detect(img, landmarks=face.keypoint)
            log.bug(
                f"Face embedding extracted for user {user_id}, dimension: {len(embedding)}"
            )
            return embedding
        except Exception as e:
            log.exception(e, f"face embedding extraction for user {user_id}")
            raise

    def upload_image_to_minio(
        self,
        user_id: int,
        face_embedding_id: int,
        image_data: bytes,
        content_type: str,
        original_filename: str,
    ) -> Optional[str]:
        """Upload image to MinIO and return the object path."""
        try:
            object_name = minio_service.upload_face_image(
                user_id=user_id,
                face_embedding_id=face_embedding_id,
                image_data=image_data,
                content_type=content_type,
                original_filename=original_filename,
            )

            log.info(
                f"Successfully uploaded face image to MinIO for user {user_id}, embedding {face_embedding_id}"
            )
            return object_name

        except Exception as e:
            log.error(f"Failed to upload image to MinIO for user {user_id}: {e}")
            log.warn(
                f"Face enrollment will continue without image storage for user {user_id}"
            )
            return None

    def save_face_embedding(
        self,
        db: Session,
        user_id: int,
        embedding: np.ndarray,
        image_path: Optional[str],
    ) -> FaceEmbedding:
        """Save face embedding to database and vector store."""
        try:
            # Create the face embedding record
            record = create_face_embedding(
                db,
                FaceEmbeddingCreate(
                    user_id=user_id, embedding=embedding.tolist(), image_path=image_path
                ),
            )

            # Store in vector database
            upsert_embedding(
                point_id=record.id, embedding=embedding.tolist(), user_id=user_id
            )

            return record
        except Exception as e:
            log.exception(e, f"saving face embedding for user {user_id}")
            raise

    def enroll_face(
        self, user_id: int, image: UploadFile, db: Session
    ) -> FaceEmbeddingPublic:
        """Enroll a user's face by processing the uploaded image"""
        start_time = time.time()
        log.info(f"Starting face enrollment for user {user_id}, file: {image.filename}")

        image_data, img = self.validate_image_upload(image, user_id)
        face = self.detect_face_in_image(img, user_id)
        embedding = self.extract_face_embedding(img, face, user_id)

        record = self.save_face_embedding(db, user_id, embedding, None)
        image_path = self.upload_image_to_minio(
            user_id=user_id,
            face_embedding_id=record.id,
            image_data=image_data,
            content_type=image.content_type,
            original_filename=image.filename,
        )

        if image_path:
            record.image_path = image_path
            db.commit()
            db.refresh(record)

        duration = time.time() - start_time
        log.perf(
            "face_enrollment", duration, user_id=user_id, file_size=len(image_data)
        )

        return FaceEmbeddingPublic.model_validate(record)


face_enrollment_service = FaceEnrollmentService()
