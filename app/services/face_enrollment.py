import time
from typing import Optional

import numpy as np
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.crud.face_embedding import create_face_embedding
from app.schema.face_embedding import (
    FaceEmbedding,
    FaceEmbeddingCreate,
    FaceEmbeddingPublic,
)
from app.services.minio import minio_service
from app.services.qdrant import upsert_embedding
from app.utils.face_service import (
    detect_face_in_image,
    extract_face_embedding,
    validate_image_upload,
)
from app.utils.logger import log


class FaceEnrollmentService:
    """Service for handling face enrollment operations."""

    def upload_image_to_minio(
        self,
        user_id: int,
        face_embedding_id: int,
        image_data: bytes,
        content_type: str,
    ) -> Optional[str]:
        """Upload image to MinIO and return the object path."""
        try:
            object_name = minio_service.upload_face_image(
                user_id=user_id,
                face_embedding_id=face_embedding_id,
                image_data=image_data,
                content_type=content_type,
            )

            log.info(
                f"Successfully uploaded face image to MinIO for user {user_id}, embedding {face_embedding_id}"
            )
            return object_name

        except Exception as e:
            log.err(f"Failed to upload image to MinIO for user {user_id}: {e}")
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

        try:
            image_data, img = validate_image_upload(image=image)
            face = detect_face_in_image(img)
            embedding = extract_face_embedding(img, face)
        except Exception as e:
            log.exception(e, f"face enrollment for user {user_id}")
            raise

        record = self.save_face_embedding(db, user_id, embedding, None)
        image_path = self.upload_image_to_minio(
            user_id=user_id,
            face_embedding_id=record.id,
            image_data=image_data,
            content_type=image.content_type,
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
