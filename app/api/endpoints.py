import time
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.runtime import get_arcface, get_scrfd
from app.crud.face_embedding import (
    create_face_embedding,
    delete_face_embedding,
    get_face_embedding,
    get_face_embeddings_by_user,
)
from app.crud.user import create_user, delete_user, get_user, get_users, update_user
from app.database.connection import get_db
from app.schema.face_embedding import FaceEmbedding, FaceEmbeddingCreate
from app.schema.user import User, UserCreate, UserUpdate
from app.services.qdrant import delete_embedding, upsert_embedding
from app.utils.logger import log

router = APIRouter()


@router.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        result = create_user(db=db, user=user)
        log.info(f"Successfully created user {user.name} with ID: {result.id}")
        return result
    except Exception as e:
        log.exception(e, f"creating user {user.name}")
        raise


@router.get("/users/", response_model=List[User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=skip, limit=limit)
    return users


@router.get("/users/{user_id}", response_model=User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        log.warn(f"User not found: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return db_user


@router.put("/users/{user_id}", response_model=User)
def update_user_info(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    updated_user = update_user(db=db, user_id=user_id, user=user)
    if updated_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return updated_user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_by_id(user_id: int, db: Session = Depends(get_db)):
    success = delete_user(db=db, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return None


@router.post(
    "/users/{user_id}/enroll",
    response_model=FaceEmbedding,
    status_code=status.HTTP_201_CREATED,
)
def enroll_face(
    user_id: int,
    image: UploadFile = File(..., media_type="image/*"),
    db: Session = Depends(get_db),
):
    start_time = time.time()
    log.info(f"Starting face enrollment for user {user_id}, file: {image.filename}")

    allowed_content_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_content_types:
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
    except Exception as e:
        log.exception(e, f"processing image for user {user_id}")
        raise

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

        log.bug(f"Face detection successful for user {user_id}, score: {face.score}")
    except HTTPException:
        raise
    except Exception as e:
        log.exception(e, f"face detection for user {user_id}")
        raise

    try:
        arc = get_arcface()
        embedding = arc.detect(img, landmarks=face.keypoint)
        log.bug(
            f"Face embedding extracted for user {user_id}, dimension: {len(embedding)}"
        )
    except Exception as e:
        log.exception(e, f"face embedding extraction for user {user_id}")
        raise

    try:
        record = create_face_embedding(
            db, FaceEmbeddingCreate(user_id=user_id, embedding=embedding.tolist())
        )

        upsert_embedding(
            point_id=record.id, embedding=embedding.tolist(), user_id=user_id
        )
        duration = time.time() - start_time
        log.perf("face_enrollment", duration, user_id=user_id, file_size=len(data))

        return record
    except Exception as e:
        log.exception(e, f"saving face embedding for user {user_id}")
        raise


@router.get(
    "/users/{user_id}/face-embeddings",
    response_model=List[FaceEmbedding],
)
def list_user_face_embeddings(user_id: int, db: Session = Depends(get_db)):
    return get_face_embeddings_by_user(db, user_id=user_id)


@router.delete(
    "/face-embeddings/{embedding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_face_embedding_by_id(embedding_id: int, db: Session = Depends(get_db)):
    """Delete a specific face embedding by ID."""
    # First get the embedding to check if it exists
    embedding = get_face_embedding(db, embedding_id)
    if not embedding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Face embedding not found"
        )

    delete_embedding(embedding_id)

    success = delete_face_embedding(db, embedding_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete face embedding",
        )

    return None
