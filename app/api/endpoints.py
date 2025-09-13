from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.crud.face_embedding import (
    delete_face_embedding,
    get_face_embedding,
    get_face_embeddings_by_user,
)
from app.crud.user import create_user, delete_user, get_user, get_users, update_user
from app.database.connection import get_db
from app.schema.face_comparison import FaceMatch
from app.schema.face_embedding import FaceEmbeddingPublic
from app.schema.media import FaceImageURL
from app.schema.user import User, UserCreate, UserUpdate
from app.services.face_comparison import face_comparison_service
from app.services.face_enrollment import face_enrollment_service
from app.services.minio import minio_service
from app.services.qdrant import delete_embedding
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
    response_model=FaceEmbeddingPublic,
    status_code=status.HTTP_201_CREATED,
)
def enroll_face(
    user_id: int,
    image: UploadFile = File(..., media_type="image/*"),
    db: Session = Depends(get_db),
):
    """Enroll a user's face by processing the uploaded image."""
    return face_enrollment_service.enroll_face(user_id, image, db)


@router.get(
    "/users/{user_id}/face-embeddings",
    response_model=List[FaceEmbeddingPublic],
)
def list_user_face_embeddings(user_id: int, db: Session = Depends(get_db)):
    return get_face_embeddings_by_user(db, user_id=user_id)


@router.get(
    "/users/{user_id}/face-images",
    response_model=List[FaceImageURL],
)
def list_user_face_images(user_id: int, db: Session = Depends(get_db)):
    """Return only presigned URLs for user's face images without exposing internal paths."""
    records = get_face_embeddings_by_user(db, user_id=user_id)
    urls: List[FaceImageURL] = []
    for rec in records:
        path = getattr(rec, "image_path", None)
        if path:
            try:
                url = minio_service.get_face_image_url(path)
                urls.append(FaceImageURL(url=url))
            except Exception:
                continue
    return urls


@router.delete(
    "/face-embeddings/{embedding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_face_embedding_by_id(embedding_id: int, db: Session = Depends(get_db)):
    """Delete a specific face embedding by ID."""
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


@router.post(
    "/face-comparison",
    response_model=List[FaceMatch],
    status_code=status.HTTP_200_OK,
)
def compare_face(
    image: UploadFile = File(..., media_type="image/*"),
    threshold: float = 0.6,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    import time

    start_time = time.time()

    try:
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Threshold must be between 0.0 and 1.0",
            )

        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Limit must be between 1 and 100",
            )

        matches = face_comparison_service.compare_face(
            image=image, db=db, threshold=threshold, limit=limit
        )
        processing_time = (time.time() - start_time) * 1000
        log.info(
            f"Face comparison completed: {len(matches)} matches found in {processing_time:.2f}ms"
        )
        return matches

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e, "face comparison endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face comparison failed",
        )
