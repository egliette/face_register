from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.crud.face_embedding import (
    delete_face_embedding,
    get_face_embedding,
    get_face_embeddings_by_user,
)
from app.database.connection import get_db
from app.schema.face_embedding import FaceEmbeddingPublic
from app.schema.media import FaceImageURL
from app.services.face_enrollment import face_enrollment_service
from app.services.minio import minio_service
from app.services.qdrant import delete_embedding

router = APIRouter()


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
