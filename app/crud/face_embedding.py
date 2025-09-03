from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.face_embedding import FaceEmbedding
from app.schema.face_embedding import FaceEmbeddingCreate


def create_face_embedding(db: Session, data: FaceEmbeddingCreate) -> FaceEmbedding:
    record = FaceEmbedding(user_id=data.user_id)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_face_embeddings_by_user(db: Session, user_id: int) -> List[FaceEmbedding]:
    return db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).all()


def get_face_embedding(db: Session, embedding_id: int) -> Optional[FaceEmbedding]:
    return db.query(FaceEmbedding).filter(FaceEmbedding.id == embedding_id).first()


def delete_face_embedding(db: Session, embedding_id: int) -> bool:
    rec = get_face_embedding(db, embedding_id)
    if not rec:
        return False
    db.delete(rec)
    db.commit()
    return True


def delete_face_embeddings_by_user(db: Session, user_id: int) -> bool:
    """Delete all face embeddings for a specific user."""
    embeddings = get_face_embeddings_by_user(db, user_id)
    if not embeddings:
        return True

    for embedding in embeddings:
        db.delete(embedding)
    db.commit()
    return True
