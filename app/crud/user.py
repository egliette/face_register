from typing import List, Optional

from sqlalchemy.orm import Session

from app.crud.face_embedding import delete_face_embeddings_by_user
from app.models.user import User
from app.schema.user import UserCreate, UserUpdate
from app.services.qdrant import delete_embeddings_by_user


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> User:
    db_user = User(name=user.name, phone=user.phone)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user: UserUpdate) -> Optional[User]:
    db_user = get_user(db, user_id)
    if not db_user:
        return None

    update_data = user.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_user, field, value)

    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int) -> bool:
    db_user = get_user(db, user_id)
    if not db_user:
        return False

    # Delete all face embeddings for this user from database
    delete_face_embeddings_by_user(db, user_id)

    # Delete all face embeddings for this user from Qdrant
    delete_embeddings_by_user(user_id)

    # Delete the user
    db.delete(db_user)
    db.commit()
    return True
