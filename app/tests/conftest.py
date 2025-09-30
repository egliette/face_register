from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.core.model_factory import get_arcface, get_scrfd
from app.database.connection import SessionLocal
from app.main import app
from app.models.face_embedding import FaceEmbedding
from app.models.user import User


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    token = settings.API_TOKEN
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    app.dependency_overrides = {}
    with TestClient(app, headers=headers) as c:
        yield c


@pytest.fixture(scope="function")
def db_session() -> Session:
    """Provide a database session for each test"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def clean_db(db_session: Session):
    """Clean the database before each test"""
    db_session.query(FaceEmbedding).delete()
    db_session.query(User).delete()
    db_session.commit()

    yield db_session

    db_session.query(FaceEmbedding).delete()
    db_session.query(User).delete()
    db_session.commit()


@pytest.fixture(scope="function")
def test_user(db_session: Session):
    """Create a test user and clean up after"""
    user = User(name="Test User", phone="1234567890")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    yield user

    db_session.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).delete()
    db_session.query(User).filter(User.id == user.id).delete()
    db_session.commit()


@pytest.fixture(scope="session")
def arcface_model():
    return get_arcface()


@pytest.fixture(scope="session")
def scrfd_model():
    return get_scrfd()
