from fastapi import APIRouter

from app.api.routers.face_comparison import router as face_comparison_router
from app.api.routers.face_embeddings import router as face_embeddings_router
from app.api.routers.users import router as users_router

router = APIRouter()

router.include_router(users_router)
router.include_router(face_embeddings_router)
router.include_router(face_comparison_router)
