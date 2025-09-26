from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/face_register"

    API_PREFIX: str = "/api"
    PROJECT_NAME: str = "Face Register API"

    FRONTEND_URL: str = "http://localhost:3000"
    ADMIN_PANEL_URL: str = "http://localhost:8080"

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False

    SCRFD_BUCKET: str = "models"
    SCRFD_MODEL_OBJECT: str = "scrfd/scrfd_640x640_kps.onnx"
    ARCFACE_BUCKET: str = "models"
    ARCFACE_MODEL_OBJECT: str = "arcface/arcface_r100_glint360k.onnx"
    FACE_IMAGES_BUCKET: str = "face-images"

    # Model runtime configuration
    MODEL_RUNTIME_TYPE: str = "onnx"
    TRITON_SERVER_URL: str = "triton:8000"

    # Qdrant vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "face_embeddings"
    QDRANT_VECTOR_SIZE: int = 512

    # Logging settings
    LOG_DIR: str = "logs"
    LOG_TO_STDOUT: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_MAX_DAYS: int = 30
    SERVICE_NAME: str = "face_register"
    API_TOKEN: str | None = None

    @property
    def BACKEND_CORS_ORIGINS(self) -> List[str]:
        """Build CORS origins list from individual URLs"""
        return [self.FRONTEND_URL, self.ADMIN_PANEL_URL]

    model_config = {"env_file": ".env"}


settings = Settings()
