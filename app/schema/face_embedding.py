from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class FaceEmbeddingBase(BaseModel):
    user_id: int


class FaceEmbeddingCreate(BaseModel):
    user_id: int
    embedding: List[float] = Field(min_length=16)


class FaceEmbedding(FaceEmbeddingBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
