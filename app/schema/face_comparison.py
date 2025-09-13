from typing import Optional

from pydantic import BaseModel, Field

from app.schema.user import UserPublic


class FaceComparisonRequest(BaseModel):
    """Request model for face comparison."""

    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 to 1.0)",
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )


class FaceMatch(BaseModel):
    """Model representing a face match result."""

    user: UserPublic = Field(description="Matched user")
    face_embedding_id: int = Field(description="ID of the face embedding")
    image_path: Optional[str] = Field(description="Path to the face image")
    similarity_score: float = Field(description="Similarity score (0.0 to 1.0)")
    created_at: Optional[str] = Field(description="When the face was enrolled")
