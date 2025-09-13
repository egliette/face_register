import time
from typing import List

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.crud.face_embedding import get_face_embedding
from app.crud.user import get_user
from app.schema.face_comparison import FaceMatch
from app.schema.user import UserPublic
from app.services.qdrant import search_similar_embeddings
from app.utils.face_service import (
    detect_face_in_image,
    extract_face_embedding,
    validate_image_upload,
)
from app.utils.logger import log


class FaceComparisonService:
    """Service for comparing faces against the database."""

    def compare_face(
        self, image: UploadFile, db: Session, threshold: float = 0.6, limit: int = 10
    ) -> List[FaceMatch]:
        """Compare a face against all faces in the database and return matches."""
        start_time = time.time()
        log.info(f"Starting face comparison, file: {image.filename}")
        try:
            image_data, img = validate_image_upload(image=image)
            face = detect_face_in_image(img)
            embedding = extract_face_embedding(img, face)
            similar_points = search_similar_embeddings(
                query_embedding=embedding.tolist(),
                limit=limit,
                score_threshold=threshold,
            )
        except HTTPException:
            raise
        except Exception as e:
            log.exception(e, "face comparison")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Face comparison failed",
            )

        results = []
        for point in similar_points:
            try:
                face_embedding = get_face_embedding(db, point.id)
                if not face_embedding:
                    continue

                user = get_user(db, face_embedding.user_id)
                if not user:
                    continue

                user_public = UserPublic(id=user.id, name=user.name, phone=user.phone)

                results.append(
                    FaceMatch(
                        user=user_public,
                        face_embedding_id=face_embedding.id,
                        image_path=face_embedding.image_path,
                        similarity_score=round(point.score, 4),
                        created_at=(
                            face_embedding.created_at.isoformat()
                            if face_embedding.created_at
                            else None
                        ),
                    )
                )

            except Exception as e:
                log.exception(e, f"processing similarity result for point {point.id}")
                continue

        duration = time.time() - start_time
        log.perf(
            "face_comparison",
            duration,
            file_size=len(image_data),
            matches_found=len(results),
        )

        log.info(f"Face comparison completed, found {len(results)} matches")
        return results


face_comparison_service = FaceComparisonService()
