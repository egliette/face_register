from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schema.face_comparison import FaceMatch
from app.services.face_comparison import face_comparison_service
from app.utils.logger import log

router = APIRouter()


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
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Threshold must be between 0.0 and 1.0",
            )

        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
