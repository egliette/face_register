from datetime import timedelta
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config.settings import settings
from app.utils.logger import log


class MinIOService:
    """Service for handling MinIO operations"""

    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.face_images_bucket = settings.FACE_IMAGES_BUCKET
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the face images bucket exists"""
        try:
            if not self.client.bucket_exists(self.face_images_bucket):
                self.client.make_bucket(self.face_images_bucket)
                log.info(f"Created MinIO bucket: {self.face_images_bucket}")
            else:
                log.bug(f"MinIO bucket already exists: {self.face_images_bucket}")
        except S3Error as e:
            log.error(f"Failed to create MinIO bucket {self.face_images_bucket}: {e}")
            raise

    def upload_face_image(
        self,
        user_id: int,
        face_embedding_id: int,
        image_data: bytes,
        content_type: str,
        original_filename: Optional[str] = None,
    ) -> str:
        """
        Upload face image to MinIO and return the object name

        Args:
            user_id: ID of the user
            face_embedding_id: ID of the face embedding record
            image_data: Raw image data
            content_type: MIME type of the image
            original_filename: Original filename (optional)

        Returns:
            str: Object name/path in MinIO
        """
        try:
            # Generate object name: face-images/user_{user_id}/embedding_{embedding_id}/image.{ext}
            file_extension = self._get_extension_from_content_type(content_type)
            object_name = (
                f"user_{user_id}/embedding_{face_embedding_id}/image{file_extension}"
            )

            # Upload the image
            image_stream = BytesIO(image_data)
            self.client.put_object(
                bucket_name=self.face_images_bucket,
                object_name=object_name,
                data=image_stream,
                length=len(image_data),
                content_type=content_type,
            )

            log.info(
                f"Successfully uploaded face image for user {user_id}, embedding {face_embedding_id}"
            )
            return object_name

        except S3Error as e:
            log.error(f"Failed to upload face image for user {user_id}: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error uploading face image for user {user_id}: {e}")
            raise

    def get_face_image_url(self, object_name: str) -> str:
        """
        Get the URL for accessing a face image

        Args:
            object_name: Object name/path in MinIO

        Returns:
            str: URL to access the image
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.face_images_bucket,
                object_name=object_name,
                expires=timedelta(days=7),
            )
            return url
        except S3Error as e:
            log.error(f"Failed to generate presigned URL for {object_name}: {e}")
            raise

    def delete_face_image(self, object_name: str) -> bool:
        """
        Delete a face image from MinIO

        Args:
            object_name: Object name/path in MinIO

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.remove_object(self.face_images_bucket, object_name)
            log.info(f"Successfully deleted face image: {object_name}")
            return True
        except S3Error as e:
            log.error(f"Failed to delete face image {object_name}: {e}")
            return False

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get file extension from content type"""
        content_type_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        return content_type_map.get(content_type, ".jpg")


minio_service = MinIOService()
