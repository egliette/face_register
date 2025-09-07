from pydantic import BaseModel


class FaceImageURL(BaseModel):
    url: str
