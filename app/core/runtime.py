from functools import lru_cache

from app.config.settings import settings
from app.core.arcface import ArcFace
from app.core.scrfd import SCRFD


@lru_cache(maxsize=1)
def get_scrfd() -> SCRFD:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    local_model_path = (
        repo_root
        / "assets"
        / "models"
        / "scrfd"
        / Path(settings.SCRFD_MODEL_OBJECT).name
    )
    model_file = str(local_model_path)

    return SCRFD(model_file=model_file)


@lru_cache(maxsize=1)
def get_arcface() -> ArcFace:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    local_model_path = (
        repo_root
        / "assets"
        / "models"
        / "arcface"
        / Path(settings.ARCFACE_MODEL_OBJECT).name
    )

    if local_model_path.exists():
        model_file = str(local_model_path)
    else:
        model_file = f"{settings.ARCFACE_BUCKET}/{settings.ARCFACE_MODEL_OBJECT}"

    return ArcFace(model_file=model_file)
