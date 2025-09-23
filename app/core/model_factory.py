from functools import lru_cache

from app.config.settings import settings
from app.core.models.arcface import ArcFace
from app.core.models.scrfd import SCRFD
from app.utils.logger import log


@lru_cache(maxsize=1)
def get_scrfd() -> SCRFD:
    from pathlib import Path

    try:
        repo_root = Path(__file__).resolve().parents[2]
        local_model_path = (
            repo_root
            / "assets"
            / "models"
            / "scrfd"
            / Path(settings.SCRFD_MODEL_OBJECT).name
        )
        model_file = str(local_model_path)

        scrfd = SCRFD(model_file=model_file)
        return scrfd
    except Exception as e:
        log.exception(e, "loading SCRFD model")
        raise


@lru_cache(maxsize=1)
def get_arcface() -> ArcFace:
    from pathlib import Path

    try:
        repo_root = Path(__file__).resolve().parents[2]
        local_model_path = (
            repo_root
            / "assets"
            / "models"
            / "arcface"
            / Path(settings.ARCFACE_MODEL_OBJECT).name
        )
        model_file = str(local_model_path)

        arcface = ArcFace(model_file=model_file)
        return arcface
    except Exception as e:
        log.exception(e, "loading ArcFace model")
        raise
