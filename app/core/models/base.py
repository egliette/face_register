from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from app.config.settings import settings


class BaseModel(ABC):
    """Abstract base class for all face analysis models."""

    def __init__(
        self,
        provider_type: str | None = None,
        model_file: str | None = None,
        server_url: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        """Initialize the base model.

        Args:
            provider_type: Type of runtime provider ('onnx' or 'triton')
            model_file: Path to model file (for ONNX)
            server_url: Triton server URL (for Triton)
            model_name: Model name on Triton server (for Triton)
            **kwargs: Additional provider-specific arguments
        """
        if provider_type is None:
            provider_type = settings.MODEL_RUNTIME_TYPE

        if server_url is None:
            server_url = settings.TRITON_SERVER_URL

        from app.core.runtime.factory import RuntimeProviderFactory

        self.runtime_provider = RuntimeProviderFactory.create_provider(
            provider_type=provider_type,
            model_file=model_file,
            server_url=server_url,
            model_name=model_name,
            **kwargs,
        )

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """Preprocess input data. Implementation depends on model type."""

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        """Postprocess model outputs. Implementation depends on model type."""

    def forward(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference using the runtime provider.

        Args:
            input_data: Preprocessed input tensor

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        return self.runtime_provider.forward(input_data)
