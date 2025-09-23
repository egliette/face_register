"""Factory classes for creating runtime providers."""

from typing import Optional

from app.core.runtime.base import RuntimeProvider
from app.core.runtime.onnx_provider import ONNXProvider
from app.core.runtime.triton_provider import TritonProvider


class RuntimeProviderFactory:
    """Factory for creating runtime providers."""

    @staticmethod
    def create_provider(
        provider_type: str,
        model_file: Optional[str] = None,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> RuntimeProvider:
        """Create a runtime provider instance.

        Args:
            provider_type: Type of provider ('onnx' or 'triton')
            model_file: Path to model file (for ONNX)
            server_url: Triton server URL (for Triton)
            model_name: Model name on Triton server (for Triton)
            **kwargs: Additional provider-specific arguments

        Returns:
            Runtime provider instance

        Raises:
            ValueError: If provider_type is not supported
        """
        if provider_type.lower() == "onnx":
            if model_file is None:
                raise ValueError("model_file is required for ONNX provider")
            return ONNXProvider(model_file, **kwargs)
        elif provider_type.lower() == "triton":
            if server_url is None or model_name is None:
                raise ValueError(
                    "server_url and model_name are required for Triton provider"
                )
            return TritonProvider(server_url, model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
