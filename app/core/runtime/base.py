from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class RuntimeProvider(ABC):
    """Abstract base class for runtime providers (ONNX, Triton, etc.)."""

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on the model.

        Args:
            input_data: Input tensor with shape (batch_size, channels, height, width)

        Returns:
            List of output tensors
        """

    @abstractmethod
    def get_input_info(self) -> Dict[str, Any]:
        """Get input tensor information.

        Returns:
            Dictionary containing input name, shape, and other metadata
        """

    @abstractmethod
    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information.

        Returns:
            List of dictionaries containing output names, shapes, and other metadata
        """
