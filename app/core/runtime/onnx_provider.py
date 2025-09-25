from typing import Any, Dict, List

import numpy as np
import onnxruntime

from app.core.runtime.base import RuntimeProvider


class ONNXProvider(RuntimeProvider):
    """ONNX Runtime provider for model inference."""

    def __init__(self, model_file: str, providers: List[str] = None):
        """Initialize ONNX provider.

        Args:
            model_file: Path to the ONNX model file
            providers: List of execution providers (default: CUDA, CPU)
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model_file = model_file
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)

    def forward(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on the ONNX model.

        Args:
            input_data: Input tensor with shape (batch_size, channels, height, width)

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        input_name = self.get_input_info()["name"]
        output_names = [output["name"] for output in self.get_output_info()]

        output = self.session.run(output_names, {input_name: input_data})

        # Convert list of outputs to dictionary
        output_data = {name: output[i] for i, name in enumerate(output_names)}

        return output_data

    def get_input_info(self) -> Dict[str, Any]:
        """Get input tensor information.

        Returns:
            Dictionary containing input name, shape, and other metadata
        """
        input_cfg = self.session.get_inputs()[0]
        return {
            "name": input_cfg.name,
            "shape": input_cfg.shape,
            "type": input_cfg.type,
        }

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information.

        Returns:
            List of dictionaries containing output names, shapes, and other metadata
        """
        outputs = self.session.get_outputs()
        return [
            {
                "name": output.name,
                "shape": output.shape,
                "type": output.type,
            }
            for output in outputs
        ]
