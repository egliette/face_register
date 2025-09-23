from typing import Any, Dict, List

import numpy as np
import tritonclient.http as httpclient

from app.core.runtime.base import RuntimeProvider


class TritonProvider(RuntimeProvider):
    """Triton Inference Server provider for model inference."""

    def __init__(
        self,
        server_url: str = "localhost:8000",
        model_name: str = "scrfd",
    ):
        """Initialize Triton provider.

        Args:
            server_url: Triton server URL
            model_name: Name of the model on Triton server
        """
        self.server_url = server_url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=server_url)
        self.metadata = self.client.get_model_metadata(model_name=model_name)

    def forward(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on the Triton model.

        Args:
            input_data: Input tensor with shape (batch_size, channels, height, width)

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        input_name = self.get_input_info()["name"]
        output_names = [output["name"] for output in self.get_output_info()]

        inputs = httpclient.InferInput(input_name, input_data.shape, "FP32")
        inputs.set_data_from_numpy(input_data)
        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

        response = self.client.infer(
            model_name=self.model_name, inputs=[inputs], outputs=outputs
        )

        output_data = {name: response.as_numpy(name) for name in output_names}

        return output_data

    def get_input_info(self) -> Dict[str, Any]:
        """Get input tensor information.

        Returns:
            Dictionary containing input name, shape, and other metadata
        """
        input_info = self.metadata["inputs"][0]
        return {
            "name": input_info["name"],
            "shape": input_info["shape"],
            "type": input_info.get("data_type", "FP32"),
        }

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information.

        Returns:
            List of dictionaries containing output names, shapes, and other metadata
        """
        return [
            {
                "name": output["name"],
                "shape": output["shape"],
                "type": output.get("data_type", "FP32"),
            }
            for output in self.metadata["outputs"]
        ]
