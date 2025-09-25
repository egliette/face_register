from typing import Dict, List, Union

import cv2
import numpy as np
import onnx

from app.core.models.base import BaseModel
from app.utils.face_core import norm_crop


class ArcFace(BaseModel):
    """ArcFace face recognition model with runtime abstraction."""

    def __init__(
        self,
        provider_type: str = None,
        model_file: str = None,
        server_url: str = None,
        model_name: str = "arcface",
        **kwargs,
    ):
        """Initialize ArcFace model.

        Args:
            provider_type: Type of runtime provider ('onnx' or 'triton')
            model_file: Path to model file (for ONNX)
            server_url: Triton server URL (for Triton)
            model_name: Model name on Triton server (for Triton)
            **kwargs: Additional provider-specific arguments
        """
        super().__init__(
            provider_type=provider_type,
            model_file=model_file,
            server_url=server_url,
            model_name=model_name,
            **kwargs,
        )

        # Set model-specific parameters
        self.input_size = (112, 112)

        # Get input/output info from runtime provider
        input_info = self.runtime_provider.get_input_info()
        self.input_name = input_info["name"]
        self.input_shape = input_info["shape"]
        self.input_size = tuple(self.input_shape[2:4][::-1])

        output_info = self.runtime_provider.get_output_info()
        self.output_names = [output["name"] for output in output_info]
        self.output_shape = output_info[0]["shape"]
        self.embedding_dim = (
            self.output_shape[1] if len(self.output_shape) > 1 else self.output_shape[0]
        )

        # Detect model type to set correct normalization parameters
        # This is ONNX-specific logic, but we need it for preprocessing
        self._detect_normalization_params()

    def _detect_normalization_params(self):
        """Detect normalization parameters based on model type."""
        # For ONNX models, we can detect the normalization type
        # For Triton models, we'll use default values
        if hasattr(self.runtime_provider, "model_file"):
            try:
                model = onnx.load(self.runtime_provider.model_file)
                graph = model.graph
                find_sub = False
                find_mul = False
                for nid, node in enumerate(graph.node[:8]):
                    if node.name.startswith("Sub") or node.name.startswith("_minus"):
                        find_sub = True
                    if node.name.startswith("Mul") or node.name.startswith("_mul"):
                        find_mul = True
                if find_sub and find_mul:
                    # mxnet arcface model
                    self.input_mean = 0.0
                    self.input_std = 1.0
                else:
                    self.input_mean = 127.5
                    self.input_std = 127.5
            except Exception:
                # Fallback to default values
                self.input_mean = 127.5
                self.input_std = 127.5
        else:
            # For Triton models, use default values
            self.input_mean = 127.5
            self.input_std = 127.5

    def preprocess(
        self, images: List[np.ndarray], landmarks_list: List[np.ndarray]
    ) -> np.ndarray:
        """Preprocess a batch of images and landmarks for face recognition.

        Args:
            images: List of input images, each of shape (H, W, 3)
            landmarks_list: List of facial landmarks, each of shape (5, 2)

        Returns:
            Preprocessed image blob of shape (B, 3, H, W)
        """
        blobs = []
        for img, landmarks in zip(images, landmarks_list):
            # Align face using landmarks
            aimg = norm_crop(img, landmark=landmarks, image_size=self.input_size[0])

            # Convert to blob
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / self.input_std,
                self.input_size,
                (self.input_mean,) * 3,
                swapRB=True,
            )
            blobs.append(blob)

        # Concatenate along batch dimension
        batched_blob = np.concatenate(blobs, axis=0)
        return batched_blob

    def postprocess(self, net_out: Dict[str, np.ndarray]) -> np.ndarray:
        """Postprocess model output to get face embedding(s).

        Args:
            net_out: Dictionary mapping output names to numpy arrays

        Returns:
            Face embedding vector(s) of shape (B, embedding_dim)
        """
        # Convert Dict to List and get the first (and typically only) output
        output_list = [net_out[name] for name in self.output_names]
        embeddings = output_list[0]

        # Always return the batch dimension for consistency
        return embeddings

    def detect(
        self,
        img: Union[np.ndarray, List[np.ndarray]],
        landmarks: Union[np.ndarray, List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Detect face embedding from image(s) and landmarks.

        Args:
            img: Input image of shape (H, W, 3) or list of images
            landmarks: Facial landmarks of shape (5, 2) or list of landmarks

        Returns:
            List of face embedding vectors, each of shape (embedding_dim,)
        """
        # Type check
        if isinstance(img, np.ndarray):
            img = [img]
            landmarks = [landmarks]

        if not isinstance(img, (list, tuple)) or not isinstance(
            landmarks, (list, tuple)
        ):
            raise TypeError(
                "img must be a numpy.ndarray or a list/tuple of numpy.ndarray images"
            )

        if len(img) != len(landmarks):
            raise ValueError("Number of images must match number of landmarks")

        if len(img) == 0:
            return []

        preprocessed_input = self.preprocess(img, landmarks)
        model_output = self.forward(preprocessed_input)
        embeddings = self.postprocess(model_output)

        # Convert to list of individual embeddings
        result = []
        for i in range(embeddings.shape[0]):
            result.append(embeddings[i])

        return result
