from typing import List

import cv2
import numpy as np
import onnx
import onnxruntime

from app.utils.face_core import norm_crop


class ArcFace:
    def __init__(self, model_file: str) -> None:
        """Initialize ArcFace face recognition model.

        Args:
            model_file (str): Path to the ONNX model file
        """
        self.model_file = model_file

        self.session = onnxruntime.InferenceSession(
            self.model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Detect model type to set correct normalization parameters
        # MXNet models use mean=0.0, std=1.0, others use mean=127.5, std=127.5
        model = onnx.load(self.model_file)
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

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_shape = input_cfg.shape
        self.input_size = tuple(self.input_shape[2:4][::-1])

        outputs = self.session.get_outputs()
        self.output_names: List[str] = []
        for out in outputs:
            self.output_names.append(out.name)
        self.output_shape = outputs[0].shape

    def preprocess(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Preprocess image and landmarks for face recognition.

        Args:
            img (numpy.ndarray): Input image of shape (H, W, 3)
            landmarks (numpy.ndarray): Facial landmarks of shape (5, 2)

        Returns:
            numpy.ndarray: Preprocessed image blob of shape (1, 3, H, W)
        """
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
        return blob

    def forward(self, blob: np.ndarray) -> np.ndarray:
        """Run forward pass through the model.

        Args:
            blob (numpy.ndarray): Preprocessed image blob of shape (1, 3, H, W)

        Returns:
            numpy.ndarray: Model output of shape (1, embedding_dim)
        """
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def postprocess(self, net_out: np.ndarray) -> np.ndarray:
        """Postprocess model output to get face embedding.

        Args:
            net_out (numpy.ndarray): Model output of shape (1, embedding_dim)

        Returns:
            numpy.ndarray: Face embedding vector of shape (embedding_dim,)
        """
        embedding = net_out.flatten()
        return embedding

    def detect(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Detect face embedding from image and landmarks.

        Args:
            img (numpy.ndarray): Input image of shape (H, W, 3)
            landmarks (numpy.ndarray): Facial landmarks of shape (5, 2)

        Returns:
            numpy.ndarray: Face embedding vector of shape (embedding_dim,)
        """
        blob = self.preprocess(img, landmarks)
        net_out = self.forward(blob)
        embedding = self.postprocess(net_out)
        return embedding
