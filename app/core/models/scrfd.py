from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.core.models.base import BaseModel
from app.core.runtime.triton_provider import TritonProvider
from app.utils.face_core import distance2bbox, distance2kps, nms


def extract_and_sort_batch_data(
    output_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Extracts data from batch dimension and sorts by:
    1. Second dimension descending (12800, 3200, 800)
    2. Third dimension ascending (1, 4, 10)

    Args:
        output_data: Dictionary mapping output names to numpy arrays with batch dimension

    Returns:
        Dict mapping output names (in their original order) to batched numpy
        arrays, remapped so that values follow the expected internal order
        (last_dim asc, middle_dim desc). The batch dimension is preserved.
    """
    # Get batch size from any tensor
    batch_size: int = next(iter(output_data.values())).shape[0]

    # Collect items with their shape-based sort keys
    sortable_items: List[Tuple[str, np.ndarray, int, int]] = []
    for name, data in output_data.items():
        if len(data.shape) >= 3:
            middle_dim: int = data.shape[1]
            last_dim: int = data.shape[2]
            sortable_items.append((name, data, middle_dim, last_dim))

    # Sort by last_dim asc, then middle_dim desc
    sortable_items.sort(key=lambda t: (t[3], -t[2]))

    # Remap sorted tensors back onto original output names in their original order
    original_names: List[str] = list(output_data.keys())
    processed: Dict[str, np.ndarray] = {}
    for i, name in enumerate(original_names):
        # Guard against length mismatch
        if i < len(sortable_items):
            processed[name] = sortable_items[i][1]
        else:
            processed[name] = output_data[name]

    return processed


@dataclass
class Face:
    """Face detection result.

    Attributes:
        score: Detection confidence score.
        box: Bounding box as (x1, y1, x2, y2) in original image scale.
        keypoint: Optional keypoints with shape (5, 2) in original image scale.
    """

    score: float
    box: Tuple[float, float, float, float]
    keypoint: Optional[np.ndarray] = None


class SCRFD(BaseModel):
    """SCRFD face detection model with runtime abstraction."""

    def __init__(
        self,
        nms_thresh: float = 0.4,
        det_thresh: float = 0.2,
        input_size: Tuple[int, int] = (640, 640),
        provider_type: str = None,
        model_file: str = None,
        server_url: str = None,
        model_name: str = "scrfd",
        **kwargs,
    ):
        """Initialize SCRFD model.

        Args:
            nms_thresh: Non-maximum suppression threshold
            det_thresh: Detection confidence threshold
            input_size: Input image size (width, height)
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

        self.nms_thresh = nms_thresh
        self.det_thresh = det_thresh
        self.input_size = input_size

        # Cache for anchor center coordinates to avoid recomputation across frames
        self.center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

        self.input_mean = 127.5
        self.input_std = 128.0

        input_info = self.runtime_provider.get_input_info()
        self.input_name = input_info["name"]
        self.input_shape = input_info["shape"]

        # Check if not input_shape = [1, 3, '?', '?']
        if not isinstance(self.input_shape[2], str) and not self.input_shape[2] < 0:
            # input_size = (W, H)
            self.input_size = tuple(self.input_shape[2:4][::-1])

        output_info = self.runtime_provider.get_output_info()
        self.output_names = [output["name"] for output in output_info]

        # Require batched output tensors. If the model lacks a batch dimension, raise.
        if len(output_info[0]["shape"]) != 3:
            raise ValueError(
                "SCRFD requires a batched model with outputs shaped as (B, ..., ...)."
            )

        self._configure_output_structure(output_info)

    def _configure_output_structure(self, output_info: List[Dict[str, Any]]) -> None:
        """Configure output structure based on number of outputs."""
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        config_map = {
            6: {"strides": [8, 16, 32], "anchors": 2, "kps": False},
            9: {"strides": [8, 16, 32], "anchors": 2, "kps": True},
            10: {"strides": [8, 16, 32, 64, 128], "anchors": 1, "kps": False},
            15: {"strides": [8, 16, 32, 64, 128], "anchors": 1, "kps": True},
        }

        cfg = config_map.get(len(output_info))
        if cfg is None:
            raise ValueError(f"Unsupported output count: {len(output_info)}")

        self._feat_stride_fpn = cfg["strides"]
        self._num_anchors = cfg["anchors"]
        self.use_kps = cfg["kps"]

    def forward(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference using the runtime provider.

        For Triton providers, applies batch data extraction and sorting.
        For other providers (like ONNX), returns raw output.

        Args:
            input_data: Preprocessed input tensor with shape (batch_size, channels, height, width)

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        raw_output: Dict[str, np.ndarray] = self.runtime_provider.forward(input_data)

        if isinstance(self.runtime_provider, TritonProvider):
            return extract_and_sort_batch_data(raw_output)

        return raw_output

    def _preprocess_single(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess input image for SCRFD model.

        Args:
            img: Input image as numpy array

        Returns:
            Tuple of (preprocessed_tensor, scale_ratio)
        """
        h, w = img.shape[:2]
        target_w, target_h = self.input_size

        im_ratio = h / w
        model_ratio = target_h / target_w

        if im_ratio > model_ratio:
            new_h = target_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = target_w
            new_h = int(new_w * im_ratio)

        # Resize the image to fit inside the target size while preserving aspect ratio
        # shape: (new_h, new_w, 3)
        resized = cv2.resize(img, (new_w, new_h))

        # Create a black canvas and place the resized image in the top-left corner
        # shape: (target_h, target_w, 3)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        scale_ratio = new_h / h
        input_size = (target_w, target_h)

        # Convert to blob format
        # shape: (1, 3, target_h, target_w)
        blob = cv2.dnn.blobFromImage(
            padded,
            1.0 / self.input_std,
            input_size,
            (self.input_mean,) * 3,
            swapRB=True,
        )

        return blob, scale_ratio

    def _process_all_feature_maps(
        self, output: List[np.ndarray], input_height: int, input_width: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Process all feature maps from model output."""
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            fmc = len(self._feat_stride_fpn)

            # outputs passed here are per-sample (no batch dimension)
            scores = output[idx]
            bbox_preds = output[idx + fmc] * stride
            kps_preds = output[idx + fmc * 2] * stride if self.use_kps else None

            height = input_height // stride
            width = input_width // stride

            # Get anchor centers
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            # anchor_centers shape: (height * width * self._num_anchors, 2)

            # Process detections
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            # bboxes shape: (height * width * self._num_anchors, 4)
            keep_indices = np.where(scores >= self.det_thresh)[0]
            scores = scores[keep_indices]
            bboxes = bboxes[keep_indices]
            # scores shape: (num_kept, 1)
            # bboxes shape: (num_kept, 4)

            scores_list.append(scores)
            bboxes_list.append(bboxes)

            # Process keypoints if available
            if self.use_kps:
                keypoints = distance2kps(anchor_centers, kps_preds)
                keypoints = keypoints.reshape((keypoints.shape[0], -1, 2))
                # keypoints shape: (height * width * self._num_anchors, 5, 2)
                keypoints = keypoints[keep_indices]
                # keypoints shape: (num_kept, 5, 2)
                kpss_list.append(keypoints)

        return scores_list, bboxes_list, kpss_list

    def _merge_detections(
        self,
        scores_list: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        keypoints_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Merge detections from all feature maps."""
        # Stack all feature maps
        scores = np.vstack(scores_list)
        # scores shape: (total_detections, 1)
        scores_ravel = scores.ravel()
        # scores_ravel shape: (total_detections,)
        order = scores_ravel.argsort()[::-1]
        boxes = np.vstack(bboxes_list)
        # boxes shape: (total_detections, 4)

        # Sort by scores
        boxes = boxes[order]
        scores = scores_ravel[order]

        # Process keypoints if available
        keypoints = None
        if self.use_kps:
            keypoints = np.vstack(keypoints_list)
            # keypoints shape: (total_detections, 5, 2)
            keypoints = keypoints[order, :, :]

        return boxes, scores, keypoints, order

    def _apply_nms(
        self, boxes: np.ndarray, scores: np.ndarray, keypoints: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply non-maximum suppression to detections."""
        # Combine boxes and scores for NMS
        det = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # det shape: (total_detections, 5)

        # Apply NMS
        keep = nms(det, self.nms_thresh)

        # Filter results
        boxes = boxes[keep]
        scores = scores[keep]
        if keypoints is not None:
            keypoints = keypoints[keep, :, :]

        return boxes, scores, keypoints

    def _scale_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        keypoints: Optional[np.ndarray],
        scale_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Scale detections back to original image size."""
        boxes = boxes / scale_ratio
        # boxes shape: (num_kept, 4)

        if keypoints is not None:
            keypoints = keypoints / scale_ratio
            # keypoints shape: (num_kept, 5, 2)

        return boxes, scores, keypoints

    def _filter_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        keypoints: Optional[np.ndarray],
        max_num: int,
        metric: str,
        input_height: int,
        input_width: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Filter detections based on max_num and metric."""
        if max_num <= 0 or boxes.shape[0] <= max_num:
            return boxes, scores, keypoints

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        img_center = (input_height // 2, input_width // 2)
        offsets = np.vstack(
            [
                (boxes[:, 0] + boxes[:, 2]) / 2 - img_center[1],
                (boxes[:, 1] + boxes[:, 3]) / 2 - img_center[0],
            ]
        )
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

        # Choose the sorting metric for picking top faces
        if metric == "max":
            values = area
        else:
            values = area - offset_dist_squared * 2.0

        bindex = np.argsort(values)[::-1]
        bindex = bindex[0:max_num]

        filtered_boxes = boxes[bindex]
        filtered_scores = scores[bindex]
        filtered_keypoints = keypoints[bindex] if keypoints is not None else None

        return filtered_boxes, filtered_scores, filtered_keypoints

    def postprocess_single(
        self,
        input_tensor: np.ndarray,
        output: Dict[str, np.ndarray],
        scale_ratio: float = 1.0,
        max_num: int = 0,
        metric: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Post-process model outputs to get final detections for a single sample."""
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]

        # Convert dictionary to list in the correct order
        output_list = [output[name] for name in self.output_names]

        scores_list, bboxes_list, keypoints_list = self._process_all_feature_maps(
            output_list, input_height, input_width
        )
        boxes, scores, keypoints, _ = self._merge_detections(
            scores_list, bboxes_list, keypoints_list
        )
        boxes, scores, keypoints = self._apply_nms(boxes, scores, keypoints)
        boxes, scores, keypoints = self._scale_detections(
            boxes, scores, keypoints, scale_ratio
        )
        boxes, scores, keypoints = self._filter_detections(
            boxes, scores, keypoints, max_num, metric, input_height, input_width
        )

        return boxes, scores, keypoints

    def _detect_with_outputs(
        self,
        model_input: np.ndarray,
        outputs: Dict[str, np.ndarray],
        scale_ratio: float,
        max_num: int,
        metric: str,
    ) -> List[Face]:
        """Run postprocess and convert results to faces for a single sample."""
        boxes, scores, keypoints = self.postprocess_single(
            model_input, outputs, scale_ratio, max_num, metric
        )
        faces: List[Face] = []
        num_detections = boxes.shape[0]
        for i in range(num_detections):
            kp = keypoints[i] if keypoints is not None else None
            face = Face(
                score=float(scores[i]),
                box=(
                    float(boxes[i, 0]),
                    float(boxes[i, 1]),
                    float(boxes[i, 2]),
                    float(boxes[i, 3]),
                ),
                keypoint=kp,
            )
            faces.append(face)
        return faces

    def preprocess(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
        """Preprocess a batch of images.

        Returns:
            Tuple of (batched_input, scale_ratios) where batched_input is the
            concatenated blob tensor and scale_ratios contains one ratio per image.
        """
        blobs: List[np.ndarray] = []
        scale_ratios: List[float] = []
        for im in images:
            blob, ratio = self._preprocess_single(im)
            blobs.append(blob)
            scale_ratios.append(ratio)
        return np.concatenate(blobs, axis=0), scale_ratios

    def postprocess(
        self,
        batched_output_dict: Dict[str, np.ndarray],
        scale_ratios: List[float],
        max_num: int,
        metric: str,
    ) -> List[List[Face]]:
        """Postprocess batched outputs to get face detections.

        Args:
            batched_output_dict: Dict of output name to batched numpy arrays
            scale_ratios: Precomputed scale ratios from preprocessing, one per image
            max_num: Max detections to keep per image
            metric: Filtering metric
        """
        results: List[List[Face]] = []
        batch_size = batched_output_dict[list(batched_output_dict.keys())[0]].shape[0]

        for b in range(batch_size):
            per_sample_output = {
                name: output[b] for name, output in batched_output_dict.items()
            }
            per_sample_input = np.zeros(
                (1, 3, self.input_size[1], self.input_size[0])
            )  # Dummy input for shape
            faces_b = self._detect_with_outputs(
                per_sample_input,
                per_sample_output,
                scale_ratios[b],
                max_num,
                metric,
            )
            results.append(faces_b)

        return results

    def detect(
        self,
        img: Union[np.ndarray, List[np.ndarray]],
        max_num: int = -1,
        metric: str = "default",
    ) -> List[List[Face]]:
        """Detect faces in one or more images.

        Args:
            img: A single image `numpy.ndarray` (H, W, 3) or a list/tuple of images
            max_num: Maximum number of detections to keep per image
            metric: Filtering metric ('max' or 'default')

        Returns:
            List of face detection results, each containing a list of `Face` objects
        """
        # Type check
        if isinstance(img, np.ndarray):
            images = [img]
        elif isinstance(img, (list, tuple)):
            images = list(img)
        else:
            raise TypeError(
                "img must be a numpy.ndarray or a list/tuple of numpy.ndarray images"
            )

        if len(images) == 0:
            return []

        batched_input, scale_ratios = self.preprocess(images)
        batched_output = self.forward(batched_input)
        face_detections = self.postprocess(
            batched_output, scale_ratios, max_num, metric
        )

        return face_detections

    def draw(
        self, img: np.ndarray, faces: List[Face], labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Draw detection results on the image.

        Args:
            img: Input image to draw on, shape (H, W, 3)
            faces: Face detections to draw
            labels: List of labels to show for each detection

        Returns:
            Image with visualizations
        """
        # Make a copy of the image to avoid modifying the original
        vis_img = img.copy()

        # Draw each detection
        for i, face in enumerate(faces):
            # Get detection box and score
            x1, y1, x2, y2 = map(int, face.box)
            score = face.score

            # Draw bounding box with thicker line
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw label if provided
            if labels is not None and i < len(labels):
                label = f"{labels[i]}: {score:.2f}"
            else:
                label = f"{score:.2f}"

            # Get text size and draw background rectangle
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(
                vis_img,
                (x1, y1 - text_height - 4),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1,
            )

            # Draw text in white color
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

            # Draw keypoints if available with larger circles
            if face.keypoint is not None:
                kps = face.keypoint.astype(int)
                for kp in kps:
                    cv2.circle(vis_img, tuple(kp), 4, (0, 0, 255), -1)

        return vis_img
