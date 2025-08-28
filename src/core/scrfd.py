from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime

from src.utils.face_utils import distance2bbox, distance2kps, nms, softmax


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


class SCRFD:
    def __init__(
        self,
        model_file: str,
        nms_thresh: float = 0.4,
        det_thresh: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.model_file = model_file
        self.session = onnxruntime.InferenceSession(
            self.model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.nms_thresh = nms_thresh
        self.det_thresh = det_thresh
        self.batched = False

        # Cache for anchor center coordinates to avoid recomputation across frames
        self.center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

        self.input_mean = 127.5
        self.input_std = 128.0

        # default value if model use dynamic size input
        self.input_size = input_size

        input_cfg = self.session.get_inputs()[0]
        self.input_shape = input_cfg.shape
        self.input_name = input_cfg.name

        # check if not input_shape = [1, 3, '?', '?']
        if not isinstance(self.input_shape[2], str):
            # input_size = (W, H)
            self.input_size = tuple(self.input_shape[2:4][::-1])

        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        self.output_names: List[str] = []
        for o in outputs:
            self.output_names.append(o.name)

        self._configure_output_structure(outputs)

    def _configure_output_structure(self, outputs: List[Any]) -> None:
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        config_map = {
            6: {"strides": [8, 16, 32], "anchors": 2, "kps": False},
            9: {"strides": [8, 16, 32], "anchors": 2, "kps": True},
            10: {"strides": [8, 16, 32, 64, 128], "anchors": 1, "kps": False},
            15: {"strides": [8, 16, 32, 64, 128], "anchors": 1, "kps": True},
        }

        cfg = config_map.get(len(outputs))
        if cfg is None:
            raise ValueError(f"Unsupported output count: {len(outputs)}")

        self._feat_stride_fpn = cfg["strides"]
        self._num_anchors = cfg["anchors"]
        self.use_kps = cfg["kps"]

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
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

        # shape: (1, 3, target_h, target_w)
        blob = cv2.dnn.blobFromImage(
            padded,
            1.0 / self.input_std,
            input_size,
            (self.input_mean,) * 3,
            swapRB=True,
        )

        return blob, scale_ratio

    def forward(self, input: np.ndarray) -> List[np.ndarray]:
        output = self.session.run(self.output_names, {self.input_name: input})
        return output

    def _process_all_feature_maps(
        self, output: List[np.ndarray], input_height: int, input_width: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Process all feature maps from model output.

        Args:
            output (list): Model output tensors. Length equals number of feature maps * (2 or 3)
                          depending on whether keypoints are used.
            input_height (int): Height of input image
            input_width (int): Width of input image

        Returns:
            tuple: A tuple containing:
                - scores_list (list): List of score tensors, each with shape (num_kept, 1)
                - bboxes_list (list): List of bbox tensors, each with shape (num_kept, 4)
                - kpss_list (list): List of keypoint tensors, each with shape (num_kept, 5, 2)
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            get = (lambda x: x[0]) if self.batched else (lambda x: x)
            fmc = len(self._feat_stride_fpn)

            scores = get(output[idx])
            bbox_preds = get(output[idx + fmc]) * stride
            kps_preds = get(output[idx + fmc * 2]) * stride if self.use_kps else None

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
        """Merge detections from all feature maps.

        Args:
            scores_list (list): List of score tensors, each with shape (num_kept, 1)
            bboxes_list (list): List of bbox tensors, each with shape (num_kept, 4)
            keypoints_list (list): List of keypoint tensors, each with shape (num_kept, 5, 2)

        Returns:
            tuple: A tuple containing:
                - boxes (numpy.ndarray): Detection boxes, shape (total_detections, 4)
                - scores (numpy.ndarray): Detection scores, shape (total_detections,)
                - keypoints (numpy.ndarray): Keypoint results, shape (total_detections, 5, 2)
                - order (numpy.ndarray): Sort order indices, shape (total_detections,)
        """
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
        """Apply non-maximum suppression to detections.

        Args:
            boxes (numpy.ndarray): Detection boxes, shape (total_detections, 4)
            scores (numpy.ndarray): Detection scores, shape (total_detections,)
            keypoints (numpy.ndarray): Keypoint results, shape (total_detections, 5, 2)

        Returns:
            tuple: A tuple containing:
                - boxes (numpy.ndarray): Filtered boxes, shape (num_kept, 4)
                - scores (numpy.ndarray): Filtered scores, shape (num_kept,)
                - keypoints (numpy.ndarray): Filtered keypoints, shape (num_kept, 5, 2)
        """
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
        """Scale detections back to original image size.

        Args:
            boxes (numpy.ndarray): Detection boxes, shape (num_kept, 4)
            scores (numpy.ndarray): Detection scores, shape (num_kept,)
            keypoints (numpy.ndarray): Keypoint results, shape (num_kept, 5, 2)
            scale_ratio (float): Scale ratio for converting back to original image size

        Returns:
            tuple: A tuple containing:
                - boxes (numpy.ndarray): Scaled boxes, shape (num_kept, 4)
                - scores (numpy.ndarray): Unchanged scores, shape (num_kept,)
                - keypoints (numpy.ndarray): Scaled keypoints, shape (num_kept, 5, 2)
        """
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
        """Filter detections based on max_num and metric.

        Args:
            boxes (numpy.ndarray): Detection boxes, shape (num_kept, 4)
            scores (numpy.ndarray): Detection scores, shape (num_kept,)
            keypoints (numpy.ndarray): Keypoint results, shape (num_kept, 5, 2)
            max_num (int): Maximum number of detections to keep
            metric (str): Filtering metric ('max' or 'default')
            input_height (int): Height of input image
            input_width (int): Width of input image

        Returns:
            tuple: A tuple containing:
                - boxes (numpy.ndarray): Filtered boxes, shape (min(num_kept, max_num), 4)
                - scores (numpy.ndarray): Filtered scores, shape (min(num_kept, max_num),)
                - keypoints (numpy.ndarray): Filtered keypoints, shape (min(num_kept, max_num), 5, 2)
        """
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

        # Choose the sorting metric for picking top faces:
        # - metric == "max": use pure bbox area (prefers larger faces)
        # - otherwise: area - distance_penalty (prefers centered, larger faces)
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

    def postprocess(
        self,
        input: np.ndarray,
        output: List[np.ndarray],
        scale_ratio: float = 1.0,
        max_num: int = 0,
        metric: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Post-process model outputs to get final detections.

        Args:
            input (numpy.ndarray): Model input tensor, shape (1, 3, H, W)
            output (list): Model output tensors
            scale_ratio (float): Scale ratio for converting back to original image size
            max_num (int): Maximum number of detections to keep
            metric (str): Filtering metric ('max' or 'default')

        Returns:
            tuple: A tuple containing:
                - boxes (numpy.ndarray): Final detection boxes, shape (num_final, 4)
                - scores (numpy.ndarray): Final detection scores, shape (num_final,)
                - keypoints (numpy.ndarray): Final keypoint results, shape (num_final, 5, 2)
        """
        input_height = input.shape[2]
        input_width = input.shape[3]

        scores_list, bboxes_list, keypoints_list = self._process_all_feature_maps(
            output, input_height, input_width
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

    def detect(
        self, img: np.ndarray, max_num: int = -1, metric: str = "default"
    ) -> List[Face]:
        """Detect faces in an image.

        Args:
            img (numpy.ndarray): Input image, shape (H, W, 3)
            max_num (int): Maximum number of detections to keep
            metric (str): Filtering metric ('max' or 'default')

        Returns:
            list[Face]: List of face detections.
        """
        model_input, scale_ratio = self.preprocess(img)
        model_output = self.forward(model_input)
        boxes, scores, keypoints = self.postprocess(
            model_input, model_output, scale_ratio, max_num, metric
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

    def draw(
        self, img: np.ndarray, faces: List[Face], labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Draw detection results on the image.

        Args:
            img (numpy.ndarray): Input image to draw on, shape (H, W, 3)
            faces (list[Face]): Face detections to draw
            labels (list, optional): List of labels to show for each detection
                Length should match number of detections

        Returns:
            numpy.ndarray: Image with visualizations
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
            font_scale = 0.7  # Increased from 0.5
            font_thickness = 2  # Increased from 1
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
                    cv2.circle(
                        vis_img, tuple(kp), 4, (0, 0, 255), -1
                    )  # Increased radius from 2 to 4

        return vis_img
