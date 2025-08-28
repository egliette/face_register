from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.linalg import norm
from skimage import transform as trans


def compute_sim(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings.

    Args:
        feat1 (numpy.ndarray): First face embedding vector
        feat2 (numpy.ndarray): Second face embedding vector

    Returns:
        float: Cosine similarity score between -1 and 1
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in z.

    Args:
        z (numpy.ndarray): Input array of shape (N, C) where N is the number of samples
                          and C is the number of classes.

    Returns:
        numpy.ndarray: Softmax probabilities of shape (N, C) where each row sums to 1.
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # Shape: (N, 1)
    e_x = np.exp(z - s)  # Shape: (N, C)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # Shape: (N, 1)
    return e_x / div  # Shape: (N, C)


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance prediction to bounding box.

    Args:
        points (numpy.ndarray): Shape (N, 2), [x, y] coordinates of anchor centers.
        distance (numpy.ndarray): Shape (N, 4), distances from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple, optional): Shape of the image (H, W). If provided, the decoded
            bbox coordinates will be clipped to this shape.

    Returns:
        numpy.ndarray: Decoded bboxes of shape (N, 4), each row is [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]  # Shape: (N,)
    y1 = points[:, 1] - distance[:, 1]  # Shape: (N,)
    x2 = points[:, 0] + distance[:, 2]  # Shape: (N,)
    y2 = points[:, 1] + distance[:, 3]  # Shape: (N,)
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)  # Shape: (N, 4)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance prediction to keypoints.

    Args:
        points (numpy.ndarray): Shape (N, 2), [x, y] coordinates of anchor centers.
        distance (numpy.ndarray): Shape (N, 10), distances from the given point to 5
            keypoints (each keypoint has x and y distances).
        max_shape (tuple, optional): Shape of the image (H, W). If provided, the decoded
            keypoint coordinates will be clipped to this shape.

    Returns:
        numpy.ndarray: Decoded keypoints of shape (N, 5, 2), where each detection has
            5 keypoints, each with (x, y) coordinates.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]  # Shape: (N,)
        py = points[:, i % 2 + 1] + distance[:, i + 1]  # Shape: (N,)
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)  # Shape: (N, 10) -> reshape to (N, 5, 2) later


def nms(dets: np.ndarray, thresh: float = 0.4) -> List[int]:
    """Apply non-maximum suppression to detection results.

    Args:
        dets (numpy.ndarray): Detection results with shape (N, 5), where each row is [x1, y1, x2, y2, score]
        thresh (float): NMS threshold

    Returns:
        list: Indices of kept detections
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def norm_crop2(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
