from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np


Outputs = Tuple[Any, ...]
Meta = dict


def process_yolo_results(
    outputs: Outputs,
    meta: Meta,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: Optional[Iterable[int]] = None,
    max_det: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process raw YOLO model outputs from OpenVINO/ONNX backends.
    
    Expects output format: (1, 5+num_classes, N) or (1, 5, N)
    Where each detection is [cx, cy, w, h, conf, ...class_probs]
    
    Args:
        outputs: Raw model output tensors
        meta: Metadata from preprocessing (contains orig_shape)
        conf: Confidence threshold
        iou: IoU threshold for NMS
        classes: Optional list of class indices to keep
        max_det: Maximum number of detections
        
    Returns:
        Tuple of (boxes, scores, classes) where:
            boxes: (N, 4) array in xyxy format
            scores: (N,) array of confidence scores
            classes: (N,) array of class indices
    """
    return _process_yolo_results(
        outputs,
        meta,
        conf=conf,
        iou=iou,
        classes=classes,
        max_det=max_det,
        **kwargs,
    )


def _process_yolo_results(
    outputs: Outputs,
    meta: Meta,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: Optional[Iterable[int]] = None,
    max_det: Optional[int] = None,
    **_: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Internal implementation of YOLO result processing."""
    
    if not outputs or len(outputs) == 0:
        return _empty()
    
    # Get raw output tensor - expected shape (1, 5+num_classes, N) or (1, 5, N)
    raw_output = outputs[0]
    
    if raw_output.ndim != 3:
        raise ValueError(f"Expected 3D output tensor, got shape {raw_output.shape}")
    
    # For single-class models: (1, 5, N) with [cx, cy, w, h, conf]
    # For multi-class models: (1, 5+num_classes, N) with [cx, cy, w, h, conf, ...class_probs]
    
    if raw_output.shape[1] == 5:
        # Single class model
        boxes_xyxy, scores = _yolo_nms_single_class(
            raw_output, conf_threshold=conf, iou_threshold=iou
        )
        cls_np = np.zeros(len(scores), dtype=np.int32)  # All class 0
    else:
        # Multi-class model (not yet implemented, will add when needed)
        raise NotImplementedError("Multi-class YOLO output not yet supported")
    
    if len(boxes_xyxy) == 0:
        return _empty()
    
    # Apply class filter if specified
    if classes is not None:
        class_set = set(classes)
        mask = np.isin(cls_np, list(class_set))
        if not np.any(mask):
            return _empty()
        boxes_xyxy = boxes_xyxy[mask]
        scores = scores[mask]
        cls_np = cls_np[mask]
    
    # Apply max_det limit
    if max_det is not None and len(boxes_xyxy) > max_det:
        order = np.argsort(scores)[::-1][:max_det]
        boxes_xyxy = boxes_xyxy[order]
        scores = scores[order]
        cls_np = cls_np[order]
    
    # Clip boxes to original image dimensions
    orig_h, orig_w = meta.get("orig_shape", (None, None))
    if orig_h is not None and orig_w is not None:
        boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, orig_w - 1)
        boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, orig_h - 1)
    
    return (
        np.ascontiguousarray(boxes_xyxy, dtype=np.float32),
        np.ascontiguousarray(scores, dtype=np.float32),
        np.ascontiguousarray(cls_np, dtype=np.int32),
    )


def _yolo_nms_single_class(
    tensor: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply NMS to single-class YOLO output.
    
    Args:
        tensor: (1, 5, N) with [cx, cy, w, h, conf]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        boxes: (M, 4) in xyxy format
        scores: (M,) confidence scores
    """
    assert tensor.ndim == 3 and tensor.shape[1] == 5, f"Unexpected shape {tensor.shape}"
    
    # Transpose to (N, 5)
    predictions = tensor[0].T  # (N, 5)
    boxes_cxcywh = predictions[:, :4]
    scores = predictions[:, 4]
    
    # Apply confidence threshold
    mask = scores >= conf_threshold
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    
    # Convert boxes from cxcywh to xyxy
    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh[mask])
    scores = scores[mask]
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    # Apply NMS
    keep = []
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        rest = order[1:]
        iou = _iou_xyxy(boxes_xyxy[i], boxes_xyxy[rest], areas[i], areas[rest])
        order = rest[iou <= iou_threshold]
    
    return boxes_xyxy[keep], scores[keep]


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)."""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_xyxy(
    box: np.ndarray,
    boxes: np.ndarray,
    area_box: float,
    area_boxes: np.ndarray,
) -> np.ndarray:
    """
    Calculate IoU between one box and multiple boxes.
    
    Args:
        box: (4,) single box in xyxy format
        boxes: (M, 4) boxes in xyxy format
        area_box: Area of single box
        area_boxes: (M,) areas of boxes
        
    Returns:
        (M,) IoU values
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h
    
    union = area_box + area_boxes - intersection + 1e-9
    return intersection / union


def _empty() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return empty result arrays."""
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
    )

