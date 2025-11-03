from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

Inputs = Tuple[np.ndarray, ...]
Meta = dict


def prepare_yolo_input(raw_input: Any) -> Tuple[Inputs, Meta]:
    """
    Prepare input for standard YOLO models (OpenVINO, ONNX, etc.).
    
    Converts image to float32 NCHW format normalized to [0, 1].
    
    Args:
        raw_input: Image bytes, bytearray, or numpy array
        
    Returns:
        Tuple of (inputs, meta) where inputs is (image_nchw,) and meta contains original shape
    """
    if isinstance(raw_input, (bytes, bytearray, memoryview)):
        np_view = np.frombuffer(raw_input, dtype=np.uint8)
        image = cv2.imdecode(np_view, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes for YOLO inference")
    elif isinstance(raw_input, np.ndarray):
        image = raw_input
    else:
        raise TypeError(f"Unsupported input type {type(raw_input)!r} for YOLO preprocessing")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected BGR image with 3 channels, got shape {image.shape}")

    # Store original shape for postprocessing
    orig_shape = image.shape[:2]
    
    # Convert BGR uint8 HWC [0, 255] -> float32 HWC [0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # Transpose HWC -> CHW and add batch dimension -> NCHW
    image_nchw = np.transpose(image_float, (2, 0, 1))[None, ...]
    
    # Ensure contiguous memory layout
    image_nchw = np.ascontiguousarray(image_nchw, dtype=np.float32)
    
    meta: Meta = {
        "orig_shape": orig_shape,
    }

    return (image_nchw,), meta

