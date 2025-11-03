from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

Inputs = Tuple[np.ndarray, ...]
Meta = dict


def prepare_yolo_input(raw_input: Any) -> Tuple[Inputs, Meta]:
    """
    Prepare input for standard YOLO models (OpenVINO, ONNX, etc.).
    
    Converts RGBA image to float32 NCHW format normalized to [0, 1].
    
    Args:
        raw_input: RGBA numpy array (H, W, 4)
        
    Returns:
        Tuple of (inputs, meta) where inputs is (image_nchw,) and meta contains original shape
    """
    if not isinstance(raw_input, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(raw_input)!r}")

    if raw_input.ndim != 3 or raw_input.shape[2] != 4:
        raise ValueError(f"Expected RGBA image with 4 channels, got shape {raw_input.shape}")

    # Store original shape for postprocessing
    orig_shape = raw_input.shape[:2]
    
    # Drop alpha and reverse to BGR: RGBA -> BGR
    image_bgr = raw_input[:, :, 2::-1]
    
    # Convert BGR uint8 HWC [0, 255] -> float32 HWC [0, 1]
    image_float = image_bgr.astype(np.float32) / 255.0
    
    # Transpose HWC -> CHW and add batch dimension -> NCHW
    image_nchw = np.transpose(image_float, (2, 0, 1))[None, ...]
    
    # Ensure contiguous memory layout
    image_nchw = np.ascontiguousarray(image_nchw, dtype=np.float32)
    
    meta: Meta = {
        "orig_shape": orig_shape,
    }

    return (image_nchw,), meta

