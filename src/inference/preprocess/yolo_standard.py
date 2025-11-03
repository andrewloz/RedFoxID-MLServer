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
    h, w = orig_shape
    
    # Pre-allocate output buffer (NCHW format)
    image_nchw = np.empty((1, 3, h, w), dtype=np.float32)
    
    # Vectorized: extract BGR channels, transpose, convert, normalize in minimal ops
    # Use direct indexing into pre-allocated buffer
    image_nchw[0, 0] = raw_input[:, :, 2].astype(np.float32, copy=False) * np.float32(1.0/255.0)
    image_nchw[0, 1] = raw_input[:, :, 1].astype(np.float32, copy=False) * np.float32(1.0/255.0)
    image_nchw[0, 2] = raw_input[:, :, 0].astype(np.float32, copy=False) * np.float32(1.0/255.0)
    
    meta: Meta = {
        "orig_shape": orig_shape,
    }

    return (image_nchw,), meta

