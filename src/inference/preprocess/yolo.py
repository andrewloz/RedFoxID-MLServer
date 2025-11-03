from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

Inputs = Tuple[np.ndarray, ...]
Meta = dict


def prepare_png_bytes(raw_input: Any) -> Tuple[Inputs, Meta]:
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

    image = np.ascontiguousarray(image)
    meta: Meta = {
        "orig_shape": image.shape[:2],
    }

    return (image,), meta

