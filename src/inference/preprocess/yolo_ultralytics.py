from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

Inputs = Tuple[np.ndarray, ...]
Meta = dict


def prepare_rgba_bytes(raw_input: Any) -> Tuple[Inputs, Meta]:
    if not isinstance(raw_input, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(raw_input)!r}")

    if raw_input.ndim != 3 or raw_input.shape[2] != 4:
        raise ValueError(f"Expected RGBA image with 4 channels, got shape {raw_input.shape}")

    # Drop alpha channel: RGBA -> BGR (OpenCV convention)
    image = raw_input[:, :, 2::-1]  # Reverse RGB to BGR, drop alpha
    
    meta: Meta = {
        "orig_shape": image.shape[:2],
    }

    return (image,), meta

