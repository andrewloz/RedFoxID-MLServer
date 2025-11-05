from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_detection_image(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    meta: Mapping[str, object],
    *,
    project: Optional[str] = None,
    name: Optional[str] = None,
    class_names: Optional[Mapping[int, str]] = None,
    verbose: bool = False,
) -> None:
    """
    Persist an annotated detection image using the original RGBA frame stored in meta.

    Args:
        boxes: Array of bounding boxes in xyxy format.
        scores: Array of confidence scores.
        classes: Array of class indices.
        meta: Metadata dictionary produced during preprocessing.
        project: Root output directory (defaults to current directory).
        name: Base filename (defaults to 'prediction').
        class_names: Optional mapping of class index to name.
        verbose: When True, emit simple diagnostic messages on failure.
    """
    image = meta.get("orig_rgba")
    if image is None:
        if verbose:
            print("Save hook: missing orig_rgba in meta, skipping image save")
        return

    try:
        image_array = np.asarray(image)
    except Exception as exc:
        if verbose:
            print(f"Save hook: failed to read image array ({exc}), skipping image save")
        return

    project_dir = Path(project or ".")
    images_dir = project_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    base_name = name or "prediction"
    save_path = images_dir / f"{base_name}.png"

    pil_image = Image.fromarray(image_array, mode="RGBA").convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    outline_color = (255, 0, 0)
    text_bg_color = (255, 0, 0)
    text_color = (255, 255, 255)

    for box, score, cls_idx in zip(boxes, scores, classes):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        draw.rectangle([(x1, y1), (x2, y2)], outline=outline_color, width=2)

        if class_names is not None and cls_idx in class_names:
            label = f"{class_names[cls_idx]} {score:.2f}"
        else:
            label = f"{int(cls_idx)} {score:.2f}"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_origin = (x1, max(y1 - text_height - 4, 0))
        bg_corner = (x1 + text_width + 4, text_origin[1] + text_height + 4)

        draw.rectangle([text_origin, bg_corner], fill=text_bg_color)
        draw.text((text_origin[0] + 2, text_origin[1] + 2), label, fill=text_color, font=font)

    pil_image.save(save_path, format="PNG")

    if verbose:
        print(f"Save hook: wrote annotated image to {save_path}")
