from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np


Outputs = Tuple[Any, ...]
Meta = dict


def process_results(
    outputs: Outputs,
    meta: Meta,
    *,
    conf: float = 0.25,
    classes: Optional[Iterable[int]] = None,
    max_det: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _process_results(
        outputs,
        meta,
        conf=conf,
        classes=classes,
        max_det=max_det,
        **kwargs,
    )


def _process_results(
    outputs: Outputs,
    meta: Meta,
    *,
    conf: float = 0.25,
    classes: Optional[Iterable[int]] = None,
    max_det: Optional[int] = None,
    **_: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = outputs[0]
    if not results:
        return _empty()

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return _empty()

    xyxy = boxes.xyxy
    scores = boxes.conf
    cls = boxes.cls

    if xyxy is None or scores is None or cls is None:
        return _empty()

    xyxy_np = xyxy.cpu().numpy().astype(np.float32, copy=False)
    scores_np = scores.cpu().numpy().astype(np.float32, copy=False)
    cls_np = cls.cpu().numpy().astype(np.int32, copy=False)

    mask = scores_np >= conf
    if classes is not None:
        class_set = set(classes)
        mask &= np.isin(cls_np, list(class_set))

    if not np.any(mask):
        return _empty()

    xyxy_np = xyxy_np[mask]
    scores_np = scores_np[mask]
    cls_np = cls_np[mask]

    if max_det is not None and xyxy_np.shape[0] > max_det:
        order = np.argsort(scores_np)[::-1][:max_det]
        xyxy_np = xyxy_np[order]
        scores_np = scores_np[order]
        cls_np = cls_np[order]

    orig_h, orig_w = meta.get("orig_shape", (None, None))
    if orig_h is not None and orig_w is not None:
        xyxy_np[:, 0::2] = np.clip(xyxy_np[:, 0::2], 0, orig_w - 1)
        xyxy_np[:, 1::2] = np.clip(xyxy_np[:, 1::2], 0, orig_h - 1)

    return (
        np.ascontiguousarray(xyxy_np, dtype=np.float32),
        np.ascontiguousarray(scores_np, dtype=np.float32),
        np.ascontiguousarray(cls_np, dtype=np.int32),
    )


def _empty() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
    )

