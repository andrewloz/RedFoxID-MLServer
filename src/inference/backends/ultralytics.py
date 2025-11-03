from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from ultralytics import YOLO

Inputs = Tuple[np.ndarray, ...]
Outputs = Tuple[Any, ...]


class UltralyticsBackend:
    __slots__ = ("_model", "_predict_kwargs", "class_names")

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        default_conf: float = 0.0,
        default_iou: float = 0.7,
        verbose: bool = False,
        **predict_kwargs: Any,
    ) -> None:
        self._model = YOLO(model_path, task="detect")
        if device:
            self._model.to(device)

        self._predict_kwargs: Dict[str, Any] = {
            "conf": default_conf,
            "iou": default_iou,
            "verbose": verbose,
            **predict_kwargs,
        }
        names = getattr(self._model, "names", None)
        self.class_names = dict(names) if isinstance(names, dict) else names

    def infer(self, inputs: Inputs, **kwargs: Any) -> Outputs:
        image = inputs[0]
        override_keys = (
            "conf",
            "iou",
            "imgsz",
            "classes",
            "agnostic_nms",
            "save",
            "project",
            "name",
            "verbose",
            "device",
        )
        overrides = {key: kwargs[key] for key in override_keys if key in kwargs}

        if overrides:
            predict_kwargs = {**self._predict_kwargs, **overrides}
        else:
            predict_kwargs = self._predict_kwargs

        results = self._model.predict(image, **predict_kwargs)
        return (results,)

    def close(self) -> None:
        # Ultralytics models do not expose an explicit close, but keep method for interface completeness.
        pass

