"""Backend adapters for inference runtimes."""

from .onnx import OnnxBackend
from .openvino import OpenvinoBackend

# Conditionally import ultralytics backend if available
try:
    from .ultralytics import UltralyticsBackend
    __all__ = ["UltralyticsBackend", "OpenvinoBackend", "OnnxBackend"]
except ImportError:
    UltralyticsBackend = None
    __all__ = ["OpenvinoBackend", "OnnxBackend"]

