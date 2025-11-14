"""Backend adapters for inference runtimes."""

from .onnx import OnnxBackend

try:
    from .openvino import OpenvinoBackend
except ImportError:
    OpenvinoBackend = None

__all__ = ["OnnxBackend"]

if OpenvinoBackend is not None:
    __all__.append("OpenvinoBackend")

# Conditionally import ultralytics backend if available
try:
    from .ultralytics import UltralyticsBackend
except ImportError:
    UltralyticsBackend = None
else:
    __all__.append("UltralyticsBackend")
