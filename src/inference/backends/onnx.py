from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


Inputs = Tuple[np.ndarray, ...]
Outputs = Tuple[Any, ...]


class OnnxBackend:
    """ONNX Runtime inference backend (stub for future implementation)."""
    
    __slots__ = ("class_names",)

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ONNX backend.
        
        Args:
            model_path: Path to .onnx model file
            device: Device to run inference on (CPU, CUDA, etc.)
            verbose: Enable verbose output
            **kwargs: Additional arguments (unused, for compatibility)
        """
        self.class_names = None
        raise NotImplementedError("ONNX backend is not yet implemented")

    def infer(self, inputs: Inputs, **kwargs: Any) -> Outputs:
        """
        Run inference on preprocessed inputs.
        
        Args:
            inputs: Tuple containing preprocessed image tensor
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing raw model output tensors
        """
        raise NotImplementedError("ONNX backend is not yet implemented")

    def close(self) -> None:
        """Clean up resources."""
        pass

