from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import openvino as ov
import yaml


Inputs = Tuple[np.ndarray, ...]
Outputs = Tuple[Any, ...]


class OpenvinoBackend:
    """OpenVINO inference backend for model-agnostic inference."""
    
    __slots__ = ("_compiled_model", "_input_name", "class_names")

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenVINO backend.
        
        Args:
            model_path: Path to model directory (containing .xml and .bin files) or .xml file
            device: Device to run inference on (CPU, GPU, AUTO, NPU, etc.)
            verbose: Enable verbose output
            **kwargs: Additional arguments (unused, for compatibility)
        """
        model_path_obj = Path(model_path)
        
        # Find .xml file
        if model_path_obj.is_dir():
            xml_files = list(model_path_obj.glob("*.xml"))
            if not xml_files:
                raise FileNotFoundError(f"No .xml file found in {model_path}")
            if len(xml_files) > 1:
                raise ValueError(f"Multiple .xml files found in {model_path}, please specify exact file")
            xml_path = xml_files[0]
        elif model_path_obj.suffix == ".xml":
            xml_path = model_path_obj
        else:
            raise ValueError(f"model_path must be a directory or .xml file, got {model_path}")
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        
        # Load and compile model
        core = ov.Core()
        model = core.read_model(str(xml_path))
        
        # Compile model with device selection
        device = device or "AUTO"
        if verbose:
            print(f"Compiling OpenVINO model on device: {device}")
        
        self._compiled_model = core.compile_model(model, device)
        
        # Get input name for inference
        input_port = self._compiled_model.input(0)
        self._input_name = input_port.any_name
        
        if verbose:
            input_shape = input_port.shape
            elem_type = input_port.element_type
            print(f"Model input: {self._input_name}, shape: {input_shape}, dtype: {elem_type}")
        
        # Extract class names from metadata.yaml if available
        self.class_names = self._load_class_names(xml_path.parent, verbose=verbose)

    def _load_class_names(self, model_dir: Path, verbose: bool = False) -> Optional[Dict[int, str]]:
        """Load class names from metadata.yaml if available."""
        metadata_path = model_dir / "metadata.yaml"
        if not metadata_path.exists():
            if verbose:
                print("No metadata.yaml found, class_names will be None")
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
            
            names = metadata.get("names")
            if names is None:
                return None
            
            # Convert to dict if it's already a dict, otherwise assume list
            if isinstance(names, dict):
                class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                class_names = {i: str(name) for i, name in enumerate(names)}
            else:
                if verbose:
                    print(f"Unexpected names format in metadata: {type(names)}")
                return None
            
            if verbose:
                print(f"Loaded {len(class_names)} class names from metadata.yaml")
            
            return class_names
            
        except Exception as e:
            if verbose:
                print(f"Failed to load class names from metadata.yaml: {e}")
            return None

    def infer(self, inputs: Inputs, **kwargs: Any) -> Outputs:
        """
        Run inference on preprocessed inputs.
        
        Args:
            inputs: Tuple containing preprocessed image tensor (already in NCHW float32 format)
            **kwargs: Additional arguments (unused, for compatibility)
            
        Returns:
            Tuple containing raw model output tensors
        """
        image = inputs[0]
        
        # Create inference request and run
        infer_request = self._compiled_model.create_infer_request()
        infer_request.infer({self._input_name: image})
        
        # Collect all output tensors
        outputs = []
        for i in range(len(self._compiled_model.outputs)):
            out_tensor = infer_request.get_output_tensor(i).data
            # Make a copy to ensure data is not overwritten on next inference
            outputs.append(np.copy(out_tensor))
        
        return tuple(outputs)

    def close(self) -> None:
        """Clean up resources."""
        # OpenVINO resources are automatically managed, but keep method for interface completeness
        pass

