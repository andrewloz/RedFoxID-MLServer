#! /usr/bin/env python3

from concurrent import futures
import argparse
import logging
import os

import grpc
import numpy as np
import cv2

import detect_object_pb2_grpc as pbgrpc
import detect_object_pb2 as pb

from server_utils import arrays_to_proto_boxes
from src.config import Config
from src.inference.pipeline import ModelPipeline
from src.inference.backends import UltralyticsBackend, OpenvinoBackend, OnnxBackend
from src.inference.preprocess import prepare_rgba_bytes, prepare_yolo_input
from src.inference.postprocess import process_ultralytics_results, process_yolo_results

APP_VERSION = "0.0.0"


class DetectObjectService:
    def __init__(self, config_path: str):
        cfg, models = Config(config_path).getAll()
        self.config = cfg
        self.models: dict[str, ModelPipeline] = {}
        backend_type = cfg.get("Backend", "ultralytics").lower()
        model_type = cfg.get("ModelType", "yolo").lower()
        print(f"Using backend: {backend_type}, model type: {model_type}")
        print("Configuration:", dict(self.config))

        device_cfg = self.config.get("Device", "") or None
        verbose_flag = bool(int(self.config.get("Verbose", "0")))

        # Select backend, preprocessing, and postprocessing based on config
        backend_class = None
        preprocess_fn = None
        postprocess_fn = None

        # Backend selection
        if backend_type == "ultralytics":
            backend_class = UltralyticsBackend
            if model_type == "yolo":
                preprocess_fn = prepare_rgba_bytes
                postprocess_fn = process_ultralytics_results
            else:
                raise ValueError(f"ModelType '{model_type}' not supported with Ultralytics backend")
                
        elif backend_type == "openvino":
            backend_class = OpenvinoBackend
            if model_type == "yolo":
                preprocess_fn = prepare_yolo_input
                postprocess_fn = process_yolo_results
            else:
                raise ValueError(f"ModelType '{model_type}' not yet implemented for OpenVINO backend")
                
        elif backend_type == "onnx":
            backend_class = OnnxBackend
            if model_type == "yolo":
                preprocess_fn = prepare_yolo_input
                postprocess_fn = process_yolo_results
            else:
                raise ValueError(f"ModelType '{model_type}' not yet implemented for ONNX backend")
        else:
            raise ValueError(f"Unknown backend: '{backend_type}'. Supported: ultralytics, openvino, onnx")

        # Load models with selected backend
        for model_path in models:
            name = os.path.basename(model_path) or model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} does not exist!")

            print(f"Loading model {name}")
            backend = backend_class(
                model_path=str(model_path),
                device=device_cfg,
                verbose=verbose_flag,
            )
            pipeline = ModelPipeline(preprocess_fn, backend, postprocess_fn)
            self.models[name] = pipeline

        if not self.models:
            raise RuntimeError("No models configured for inference.")

        # Warmup inference to pre-compile model for GPU/OpenVINO
        warmup_enabled = bool(int(self.config.get("WarmupInference", "0")))
        warmup_image_path = self.config.get("WarmupImagePath", "tests/img_test.png")
        
        if warmup_enabled and os.path.exists(warmup_image_path):
            print(f"Running warmup inference with {warmup_image_path}...")
            # Load as RGBA for warmup
            warmup_image = cv2.imread(warmup_image_path, cv2.IMREAD_UNCHANGED)
            if warmup_image is not None:
                # Ensure RGBA format
                if warmup_image.shape[2] == 3:
                    warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2BGRA)
                
                for name, model in self.models.items():
                    try:
                        print(f"  Warming up model: {name}")
                        _ = model(
                            warmup_image,
                            imgsz=(640, 640),
                            conf=0.25,
                            iou=0.7,
                            verbose=False,
                            save=False,
                        )
                        print(f"  Warmup complete for {name}")
                    except Exception as e:
                        print(f"  Warning: Warmup failed for {name}: {e}")
        elif warmup_enabled:
            print(f"Warning: Warmup enabled but image not found at {warmup_image_path}")

    def _get_model(self, name: str) -> ModelPipeline:
        if name in self.models:
            return self.models[name]
        if len(self.models) == 1:
            return next(iter(self.models.values()))
        raise Exception(
            f"Model '{name}' not available. Available models: {', '.join(self.models.keys())}"
        )

    def Request(self, request, context):
        try:
            if not getattr(request, "image_bytes", None):
                raise Exception("No image_bytes in payload loaded, please provide image bytes for inference")

            if not getattr(request, "model_name", None):
                raise Exception("No model_name in payload loaded, please provide model_name")
            
            if not request.image_width or not request.image_height:
                raise Exception("image_width and image_height must be provided")

            model = self._get_model(request.model_name)

            # Reshape raw RGBA bytes to numpy array (fastest path)
            image_array = np.frombuffer(request.image_bytes, dtype=np.uint8).reshape(
                request.image_height, request.image_width, 4
            )

            pipeline_kwargs = {
                "imgsz": (640, 640),
                "conf": request.confidence_threshold,
                "iou": request.iou_threshold,
                "verbose": bool(int(self.config.get("Verbose", "0"))),
                "save": bool(int(self.config.get("SaveImg", "0"))),
                "project": "./output",
                "name": request.image_name,
            }

            boxes, scores, cls_ids = model(
                image_array,
                **pipeline_kwargs,
            )

            results = arrays_to_proto_boxes(
                boxes,
                scores,
                cls_ids,
                pb,
                getattr(model, "class_names", None),
            )

            if pipeline_kwargs["save"]:
                from flatten_output_dir import flatten
                from pathlib import Path

                flatten(Path("./output") / "Images")

            return results

        except Exception as e:
            print(e)
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(e))
            return pb.ResponsePayload(objects=[])


def serve(config_path: str):
    cfg, _ = Config(config_path).getAll()
    port = cfg.get("Port", "50051")
    host = cfg.get("Host", "[::]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(cfg.get("MaxWorkers", "1"))))
    pbgrpc.add_DetectObjectServicer_to_server(DetectObjectService(config_path), server)
    server.add_insecure_port(f"{host}:{str(port)}")
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    parser = argparse.ArgumentParser(description="RedFox ML Inference Server")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to configuration INI file.")
    parser.add_argument("-v", "--version", action="store_true", help="Print application version and exit")
    args = parser.parse_args()

    if args.version:
        print(APP_VERSION)
        raise SystemExit(0)

    if args.config_path is None:
        parser.error("config_path is required")

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(f"Config file '{args.config_path}' not found.")

    serve(args.config_path)
