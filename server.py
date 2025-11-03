#! /usr/bin/env python3

from concurrent import futures
import argparse
import logging
import os

import grpc

import detect_object_pb2_grpc as pbgrpc
import detect_object_pb2 as pb

from server_utils import arrays_to_proto_boxes
from src.config import Config
from src.inference.pipeline import ModelPipeline
from src.inference.backends import UltralyticsBackend
from src.inference.preprocess import prepare_png_bytes
from src.inference.postprocess import process_results

APP_VERSION = "0.0.0"


class DetectObjectService:
    def __init__(self, config_path: str):
        cfg, models = Config(config_path).getAll()
        self.config = cfg
        self.models: dict[str, ModelPipeline] = {}
        self.inferenceLib = cfg.get("InferenceLibrary", "ultralytics")
        print(f"Using library: {self.inferenceLib}")
        print("Configuration:", dict(self.config))

        if self.inferenceLib != "ultralytics":
            raise NotImplementedError(
                "ModelPipeline server path currently supports only 'ultralytics'."
            )

        device_cfg = self.config.get("Device", "") or None
        verbose_flag = bool(int(self.config.get("Verbose", "0")))

        for model_path in models:
            name = os.path.basename(model_path) or model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} does not exist!")

            print(f"loading model {name}")
            backend = UltralyticsBackend(
                model_path=str(model_path),
                device=device_cfg,
                verbose=verbose_flag,
            )
            pipeline = ModelPipeline(prepare_png_bytes, backend, process_results)
            self.models[name] = pipeline

        if not self.models:
            raise RuntimeError("No models configured for inference.")

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
            if not getattr(request, "image_png_bytes", None):
                raise Exception("No image_png_bytes in payload loaded, please provide image bytes for inference")

            if not getattr(request, "model_name", None):
                raise Exception("No model_name in payload loaded, please provide model_name")

            model = self._get_model(request.model_name)

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
                request.image_png_bytes,
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
