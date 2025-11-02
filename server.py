#! /usr/bin/env python3

from ultralytics import YOLO
from concurrent import futures
import logging
import argparse
import os

import grpc
import numpy as np
import cv2

import detect_object_pb2_grpc as pbgrpc
import detect_object_pb2 as pb

from server_utils import results_to_proto_boxes
from src.config import Config

APP_VERSION = "0.0.0"

class DetectObjectService():
    def __init__(self, config_path: str):
        cfg, models = Config(config_path).getAll()
        self.config = cfg
        self.models = {}
        print("Configuration:", dict(self.config))
        print("Available Models:", self.models)

        for m in models:
            name = os.path.basename(m)
            if not os.path.exists(m):
                raise FileNotFoundError(f"{m} does not exist!")
            if name == "":
                name = m
            print(f"loading model {name}")
            self.models[name] = YOLO(f"{m}", task="detect")
            device = self.config.get('Device', '')
            print(f"Device configure: {device}")
            self.models[name].predict(device=device)
            break  # load only the first model

    def _get_model(self, name):
        first_model = next(iter(self.models.values()))
        return first_model

    def Request(self, request, context):
        try:
            if not getattr(request, "image_png_bytes", None):
                raise Exception("No image_png_bytes in payload loaded, please provide image btyes for inference")

            model = self._get_model(request.model_name)

            np_arr = np.frombuffer(request.image_png_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Could not decode image")
                return pb.ResponsePayload(objects=[])

            results = model.predict(
                img,
                imgsz=(img.shape[0], img.shape[1]),
                conf=request.confidence_threshold,
                iou=request.iou_threshold,
                verbose=bool(int(self.config.get("Verbose", "0"))),
                save=bool(int(self.config.get("SaveImg", "0"))),
                project="./output",
                name=request.image_name,
                device=self.config.get("Device", "")
            )

            if bool(int(self.config.get("SaveImg", "0"))):
                from flatten_output_dir import flatten
                from pathlib import Path
                flatten(Path("./output") / "Images")

            objects = results_to_proto_boxes(results[0], pb)
            return objects

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

