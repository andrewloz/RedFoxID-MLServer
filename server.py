from ultralytics import YOLO
from concurrent import futures
import logging

import grpc
import numpy as np
import cv2
from PIL import Image

import detect_object_pb2_grpc as pbgrpc
import detect_object_pb2 as pb
import configparser


from server_utils import results_to_proto_boxes

import time
from timings_opt import TimingsInterceptor, start_metrics_server, phase_observe

class DetectObjectService(pbgrpc.DetectObjectServicer):
    def __init__(self):
        self.config = config()["InferenceServer"]
        print("Configuration:", dict(self.config))

        self.models = {}

    def _get_model(self, name):
        if name not in self.models:
            self.models[name] = YOLO(f"model/{name}", task="detect")
        return self.models[name]

    def Request(self, request, context):
        if not getattr(request, "model_name", None):
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No model loaded, please provide model weights")
            return pb.ResponsePayload(objects=[])

        model = self._get_model(request.model_name)

        # ----- deserialize (timed) -----
        t0 = time.perf_counter()
        img = Image.frombytes(
            "RGBA",
            (request.image_width, request.image_height),
            bytes(request.image_rgba_bytes),
        ).convert("RGB")
        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Could not decode image")
            return pb.ResponsePayload(objects=[])
        t1 = time.perf_counter(); phase_observe("deserialize", request.model_name, t1 - t0)

        # ----- model inference (timed) -----
        t2 = time.perf_counter()
        results = model.predict(
            img,
            conf=request.confidence_threshold,
            iou=request.iou_threshold,
            verbose=bool(int(self.config.get("Verbose", "0"))),
            save=bool(int(self.config.get("SaveImg", "0"))),
            project="./output",
            name=request.image_name,
            device=self.config.get("Device", "")
        )
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # accurate GPU timing
        except Exception:
            pass
        t3 = time.perf_counter(); phase_observe("model", request.model_name, t3 - t2)

        # optional flatten
        if bool(int(self.config.get("SaveImg", "0"))):
            from flatten_output_dir import flatten
            from pathlib import Path
            flatten(Path("./output") / "Images")

        # ----- serialize (timed) -----
        t4s = time.perf_counter()
        objects = results_to_proto_boxes(results[0], pb)
        t4e = time.perf_counter(); phase_observe("serialize", request.model_name, t4e - t4s)
        phase_observe("total", request.model_name, t4e - t0)

        return objects

def serve():
    cfg = config()["InferenceServer"]
    port = cfg.get("Port", "50051")

    if bool(int(cfg.get("Profiling", "0"))):
            start_metrics_server(8000)

    interceptors = []
    if TimingsInterceptor is not None:
        interceptors.append(TimingsInterceptor())


    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(cfg.get("MaxWorkers", "1"))),
        interceptors=interceptors
    )

    pbgrpc.add_DetectObjectServicer_to_server(DetectObjectService(), server)
    server.add_insecure_port("[::]:" + port) # do we need any kind of secure transport if its inside a secure network?
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

def config():
    config = configparser.ConfigParser()
    with open("config.ini", "r") as f:
        config.read_file(f)
    return config


if __name__ == "__main__":
    logging.basicConfig()
    serve()