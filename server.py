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
import os


from server_utils import results_to_proto_boxes
from config import config

class DetectObjectService(pbgrpc.DetectObjectServicer):
    def __init__(self):
        cfg, models = config()
        self.config = cfg
        self.models = {}
        print("Configuration:", dict(self.config))
        print("Available Models:",self.models)

        for m in models:
            name = os.path.basename(m)

            if not os.path.exists(m):
                raise FileNotFoundError(f"{m} does not exist!")

            print(f"loading model {name}")
            
            self.models[name] = YOLO(f"{m}", task="detect")
            # warm up the models, uses an asset from the ultralytics package to test, you will see warning in console.
            self.models[name].predict()

    def _get_model(self, name):
        if name not in self.models:
            raise Exception(f"Model not available available models are: {', '.join(list(self.models.keys()))}")
        return self.models[name]

    def Request(self, request, context):
        try:
            if not getattr(request, "model_name", None):
                raise Exception("No model_name in payload loaded, please provide model_name")
            
            model = self._get_model(request.model_name)

            img = Image.frombytes(
                "RGBA",
                (request.image_width, request.image_height),
                bytes(request.image_rgba_bytes),
            )

            if img is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Could not decode image")
                return pb.ResponsePayload(objects=[])

            results = model.predict(
                img,
                conf=request.confidence_threshold,
                iou=request.iou_threshold,
                verbose=bool(int(self.config.get("Verbose", "0"))),
                save=bool(int(self.config.get("SaveImg", "0"))),
                project="./output",
                name=request.image_name, # this stops subdirectories being created, when save is true
                device=self.config.get("Device", "") # you will want to change this to match your hardware
            )

            # force flattening if output is true
            if bool(int(self.config.get("SaveImg", "0"))):
                from flatten_output_dir import flatten
                from pathlib import Path

                flatten(Path("./output") / "Images")

            objects = results_to_proto_boxes(results[0], pb)
            return objects

        except Exception as e: 
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(e))
            return pb.ResponsePayload(objects=[])

def serve():
    cfg, models = config()
    port = cfg.get("Port", "50051")
    host = cfg.get("Host", "[::]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(cfg.get("MaxWorkers", "1"))))
    pbgrpc.add_DetectObjectServicer_to_server(DetectObjectService(), server)
    server.add_insecure_port(f"{host}:{str(port)}") # do we need any kind of secure transport if its inside a secure network?
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()