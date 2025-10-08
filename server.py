from ultralytics import YOLO
from concurrent import futures
import logging

import grpc
import numpy as np
import cv2
from PIL import Image

import detect_object_pb2_grpc as pbgrpc
import detect_object_pb2 as pb


from server_utils import results_to_proto_boxes

class DetectObjectService(pbgrpc.DetectObjectServicer):
    def __init__(self, weights="./model/best.onnx"):
            self.model = YOLO(weights, task="detect")
    
    def Request(self, request, context):
        img = Image.frombytes('RGBA', (request.image_width, request.image_height), bytes(request.image_rgba_bytes))

        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Could not decode image")
            return pb.ResponsePayload(objects=[])

        results = self.model.predict(
            img, 
            conf=request.confidence_threshold, 
            iou=request.iou_threshold,
            verbose=False, 
            # save=False, 
            # project="output", 
            device="cuda:0" # you will want to change this to match your hardware
        )
        
        objects = results_to_proto_boxes(results[0], pb)
        return objects

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pbgrpc.add_DetectObjectServicer_to_server(DetectObjectService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()