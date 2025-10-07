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

# Load model
class DetectObjectService(pbgrpc.DetectObjectServicer):
    def __init__(self, weights="./model/best.onnx"):
            self.model = YOLO(weights, task="detect")
            # self.model = YOLO(weights, task="detect")
            # self.model.to("cuda")
    
    def Request(self, request, context):
        # buf = np.frombuffer(request.image_bytes, dtype=np.uint8)
        # TODO: this requires a better solution, maybe an option in proto to send the format of the payload.
        img = Image.frombytes('RGBA', (request.image_width, request.image_height), bytes(request.image_bytes))
        # img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Could not decode image")
            return pb.ResponsePayload(objects=[])

        results = self.model.predict(img, verbose=True, save=True, project="output", device="cuda:0")
        
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