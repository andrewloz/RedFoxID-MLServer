from __future__ import print_function

import logging
import json
import statistics

import grpc
import detect_object_pb2_grpc
import detect_object_pb2
import os
import time

from PIL import Image
from io import BytesIO


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = detect_object_pb2_grpc.DetectObjectStub(channel)

        latencies_ms = []
        results_ms = []

        for f in os.listdir("./input"):
            img = Image.open(os.path.join("./input", f))
            buf = BytesIO()
            img.save(buf, format="PNG") 
            img_bytes = buf.getvalue()  

            call_started = time.perf_counter()

            response = stub.Request(detect_object_pb2.RequestPayload(image_bytes=img_bytes))

            elapsed_ms = (time.perf_counter() - call_started) * 1000
            latencies_ms.append(elapsed_ms)

            for obj in response.objects:
                results_ms.append(obj.score)
            


        if results_ms:
            avg_score = statistics.mean(results_ms)
            max_score = max(results_ms)
            min_score = min(results_ms)
            median_score = statistics.median(results_ms)
            print("\nScore summary:")
            print(f"  Detections measured: {len(results_ms)}")
            print(f"  Average score: {avg_score:.4f}")
            print(f"  Median score: {median_score:.4f}")
            print(f"  Min score: {min_score:.4f}")
            print(f"  Max score: {max_score:.4f}")

        if latencies_ms:
            measured = latencies_ms[1:] if len(latencies_ms) > 1 else []
            if measured:
                avg_ms = statistics.mean(measured)
                longest_ms = max(measured)
                shortest_ms = min(measured)
                median_ms = statistics.median(measured)
                print("\nLatency summary (excluding warm-up):")
                print(f"  Requests measured: {len(measured)}")
                print(f"  Average: {avg_ms:.2f} ms")
                print(f"  Median: {median_ms:.2f} ms")
                print(f"  Shortest: {shortest_ms:.2f} ms")
                print(f"  Longest: {longest_ms:.2f} ms")
            else:
                print("\nLatency summary: only warm-up request was recorded; no statistics computed.")



if __name__ == "__main__":
    logging.basicConfig()
    run()