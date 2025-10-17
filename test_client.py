import socket
from src.config import Config
import time
import struct
from pathlib import Path
import time 


CONTENT_LENGTH_HEADER = 64 # a number that represents the length of content being recieved
FORMAT = 'utf-8'
DISCONNECT_MESSAGE="!DISCONNECT"

PACKET_HEADER_FORMAT = '<IBBBB'
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)

IMAGE_MSG = 1
IMAGE_MSG_HEADER_FORMAT = '<iiffIBB'

def build_image_packet(img_bytes: bytes, image_name: str, model_name: str, width: int, height: int, conf: float, iou: float) -> bytes:
    name_b = image_name.encode('utf-8')
    model_b = model_name.encode('utf-8')
    if len(name_b) > 255 or len(model_b) > 255:
        raise ValueError('image_name and model_name must be <= 255 bytes')

    # print(f"build: {image_name} {model_name} {width} {height} {conf} {iou}")
    
    # build image packet which is img_metadata followed by img_bytes then name_b(ytes) and model bytes
    img_meta = struct.pack(IMAGE_MSG_HEADER_FORMAT, int(width), int(height), int(conf), int(iou), len(img_bytes), len(name_b), len(model_b))
    body = img_meta + img_bytes + name_b + model_b

    # now get the length of the full body of the packet
    packet_length = PACKET_HEADER_SIZE + len(body)

    # now build header
    header = struct.pack(PACKET_HEADER_FORMAT, packet_length, IMAGE_MSG, 0, 0, 0)

    # add the header to the start of the packet and body after, as we read the header to understand what we need to read from the body.
    return header + body


def send(image_path: str, model_name: str, times: int = 50, conf: float = 0.5, iou: float = 0.5):
    cfg, models = Config("config.ini").getAll()
    host = cfg.get("Host", "[::]")
    port = cfg.get("Port", "8089")

    img_path = Path(image_path)
    img_bytes = img_path.read_bytes()
    width = 640
    height = 640

    packet = build_image_packet(img_bytes, img_path.name, model_name, width, height, conf, iou)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, int(port)))

        for _ in range(times):
            start = time.time()
            s.sendall(packet)
            print(f"time taken: {round((time.time() - start) * 1000, 4)}ms")


if __name__ == "__main__":
    send("./input/test.png", "test")

