import socket
import configparser
import struct
import threading
from src.utils.response import Response

# first message to the client is going to be a header thats 8 in length
FORMAT = 'utf-8'
DISCONNECT_MESSAGE="!DISCONNECT"
PACKET_HEADER_FORMAT = '<IBBBB'
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)
IMAGE_MSG = 1
IMAGE_MSG_HEADER_FORMAT = '<iiffIBB'
IMAGE_MSG_HEADER_SIZE = struct.calcsize(IMAGE_MSG_HEADER_FORMAT)

class TCPListen:
    def __init__(self, cfg):
        self.config = cfg
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def listen(self):
        try:

            # binding to localhost is supposedly faster as it bypasses some platform networking code.
            # however, to be seen externally, we need to bind to host: 0.0.0.0
            host = self.config.get("Host", "[::]")
            port = self.config.get("Port", "8089")

            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((host, int(port)))
            self.sock.listen(5) # queue up 5 connect requests max. this is standard max, we shouldn't need more than this, probably less
       
            print(f"listening on {host, port}")

            while True:
                self.accept_wrapper(self.sock)

        finally:
            self.sock.close()

    # returns None if failed to read full length
    def recv_all(self, conn, length):
        data = bytearray(length)
        view = memoryview(data)
        bytes_received = 0
        while bytes_received < length:
            n = conn.recv_into(view[bytes_received:], length - bytes_received)
            if n == 0:  # connection closed early
                return None
            bytes_received += n
        return bytes(data)  # full buffer guaranteed

    def unpack_header(self, conn, addr):
        header = self.recv_all(conn, PACKET_HEADER_SIZE) # blocking
        if header is None:
            print(f"Failed to read header from {addr}")
            return False
        return struct.unpack(PACKET_HEADER_FORMAT, header)
        
    def unpack_image(self, conn, addr, packet_length):
        body_length = packet_length - PACKET_HEADER_SIZE
        if body_length < IMAGE_MSG_HEADER_SIZE:
            print(f"IMAGE packet is smaller than image header from {addr}")
            return False

        body = self.recv_all(conn, body_length)
        if body is None:
            print(f"Failed to read full IMAGE body from {addr}")
            return False

        (
            image_width, image_height, confidence_threshold, iou_threshold,
            image_length, image_name_length, model_name_length
        ) = struct.unpack_from(IMAGE_MSG_HEADER_FORMAT, body, 0)

        offset = IMAGE_MSG_HEADER_SIZE
        image = body[offset:offset+image_length]
        offset += image_length
        image_name = body[offset:offset+image_name_length].decode(FORMAT)
        offset += image_name_length
        model_name = body[offset:offset+model_name_length].decode(FORMAT)
        offset += model_name_length

        return image_width, image_height, confidence_threshold, iou_threshold, image_length, image_name, model_name, image

    def handle_client(self, conn, addr):
        try:
            # this will run concurrently
            print(f"New connection {addr} connected")
    

            while True:
                header = self.unpack_header(conn, addr)
                if not header:
                    break

                packet_length, msg_type, _, _, _ = header            

                if msg_type == IMAGE_MSG:
                    image_unpacked = self.unpack_image(conn, addr, packet_length)
                    if not image_unpacked:
                        break

                    # TODO: process image here

                    # TODO: send response here
                    response = Response()
                    responsePacket = response.build_detections_packet(
                        640, 640, 3, "test", "test", [
                            (50.0, 40.0, 180.0, 160.0, 0.92, 1)
                        ]
                    )

                    conn.sendall(responsePacket)


                else:
                    # default case. Read the full packet and ignore it.
                    print(f"Received message type {msg_type} not handled")
                    body_length = packet_length - PACKET_HEADER_SIZE
                    body = self.recv_all(conn, body_length)
                    if body is None:
                        print(f"Failed to read full body from {addr}")
                        break
                    continue


        except Exception as e:
            raise Exception(f"something went wrong {e}")
        finally:
            conn.close()

    def accept_wrapper(self, sock):
        conn, addr = sock.accept() # blocking
        print(f"Accepted connection from {addr}")

        thread = threading.Thread(target=self.handle_client, args=(conn, addr))
        thread.start()

        print(f"Active threads {threading.active_count() - 1}")