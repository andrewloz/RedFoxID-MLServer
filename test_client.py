import socket
from src.config import Config
import time

CONTENT_LENGTH_HEADER = 64 # a number that represents the length of content being recieved
FORMAT = 'utf-8'
DISCONNECT_MESSAGE="!DISCONNECT"


def send(msg, client):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)

    # now we pad send length to make sure its 64 bytes long
    send_length += b' ' * (CONTENT_LENGTH_HEADER - len(send_length))

    client.send(send_length)
    client.send(message)

if __name__ == "__main__":
    cfg, models = Config("config.ini").getAll()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        HOST = cfg.get("Host", "[::]")
        PORT = cfg.get("Port", "8089")

        connection_time = time.time()
        s.connect((HOST, int(PORT)))
        print(f"Took {round((time.time() - connection_time) * 1000, 3)}ms to connect")


        amount = 0
        while amount < 50:
            message_start = time.time()
            send("Hello!", s)
            send(DISCONNECT_MESSAGE, s)
            print(f"Took {round((time.time() - message_start) * 1000, 3)}ms to send message")

            amount += 1 

        # s.sendall(b"Hello, world")
        # data = s.recv(1024)


