import socket
import configparser
import threading

# first message to the client is going to be a header thats 64 in length
CONTENT_LENGTH_HEADER = 64 # a number that represents the length of content being recieved
FORMAT = 'utf-8'
DISCONNECT_MESSAGE="!DISCONNECT"

class TCPListen:
    def __init__(self, cfg):
        self.config = cfg
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       
    def listen(self):
        # binding to localhost is supposedly faster as it bypasses some platform networking code.
        # however, to be seen externally, we need to bind to host: 0.0.0.0
        host = self.config.get("Host", "[::]")
        port = self.config.get("Port", "8089")

        self.sock.bind((host, int(port)))
        self.sock.listen(5) # queue up 5 connect requests max. this is standard max, we shouldn't need more than this, probably less
        
        print(f"listening on {host, port}")

        while True:
            self.accept_wrapper(self.sock)

    def handle_client(self, conn, addr):
        try:
            # this will run concurrently
            print(f"New connection {addr} connected")

            connected = True
            while connected:
                msg_length = conn.recv(CONTENT_LENGTH_HEADER).decode(FORMAT) # blocking
                
                if msg_length:
                    msg_length = int(msg_length)
                    msg = conn.recv(msg_length).decode(FORMAT) # blocking

                    if msg == DISCONNECT_MESSAGE:
                        connected = False

                    print(f"[{addr}]: message length {msg_length}, {msg}")
                
        except Exception as e:
            raise Exception(f"something went wrong {e}")
        finally:
            conn.close()            

    def accept_wrapper(self, sock):
        conn, addr = sock.accept() # blocking
        print(f"Accepted connection from {addr}")

        thread = threading.Thread(target=self.handle_client, args=(conn, addr))
        thread.start()

        print(f"Active threads {threading.activeCount() - 1}")