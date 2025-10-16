from src.utils.tcp import TCPListen
from src.config import Config

if __name__ == "__main__":
    cfg, models = Config("config.ini").getAll()

    tcp = TCPListen(cfg)

    tcp.listen()
