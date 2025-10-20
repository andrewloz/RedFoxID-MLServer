from src.utils.tcp import TCPListen
from src.config import Config
from ultralytics import YOLO

import logging
import signal
import sys
import atexit
import yappi

OUTPUT = "server_profile.prof"

def dump_stats():
    try:
        yappi.stop()
        yappi.get_func_stats().save(OUTPUT, type="pstat")
        print(f"[yappi] Saved {OUTPUT}")
    except Exception as e:
        print(f"[yappi] Failed to save profile: {e}", file=sys.stderr)

def handle_signal(signum, frame):
    print(f"[yappi] Caught signal {signum}, saving profile...")
    dump_stats()
    sys.exit(0)

if __name__ == "__main__":
    logging.basicConfig()

    yappi.set_clock_type("wall")  # or "cpu"
    yappi.start()

    atexit.register(dump_stats)

    signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except AttributeError:
        pass

    try:
        cfg, models = Config("config.ini").getAll()

        tcp = TCPListen(cfg)

        # model
        model = YOLO("./model/side_plane_model_v26.engine", task="detect")

        results = model.predict(
            imgsz=(640, 640),
            verbose=True,
            save=False,
            # device=self.config.get("Device", "") # you will want to change this to match your hardware
        )

        tcp.listen(model)
    finally:
        dump_stats()
