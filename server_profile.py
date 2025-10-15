import logging
import signal
import sys
import atexit
import yappi

from server import serve  # your gRPC server starter

OUTPUT = "server_profile.prof"

def dump_stats():
    try:
        # Safe to call even if already stopped
        yappi.stop()
        yappi.get_func_stats().save(OUTPUT, type="pstat")
        print(f"[yappi] Saved {OUTPUT}")
    except Exception as e:
        print(f"[yappi] Failed to save profile: {e}", file=sys.stderr)

def handle_signal(signum, frame):
    print(f"[yappi] Caught signal {signum}, saving profile...")
    dump_stats()
    # Exit cleanly so atexit doesn't run dump_stats twice
    sys.exit(0)

if __name__ == "__main__":
    logging.basicConfig()

    # Start profiling
    yappi.set_clock_type("wall")  # or "cpu"
    yappi.start()

    # Save on normal interpreter shutdown
    atexit.register(dump_stats)

    # Save on Ctrl+C and TERM
    signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except AttributeError:
        # SIGTERM may not exist on some platforms (e.g., older Windows)
        pass

    try:
        serve()  # blocks
    finally:
        # Runs on KeyboardInterrupt and most other exceptions
        dump_stats()
