import os
from pathlib import Path

# Ensure Ultralytics writes its settings inside the repository rather than $HOME.
CONFIG_ROOT = Path(__file__).resolve().parent / ".config"
os.environ.setdefault("XDG_CONFIG_HOME", str(CONFIG_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT / "Ultralytics"))
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
(CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)

from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info, select_device, torch  # noqa: E402


def main() -> None:
    """Print the compute targets visible to Ultralytics."""
    default_device = select_device("", verbose=False)
    print(f"Ultralytics default device: {default_device}")
    print(f"CPU: {get_cpu_info()}")

    if torch.cuda.is_available():
        print("CUDA devices:")
        for idx in range(torch.cuda.device_count()):
            print(f" - cuda:{idx} -> {get_gpu_info(idx)}")
    else:
        print("No CUDA devices detected.")

    if torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders (MPS) backend is available.")


if __name__ == "__main__":
    main()
