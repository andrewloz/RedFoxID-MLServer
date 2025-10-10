from __future__ import annotations

from pathlib import Path

# Because the YOLO package is annoying, and not intended for production use. 
# it puts detections in sub directories, which is really annoying, so we flattern that dir with this helper function.
def flatten(output_dir: Path) -> None:
	if not output_dir.exists():
		return
	for folder in output_dir.iterdir():
		if not folder.is_dir():
			continue
		src = folder / "image0.jpg"
		if src.exists():
			dst = output_dir / f"{folder.name}.jpg"
			src.rename(dst)
			try:
				folder.rmdir()
			except OSError:
				pass


if __name__ == "__main__":
	here = Path(__file__).resolve().parent
	flatten(here / "output" / "Images")
