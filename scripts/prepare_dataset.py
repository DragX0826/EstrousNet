from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "data"
    raw_dir = src_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    copied = 0
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix.lower() in image_ext:
            shutil.copy2(path, raw_dir / path.name)
            copied += 1

    print(f"Copied {copied} image(s) into {raw_dir}")


if __name__ == "__main__":
    main()

