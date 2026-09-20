"""Package the repository's icon.png as the desktop PNG and macOS ICNS.

Run from any directory with the project Python environment. The supplied source
artwork remains unchanged; ICNS contains the sizes macOS needs for its Dock and
Finder representations.
"""
from pathlib import Path
import shutil

from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "icon.png"
    assets = root / "app/assets"
    with Image.open(source) as image:
        if image.format != "PNG" or image.width != image.height:
            raise ValueError("icon.png must be a square PNG image")
        if image.width < 1024:
            raise ValueError("icon.png must be at least 1024 × 1024 pixels")
        assets.mkdir(parents=True, exist_ok=True)
        image.convert("RGBA").save(assets / "icon.icns", format="ICNS")
    shutil.copyfile(source, assets / "icon.png")
    print("Updated app/assets/icon.png and app/assets/icon.icns from icon.png")


if __name__ == "__main__":
    main()
