"""
diagnose_images.py
===================
Quick standalone diagnostic — prints the shape, dtype, and channel count of
every image in data/dataset/, so you can see exactly which images are
RGBA/grayscale/16-bit/corrupted before running encode_faces.py.

Usage (from project root):
    python ml_pipeline\\diagnose_images.py
"""

import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"


def main():
    for student_folder in sorted(DATASET_DIR.iterdir()):
        if not student_folder.is_dir():
            continue

        print(f"\n{student_folder.name}:")
        for image_path in sorted(student_folder.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"  {image_path.name}: FAILED TO READ (corrupted or unsupported file)")
                continue

            shape = img.shape
            dtype = img.dtype
            channels = shape[2] if len(shape) == 3 else 1

            note = ""
            if channels == 4:
                note = "  <-- RGBA (has alpha channel)"
            elif channels == 1:
                note = "  <-- Grayscale"
            elif dtype != "uint8":
                note = f"  <-- Unusual bit depth ({dtype})"

            print(f"  {image_path.name}: shape={shape}, dtype={dtype}, channels={channels}{note}")


if __name__ == "__main__":
    main()