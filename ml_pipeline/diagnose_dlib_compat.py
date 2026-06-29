"""
diagnose_dlib_compat.py
========================
Deeper diagnostic for the specific "Unsupported image type, must be 8bit
gray or RGB image" dlib error -- checks things diagnose_images.py can't see:
array contiguity, strides, and actually attempts the real face_recognition
call on each image individually so we see EXACTLY which image and which
step fails.

Usage (from project root):
    python ml_pipeline\\diagnose_dlib_compat.py
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"


def check_image(image_path: Path):
    print(f"\n--- {image_path} ---")

    # Step 1: raw OpenCV read
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("  FAILED at cv2.imread - file unreadable")
        return
    print(f"  cv2.imread OK: shape={img_bgr.shape}, dtype={img_bgr.dtype}, "
          f"C-contiguous={img_bgr.flags['C_CONTIGUOUS']}")

    # Step 2: BGR -> RGB conversion
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"  cv2.cvtColor OK: shape={img_rgb.shape}, dtype={img_rgb.dtype}, "
          f"C-contiguous={img_rgb.flags['C_CONTIGUOUS']}, "
          f"strides={img_rgb.strides}")

    # Step 3: force contiguous copy explicitly
    img_rgb_contig = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    print(f"  np.ascontiguousarray: C-contiguous={img_rgb_contig.flags['C_CONTIGUOUS']}")

    # Step 4: try face_recognition on the RAW converted array (no fix)
    try:
        locations = face_recognition.face_locations(img_rgb, model="hog")
        print(f"  face_recognition on RAW array: SUCCESS, found {len(locations)} face(s)")
    except RuntimeError as e:
        print(f"  face_recognition on RAW array: FAILED -> {e}")

    # Step 5: try face_recognition on the explicitly contiguous array
    try:
        locations = face_recognition.face_locations(img_rgb_contig, model="hog")
        print(f"  face_recognition on CONTIGUOUS array: SUCCESS, found {len(locations)} face(s)")
    except RuntimeError as e:
        print(f"  face_recognition on CONTIGUOUS array: FAILED -> {e}")


def main():
    # Just test the abiel folder since that's where it's failing
    target_folder = DATASET_DIR / "abiel"
    for image_path in sorted(target_folder.iterdir()):
        if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            check_image(image_path)


if __name__ == "__main__":
    main()