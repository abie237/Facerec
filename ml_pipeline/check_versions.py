"""
check_versions.py
==================
Prints the exact installed versions of numpy, dlib, and face_recognition.
This confirms or rules out a numpy/dlib ABI mismatch, which is the most
common cause of "Unsupported image type, must be 8bit gray or RGB image"
when the image array itself checks out fine (correct shape/dtype/contiguity).

Usage:
    python ml_pipeline\\check_versions.py
"""

import numpy
import dlib
import face_recognition
import cv2

print(f"numpy version            : {numpy.__version__}")
print(f"dlib version             : {dlib.__version__}")
print(f"face_recognition version : {face_recognition.__version__}")
print(f"opencv version           : {cv2.__version__}")