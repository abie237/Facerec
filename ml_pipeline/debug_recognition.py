"""
debug_recognition.py
=====================
Opens the webcam and prints the FULL probability distribution from the
SVM for every detected face, every ~15 frames -- bypassing the confidence
threshold in recognize_faces.py so you can see what the model is
ACTUALLY predicting, not just whether it cleared the cutoff.

This tells you whether:
  (a) the model predicts your name but with low confidence (a
      threshold/training problem), or
  (b) the model predicts someone else's name entirely (a real
      misclassification / data problem)

Press 'q' to quit.

Usage (from project root):
    python ml_pipeline\\debug_recognition.py
"""

import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "ml_pipeline" / "models"

with open(MODELS_DIR / "svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print(f"Known students: {list(label_encoder.classes_)}\n")

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.\n")

frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    frame_count += 1

    if face_encodings and frame_count % 15 == 0:  # print every ~15 frames, not every frame (too noisy)
        for encoding in face_encodings:
            probabilities = svm_model.predict_proba([encoding])[0]
            print(f"--- Frame {frame_count} ---")
            for idx, name in enumerate(label_encoder.classes_):
                print(f"  {name:<20} {probabilities[idx]:.4f}")
            best_idx = int(np.argmax(probabilities))
            print(f"  >> Best guess: {label_encoder.classes_[best_idx]} "
                  f"({probabilities[best_idx]:.4f})\n")

    # Draw boxes so you can see what's being detected
    for (top, right, bottom, left) in face_locations:
        scale = 2
        cv2.rectangle(frame, (left * scale, top * scale), (right * scale, bottom * scale), (0, 255, 0), 2)

    cv2.imshow("Debug Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()