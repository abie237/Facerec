"""
diagnose_distance.py
=====================
Captures ONE frame from the webcam, encodes the detected face, then prints
the EUCLIDEAN DISTANCE from that live encoding to EVERY individual training
image encoding stored in the database (face_encodings table) -- not just
the SVM's classification, but the raw geometric distance in face space.

This answers a more fundamental question than the SVM probabilities do:
is your live face encoding actually closer to YOUR OWN training photos,
or genuinely closer to someone else's? If it's closer to your own photos,
the problem is in how the SVM is drawing its decision boundary (fixable
via more data / different model). If it's closer to someone else's
photos, the problem is in the training images themselves (lighting/
quality issue) or the photos are mislabeled.

Lower distance = more similar. face_recognition's own documentation
treats anything below ~0.6 as "likely the same person" as a rule of
thumb, though this is approximate.

Press SPACE to capture and analyze a frame, 'q' to quit.

Usage (from project root):
    python ml_pipeline\\diagnose_distance.py
"""

import pickle
import sqlite3
from pathlib import Path

import cv2
import face_recognition
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "attendance.db"


def load_all_training_encodings():
    """Returns list of (student_name, image_filename, encoding) for every
    row in face_encodings, joined with students for the name."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.student_name, f.image_filename, f.face_encoding
        FROM face_encodings f
        JOIN students s ON f.student_id = s.student_id
        ORDER BY s.student_name, f.image_filename
        """
    )
    rows = cursor.fetchall()
    conn.close()

    return [(name, filename, pickle.loads(blob)) for name, filename, blob in rows]


def analyze_frame(frame, training_data):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    if not face_locations:
        print("[INFO] No face detected in this frame. Try again.")
        return

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    live_encoding = face_encodings[0]

    distances = []
    for name, filename, train_encoding in training_data:
        dist = np.linalg.norm(live_encoding - train_encoding)
        distances.append((dist, name, filename))

    distances.sort(key=lambda x: x[0])

    print(f"\n{'=' * 60}")
    print("DISTANCE FROM LIVE FACE TO EVERY TRAINING IMAGE (closest first)")
    print(f"{'=' * 60}")
    print(f"{'Distance':<12}{'Student':<20}{'Image'}")
    for dist, name, filename in distances:
        marker = "  <-- CLOSEST MATCH" if dist == distances[0][0] else ""
        print(f"{dist:<12.4f}{name:<20}{filename}{marker}")

    print(f"\n[INTERPRETATION] Closest training image overall belongs to: "
          f"'{distances[0][1]}' (distance {distances[0][0]:.4f})")
    print("Rule of thumb: distances below ~0.6 suggest the same person; "
          "above ~0.6 suggests a different person, but this is approximate.")


def main():
    training_data = load_all_training_encodings()
    print(f"[INFO] Loaded {len(training_data)} training image encodings "
          f"across {len(set(t[0] for t in training_data))} students.\n")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press SPACE to capture and analyze the current frame.")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        cv2.imshow("Diagnose Distance - press SPACE to analyze, q to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            analyze_frame(frame, training_data)
        elif key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()