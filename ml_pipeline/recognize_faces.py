"""
recognize_faces.py
===================
Step 3 of the ML pipeline.

Opens the webcam with OpenCV, detects faces in each frame using
face_recognition, encodes them, and classifies each detected face using
the trained KNN model (knn_model.pkl) from train_classifier.py.

WHY KNN INSTEAD OF SVM:
  Diagnostic testing (diagnose_distance.py) confirmed that raw Euclidean
  distance between face encodings correctly separates registered students
  -- i.e. the embeddings themselves are discriminative. However, the
  linear SVM trained on only 2-3 images per student misclassified live
  faces with high confidence (e.g. consistently predicting the wrong
  student). This is a known failure mode of SVMs with very few training
  examples per class: there isn't enough data to fit a reliable decision
  boundary, even when the underlying data is well-separated.

  KNN, by contrast, classifies directly by distance to the nearest stored
  examples -- which is exactly the comparison diagnose_distance.py already
  confirmed works correctly for this dataset. This is a deliberate,
  diagnosed choice, not just "whichever scored higher on the test split."

WHY THERE'S ALSO A SEPARATE DISTANCE CHECK (not just confidence):
  KNN always returns the NEAREST registered student, even for a face that
  doesn't match anyone -- it has no built-in concept of "none of the
  above." predict_proba()'s confidence is RELATIVE (how much do nearby
  neighbors agree with each other), not ABSOLUTE (how close is this face
  to anyone at all). A stranger's face can score confidence=1.0 if their
  encoding happens to be closest to one student even by a wide margin.

  To fix this, every prediction is also checked against the ACTUAL
  Euclidean distance to its nearest neighbor (via knn_model.kneighbors()).
  If that absolute distance exceeds MAX_DISTANCE_THRESHOLD, the face is
  rejected as "Unknown" regardless of how confident the relative
  prediction was. This mirrors the same distance rule of thumb already
  used in diagnose_distance.py.

Run this AFTER encode_faces.py and train_classifier.py have been run.

Controls:
    q  -> quit

Usage:
    python ml_pipeline/recognize_faces.py
"""

import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np

# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "ml_pipeline" / "models"

KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Confidence threshold: predictions below this are labeled "Unknown"
# rather than forced into the closest matching class.
#
# KNN with weights="distance" and small k (often 2-3, auto-reduced by
# train_classifier.py for small classes) produces COARSE probability
# values -- e.g. exactly 1.0, 0.667, 0.5, 0.333, 0.0 -- rather than a
# smooth gradient like SVM. A threshold of 0.5 cleanly means "more than
# half of the nearest-neighbor weight agrees on this student," which is
# a meaningful and simple cutoff for this model, unlike SVM where 0.35-0.55
# required delicate tuning.
CONFIDENCE_THRESHOLD = 0.5

# ABSOLUTE distance threshold (separate from the relative confidence
# above). face_recognition's own documentation treats ~0.6 as a rough
# rule of thumb for "same person" vs "different person" on its 128-d
# encodings. A face whose nearest training example is farther than this
# is rejected as Unknown EVEN IF KNN's relative confidence was 1.0,
# since KNN has no concept of "doesn't match anyone" on its own.
#
# Tune this by running diagnose_distance.py on a few known and a few
# unknown faces and looking at where the gap actually falls for YOUR
# camera/lighting setup -- 0.6 is a reasonable starting point, not a
# universal constant.
MAX_DISTANCE_THRESHOLD = 0.6

# face_recognition detection model: "hog" (CPU, fast) or "cnn" (GPU, accurate)
DETECTION_MODEL = "hog"

# Resize frame before detection to speed up processing (1.0 = full size)
FRAME_RESIZE_SCALE = 0.5


def load_model_and_labels():
    if not KNN_MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Trained KNN model or label encoder not found in "
            f"{MODELS_DIR}. Run train_classifier.py first."
        )

    with open(KNN_MODEL_PATH, "rb") as f:
        knn_model = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return knn_model, label_encoder


def recognize_frame(frame, knn_model, label_encoder):
    """
    Detect and classify all faces in a single BGR frame.

    Returns a list of dicts: [{"name": str, "confidence": float,
    "distance": float, "box": (top, right, bottom, left)}, ...]
    """
    # Resize for faster detection, convert BGR -> RGB for face_recognition
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=DETECTION_MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Scale box coordinates back up to the original frame size
        scale = 1.0 / FRAME_RESIZE_SCALE
        box = (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))

        probabilities = knn_model.predict_proba([encoding])[0]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        # ---- ABSOLUTE distance check (separate from relative confidence) ----
        # kneighbors() returns the actual Euclidean distance to the single
        # nearest stored training encoding, regardless of which class it
        # belongs to. This is the check that catches strangers: even if
        # KNN's relative confidence is 1.0, a large nearest-neighbor
        # distance means "nobody in the database actually looks like this."
        nearest_distances, _ = knn_model.kneighbors([encoding], n_neighbors=1)
        nearest_distance = float(nearest_distances[0][0])

        if confidence >= CONFIDENCE_THRESHOLD and nearest_distance <= MAX_DISTANCE_THRESHOLD:
            name = label_encoder.inverse_transform([best_idx])[0]
        else:
            name = "Unknown"

        results.append({
            "name": name,
            "confidence": confidence,
            "distance": nearest_distance,
            "box": box,
        })

    return results


def draw_results(frame, results):
    """Draw bounding boxes and name+confidence+distance labels on the frame."""
    for result in results:
        top, right, bottom, left = result["box"]
        name = result["name"]
        confidence = result["confidence"]
        distance = result["distance"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({confidence:.2f}, d={distance:.2f})"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            frame, label, (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1,
        )

    return frame


def main():
    knn_model, label_encoder = load_model_and_labels()
    print(f"[INFO] Loaded KNN model. Known students: {list(label_encoder.classes_)}")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera connection/permissions.")

    print("[INFO] Webcam started. Press 'q' to quit.")

    recognized_names_this_run = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam.")
            break

        results = recognize_frame(frame, knn_model, label_encoder)
        frame = draw_results(frame, results)

        for result in results:
            if result["name"] != "Unknown" and result["name"] not in recognized_names_this_run:
                recognized_names_this_run.add(result["name"])
                print(f"[RECOGNIZED] {result['name']} (confidence: {result['confidence']:.2f})")

        cv2.putText(
            frame, f"Recognized so far: {len(recognized_names_this_run)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    print(f"\n[SUMMARY] Total unique students recognized this session: "
          f"{len(recognized_names_this_run)}")
    for name in sorted(recognized_names_this_run):
        print(f"  - {name}")


if __name__ == "__main__":
    main()