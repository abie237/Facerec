"""
encode_faces.py
================
Step 1 of the ML pipeline.

Walks through data/dataset/<student_name>/*.jpg, detects the face in each
image using face_recognition (built on dlib's CNN/HOG detector), computes
a 128-dimensional face encoding for every image, then AVERAGES the
encodings per student into a single representative vector.

Why average instead of keeping all 2-3 encodings separately?
  - Keeps one row per student in the `students` table (clean schema, fast
    lookup at runtime: 1 comparison per student instead of 2-3).
  - Reduces noise from lighting/angle differences across the few photos.
  - Standard, defensible approach for small per-class sample sizes.

Outputs:
  1. data/encodings/encodings.pkl   -> dict {name: averaged_128d_vector}
                                       AND raw per-image encodings/labels
                                       (kept as a backup file).
  2. database/attendance.db         -> populates/refreshes:
       - the `students` table (name + AVERAGED encoding, used for fast
         single-vector lookup at recognition time)
       - the `face_encodings` table (ONE row per training image, used by
         train_classifier.py to build the train/test split — a single
         averaged vector per student cannot be split into train/test).

IMPORTANT: train_classifier.py trains on the RAW per-image encodings
stored in face_encodings (more data points = real train/test split),
NOT the averaged one. The averaged vector stored in `students` is used
only for fast nearest-neighbour lookup / display purposes.
"""

import pickle
import sqlite3
from pathlib import Path

import cv2
import face_recognition
import numpy as np

# ------------------------------------------------------------------
# Paths (relative to project root — run this script from project root,
# e.g. `python ml_pipeline/encode_faces.py`)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
ENCODINGS_DIR = PROJECT_ROOT / "data" / "encodings"
DB_PATH = PROJECT_ROOT / "database" / "attendance.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "db_schema.sql"

ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)

# face_recognition detection model:
# "hog" -> CPU-friendly, faster, slightly less accurate (good for laptops)
# "cnn" -> more accurate, needs GPU/dlib CUDA build, much slower on CPU
DETECTION_MODEL = "hog"


def ensure_database():
    """Create the database file from schema if it doesn't already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def load_image_bgr_to_rgb(image_path: str) -> np.ndarray:
    """
    Read image with OpenCV and convert to a clean 8-bit, 3-channel RGB array,
    since face_recognition (dlib) requires exactly that format and will raise
    "Unsupported image type, must be 8bit gray or RGB image" on anything else
    (e.g. RGBA PNGs, grayscale images, 16-bit images, non-contiguous arrays).

    cv2.IMREAD_COLOR forces OpenCV to always decode to 3-channel BGR
    regardless of the source file's original format (RGBA, grayscale, etc.),
    which is what makes this robust to "weird" input images.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # Force 8-bit depth (handles 16-bit PNGs / unusual bit depths)
    if image_bgr.dtype != np.uint8:
        image_bgr = cv2.normalize(image_bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # dlib's C++ backend requires a C-contiguous array; cv2.cvtColor usually
    # produces one, but enforce it explicitly to be safe.
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    return image_rgb


def encode_dataset():
    """
    Walk data/dataset/<student_name>/*.jpg, detect + encode every face.

    Returns:
        raw_encodings: list of 128-d np.ndarray (one per successfully
                        encoded image — this is the real training data)
        raw_labels:    list of str, parallel to raw_encodings
        raw_filenames: list of str, parallel to raw_encodings (e.g. "1.jpg")
                        — kept for traceability in the face_encodings table
        averaged_encodings: dict {student_name: averaged 128-d np.ndarray}
    """
    raw_encodings = []
    raw_labels = []
    raw_filenames = []

    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        raise FileNotFoundError(
            f"No student folders found in {DATASET_DIR}. "
            f"Add folders like data/dataset/John_Doe/1.jpg before running this script."
        )

    student_folders = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])

    for student_folder in student_folders:
        student_name = student_folder.name
        image_paths = sorted(
            [p for p in student_folder.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )

        if len(image_paths) == 0:
            print(f"[WARNING] No images found for '{student_name}', skipping.")
            continue

        print(f"\n[INFO] Processing '{student_name}' ({len(image_paths)} images)...")

        encodings_for_student = []

        for image_path in image_paths:
            try:
                rgb_image = load_image_bgr_to_rgb(str(image_path))
            except ValueError as e:
                print(f"  [ERROR] {e}")
                continue

            # Detect face bounding boxes first (face_locations), then encode.
            # Doing this in two steps lets us warn clearly when detection fails,
            # rather than face_recognition silently returning an empty list.
            face_locations = face_recognition.face_locations(
                rgb_image, model=DETECTION_MODEL
            )

            if len(face_locations) == 0:
                print(f"  [WARNING] No face detected in {image_path.name}, skipping.")
                continue

            if len(face_locations) > 1:
                print(
                    f"  [WARNING] {len(face_locations)} faces detected in "
                    f"{image_path.name} (expected 1). Using the first/largest face."
                )
                # Keep the largest detected face box (likely the intended subject)
                face_locations = [
                    max(
                        face_locations,
                        key=lambda box: (box[2] - box[0]) * (box[1] - box[3]),
                    )
                ]

            face_encodings = face_recognition.face_encodings(
                rgb_image, known_face_locations=face_locations
            )

            if len(face_encodings) == 0:
                print(f"  [WARNING] Encoding failed for {image_path.name}, skipping.")
                continue

            encoding = face_encodings[0]
            encodings_for_student.append(encoding)
            raw_encodings.append(encoding)
            raw_labels.append(student_name)
            raw_filenames.append(image_path.name)
            print(f"  [OK] Encoded {image_path.name}")

        if len(encodings_for_student) == 0:
            print(f"  [ERROR] No usable encodings for '{student_name}'. "
                  f"This student will NOT be registered.")

    if len(raw_encodings) == 0:
        raise RuntimeError(
            "No faces were successfully encoded across the entire dataset. "
            "Check image quality/paths before continuing."
        )

    # Compute the per-student averaged encoding (for DB storage / display)
    averaged_encodings = {}
    unique_names = sorted(set(raw_labels))
    for name in unique_names:
        vectors = [enc for enc, label in zip(raw_encodings, raw_labels) if label == name]
        averaged_encodings[name] = np.mean(vectors, axis=0)

    return raw_encodings, raw_labels, raw_filenames, averaged_encodings


def save_encodings_pickle(raw_encodings, raw_labels, averaged_encodings):
    """Persist both raw (for training) and averaged (for DB) encodings to disk
    as a backup file, in addition to the database (which is the source of
    truth used by train_classifier.py)."""
    output_path = ENCODINGS_DIR / "encodings.pkl"
    data = {
        "raw_encodings": raw_encodings,      # list[np.ndarray], training data (X)
        "raw_labels": raw_labels,            # list[str], training labels (y)
        "averaged_encodings": averaged_encodings,  # dict[name -> np.ndarray]
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n[INFO] Saved backup encodings to {output_path}")
    return output_path


def update_students_table(conn, averaged_encodings):
    """
    Insert or update each student's row in the `students` table with their
    averaged 128-d encoding (stored as a pickled BLOB).

    Returns a dict {student_name: student_id} for use when populating
    face_encodings (which needs the foreign key student_id).
    """
    cursor = conn.cursor()
    name_to_id = {}

    for student_name, encoding in averaged_encodings.items():
        encoding_blob = pickle.dumps(encoding)

        cursor.execute(
            "SELECT student_id FROM students WHERE student_name = ?",
            (student_name,),
        )
        existing = cursor.fetchone()

        if existing:
            student_id = existing[0]
            cursor.execute(
                "UPDATE students SET face_encoding = ? WHERE student_name = ?",
                (encoding_blob, student_name),
            )
            print(f"[DB] Updated encoding for existing student '{student_name}'")
        else:
            cursor.execute(
                "INSERT INTO students (student_name, face_encoding) VALUES (?, ?)",
                (student_name, encoding_blob),
            )
            student_id = cursor.lastrowid
            print(f"[DB] Registered new student '{student_name}'")

        name_to_id[student_name] = student_id

    conn.commit()
    return name_to_id


def update_face_encodings_table(conn, name_to_id, per_image_records):
    """
    Populate the face_encodings table with ONE row per training image.

    This is the data train_classifier.py reads for the train/test split
    (multiple samples per student), as opposed to the single averaged
    vector stored in `students`.

    per_image_records: list of dicts {"name": str, "filename": str,
                                       "encoding": np.ndarray}

    Existing rows for these students are cleared first, so re-running
    encode_faces.py after adding/removing photos doesn't leave stale
    encodings behind for deleted images.
    """
    cursor = conn.cursor()

    affected_student_ids = {name_to_id[name] for name in {r["name"] for r in per_image_records}}
    for student_id in affected_student_ids:
        cursor.execute("DELETE FROM face_encodings WHERE student_id = ?", (student_id,))

    for record in per_image_records:
        student_id = name_to_id[record["name"]]
        encoding_blob = pickle.dumps(record["encoding"])
        cursor.execute(
            "INSERT INTO face_encodings (student_id, image_filename, face_encoding) "
            "VALUES (?, ?, ?)",
            (student_id, record["filename"], encoding_blob),
        )

    conn.commit()
    print(f"[DB] Stored {len(per_image_records)} per-image encoding(s) "
          f"in face_encodings table across {len(affected_student_ids)} student(s).")


def main():
    print("=" * 60)
    print("FACE ENCODING PIPELINE — Step 1: Dataset Encoding")
    print("=" * 60)

    raw_encodings, raw_labels, raw_filenames, averaged_encodings = encode_dataset()

    print(f"\n[SUMMARY] Encoded {len(raw_encodings)} images "
          f"across {len(averaged_encodings)} students.")
    for name in averaged_encodings:
        count = raw_labels.count(name)
        print(f"  - {name}: {count} image(s) encoded")

    save_encodings_pickle(raw_encodings, raw_labels, averaged_encodings)

    conn = ensure_database()
    name_to_id = update_students_table(conn, averaged_encodings)

    per_image_records = [
        {"name": name, "filename": filename, "encoding": encoding}
        for encoding, name, filename in zip(raw_encodings, raw_labels, raw_filenames)
    ]
    update_face_encodings_table(conn, name_to_id, per_image_records)

    conn.close()

    print("\n[DONE] Encoding complete. Run train_classifier.py next.")


if __name__ == "__main__":
    main()