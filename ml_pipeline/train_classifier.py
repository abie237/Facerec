"""
train_classifier.py
====================
Step 2 of the ML pipeline.

Loads per-image face encodings DIRECTLY FROM THE SQLITE DATABASE
(face_encodings table — populated by encode_faces.py), performs a
stratified train/test split, establishes a baseline, then trains TWO
classifiers on the identical split so they can be fairly compared:

  1. SVM  (Support Vector Machine, linear kernel, probability=True)
  2. KNN  (K-Nearest Neighbors)

Why read from the database instead of a pickle file?
  The `students` table stores only ONE averaged encoding per student
  (used for fast lookup at recognition time). That single vector per
  class cannot be split into train/test. The `face_encodings` table
  stores the RAW per-image encoding for every training photo, with a
  foreign key back to `students` — this is the actual training data
  (X = encodings, y = student_name), and it lives in the database
  itself rather than a side-channel file, so the database is the
  single source of truth for the project.

Why a baseline?
  Before trusting any classifier's accuracy, we need a naive reference
  point: a "majority-class" baseline classifier that always predicts
  the most frequent student. Any real classifier should clearly beat
  this baseline; if it doesn't, the model has learned nothing useful.

Why train two classifiers and compare them (Path B)?
  Rather than relying purely on raw Euclidean-distance matching against
  stored encodings (Path A — simple nearest-neighbour lookup with no
  learned decision boundary), training real classifiers on the 128-d
  embedding space lets the model learn a discriminative boundary between
  students, and gives a quantifiable, defensible accuracy figure for the
  thesis (precision/recall/F1, not just "looks about right").

IMPORTANT — small dataset caveat (document this in your thesis limitations):
  With only a handful of registered students and 2-3 images each, the
  test set will be VERY small (e.g. 1 image per class held out). Metrics
  on such a small test set are high-variance — a single misclassified
  image can swing "accuracy" by 10-30%. The pipeline below is
  methodologically correct regardless of dataset size; the SIZE of the
  dataset is the limitation, not the method.

Outputs:
  - ml_pipeline/models/svm_model.pkl
  - ml_pipeline/models/knn_model.pkl
  - ml_pipeline/models/label_encoder.pkl   (shared by both models)
  - ml_pipeline/models/evaluation_results.pkl (metrics, for the notebook)
"""

import pickle
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "attendance.db"
MODELS_DIR = PROJECT_ROOT / "ml_pipeline" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42        # fixed seed -> reproducible split, required for thesis repeatability
TEST_SIZE = 0.25         # 25% held out for testing


# ------------------------------------------------------------------
# 1. LOAD DATA FROM THE DATABASE
# ------------------------------------------------------------------
def load_training_data_from_db():
    """
    Reads every row from face_encodings, joined with students to get the
    student_name label, and unpickles each stored 128-d encoding.

    Returns:
        X: np.ndarray of shape (n_images, 128)
        y: np.ndarray of shape (n_images,) — student_name strings
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run database/init_db.py "
            f"and ml_pipeline/encode_faces.py first."
        )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.student_name, f.face_encoding
        FROM face_encodings f
        JOIN students s ON f.student_id = s.student_id
        ORDER BY s.student_name
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(
            "No rows found in face_encodings. Run ml_pipeline/encode_faces.py "
            "first to populate the database."
        )

    labels = []
    encodings = []
    for student_name, encoding_blob in rows:
        encoding = pickle.loads(encoding_blob)
        encodings.append(encoding)
        labels.append(student_name)

    X = np.array(encodings)
    y = np.array(labels)

    print(f"[INFO] Loaded {X.shape[0]} per-image encodings from the database "
          f"across {len(set(y))} students.")
    return X, y


def check_minimum_samples(y):
    """
    Warn (don't crash) if any class has too few samples to stratify properly.
    With < 2 samples for a class, sklearn's stratified split will fail outright.
    """
    counts = Counter(y)
    print("\n[INFO] Samples per student (from face_encodings table):")
    for name, count in sorted(counts.items()):
        print(f"  - {name}: {count}")

    min_count = min(counts.values())
    if min_count < 2:
        raise ValueError(
            f"At least one student has only {min_count} image(s) encoded. "
            f"Each student needs at least 2 images for a train/test split "
            f"(1 to train, 1 to test). Add more images and re-run encode_faces.py."
        )
    if min_count < 3:
        print(
            "\n[WARNING] At least one student has fewer than 3 images. "
            "The test set for that class will contain only 1 image — "
            "expect noisy, high-variance metrics. Document this as a "
            "dataset-size limitation in the thesis."
        )


# ------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------------------------------
def split_data(X, y):
    """
    Stratified train/test split — stratify=y ensures each student is
    proportionally represented in both train and test sets (critical with
    small per-class counts, otherwise a class could vanish entirely from
    one side of the split).

    With very small datasets, a fixed 25% test_size can round down to
    fewer test samples than there are classes (e.g. 12 images / 4 students
    -> 3 test samples for 4 classes), which sklearn rejects outright since
    stratification requires at least 1 test sample per class. In that case
    we raise test_size to the minimum that still satisfies stratification.
    """
    n_classes = len(set(y))
    n_samples = len(y)

    effective_test_size = TEST_SIZE
    min_test_size_for_stratify = n_classes / n_samples

    if effective_test_size < min_test_size_for_stratify:
        print(
            f"\n[INFO] Dataset is small ({n_samples} images, {n_classes} students). "
            f"Raising test_size from {TEST_SIZE:.2f} to {min_test_size_for_stratify:.2f} "
            f"so every student has at least 1 test sample."
        )
        effective_test_size = min_test_size_for_stratify

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=effective_test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[INFO] Train/test split: {len(X_train)} train / {len(X_test)} test "
          f"({len(X_train) / n_samples * 100:.0f}/{len(X_test) / n_samples * 100:.0f} split)")
    return X_train, X_test, y_train, y_test


def encode_labels(y_train, y_test):
    """
    Convert string names to integer class labels (required by sklearn).
    The SAME LabelEncoder must be reused at inference time (recognition),
    so it's saved to disk alongside the models.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_test]))  # fit on full label set
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    return y_train_enc, y_test_enc, label_encoder


# ------------------------------------------------------------------
# 3. BASELINE
# ------------------------------------------------------------------
def train_baseline(X_train, y_train):
    """
    Naive reference model: always predicts the most frequent class in the
    training set, ignoring the input encoding entirely. Any real classifier
    (SVM/KNN) should clearly outperform this — if it doesn't, the model has
    not learned anything useful from the face encodings.
    """
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    return baseline


# ------------------------------------------------------------------
# 4. CLASSIFIERS
# ------------------------------------------------------------------
def train_svm(X_train, y_train):
    """
    Linear-kernel SVM. probability=True enables predict_proba, which is
    used at inference time as a confidence score (stored in
    attendance.confidence when marking someone present).
    """
    model = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=3):
    """
    KNN classifier. n_neighbors is capped to the smallest class size to
    avoid sklearn errors when a class has fewer training samples than
    n_neighbors.
    """
    min_class_size = min(Counter(y_train).values())
    k = min(n_neighbors, min_class_size)
    if k != n_neighbors:
        print(f"[INFO] Reduced KNN n_neighbors from {n_neighbors} to {k} "
              f"(smallest training class has only {min_class_size} sample(s)).")

    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# 5. EVALUATION
# ------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    """
    Compute accuracy, precision, recall, F1 (macro-averaged — treats every
    student equally regardless of how many images they have), and the
    confusion matrix.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print(f"\n{'=' * 60}")
    print(f"{model_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (macro-avg)")
    print(f"Recall   : {recall:.4f} (macro-avg)")
    print(f"F1 Score : {f1:.4f} (macro-avg)")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_artifact(obj, filename):
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVED] {path}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CLASSIFIER TRAINING PIPELINE — Step 2: Baseline / Train / Test / Evaluate")
    print("=" * 60)

    # 1. Load data FROM THE DATABASE (face_encodings table)
    X, y = load_training_data_from_db()
    check_minimum_samples(y)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

    # 3. Baseline
    print("\n[INFO] Training baseline (majority-class) classifier...")
    baseline_model = train_baseline(X_train, y_train_enc)
    baseline_results = evaluate_model(
        baseline_model, X_test, y_test_enc, label_encoder, "Baseline (most frequent)"
    )

    # 4. Real classifiers — SVM and KNN on the SAME split
    print("\n[INFO] Training SVM...")
    svm_model = train_svm(X_train, y_train_enc)

    print("[INFO] Training KNN...")
    knn_model = train_knn(X_train, y_train_enc)

    svm_results = evaluate_model(svm_model, X_test, y_test_enc, label_encoder, "SVM")
    knn_results = evaluate_model(knn_model, X_test, y_test_enc, label_encoder, "KNN")

    # ---- Comparison table: Baseline vs SVM vs KNN ----
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON (Baseline vs SVM vs KNN)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<12}{'Baseline':>12}{'SVM':>10}{'KNN':>10}")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"{metric:<12}{baseline_results[metric]:>12.4f}"
              f"{svm_results[metric]:>10.4f}{knn_results[metric]:>10.4f}")

    if svm_results["accuracy"] <= baseline_results["accuracy"] and \
       knn_results["accuracy"] <= baseline_results["accuracy"]:
        print(
            "\n[WARNING] Neither SVM nor KNN outperformed the majority-class "
            "baseline. With very few samples per class this can happen by "
            "chance — consider this carefully before reporting strong "
            "accuracy claims in the thesis."
        )

    # ---- Save everything needed for inference + the notebook ----
    save_artifact(svm_model, "svm_model.pkl")
    save_artifact(knn_model, "knn_model.pkl")
    save_artifact(label_encoder, "label_encoder.pkl")
    save_artifact(
        {"baseline": baseline_results, "svm": svm_results, "knn": knn_results},
        "evaluation_results.pkl",
    )

    print("\n[DONE] Training complete. Open evaluate_models.ipynb to "
          "generate the full visual report (confusion matrix plots, "
          "comparison charts) for the thesis.")


if __name__ == "__main__":
    main()