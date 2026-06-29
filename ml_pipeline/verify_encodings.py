"""
verify_encodings.py
====================
Confirms that every student in the `students` table has a VALID, usable
128-dimensional face encoding stored as a BLOB -- not just that a row
exists. Unpickles each stored encoding and checks its shape/dtype.

Usage (from project root):
    python ml_pipeline\verify_encodings.py
"""

import pickle
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "attendance.db"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, student_name, face_encoding, date_registered FROM students")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[EMPTY] No students found in the database. Run encode_faces.py first.")
        return

    print(f"[INFO] Found {len(rows)} student record(s) in the database.\n")
    print(f"{'ID':<4}{'Name':<20}{'Encoding shape':<18}{'dtype':<10}{'Registered'}")
    print("-" * 75)

    all_valid = True

    for student_id, name, encoding_blob, date_registered in rows:
        try:
            encoding = pickle.loads(encoding_blob)
            shape = encoding.shape
            dtype = encoding.dtype
            is_valid = isinstance(encoding, np.ndarray) and shape == (128,)
            status = "" if is_valid else "  <-- INVALID SHAPE"
            if not is_valid:
                all_valid = False
            print(f"{student_id:<4}{name:<20}{str(shape):<18}{str(dtype):<10}{date_registered}{status}")
        except Exception as e:
            all_valid = False
            print(f"{student_id:<4}{name:<20}{'UNPICKLE FAILED':<18}{'-':<10}{date_registered}  <-- {e}")

    print("-" * 75)
    if all_valid:
        print("\n[OK] All stored encodings are valid 128-dimensional vectors.")
    else:
        print("\n[WARNING] One or more encodings are invalid or corrupted. "
              "Re-run encode_faces.py for the affected student(s).")


if __name__ == "__main__":
    main()