"""
db_handler.py
=============
All database read/write logic for courses, enrollments, sessions, and
attendance. This is the layer the Streamlit app will call into — kept
separate and tested here first so the UI layer has no database logic of
its own, just button clicks calling these functions.

Tables touched by this module (see database/db_schema.sql):
    courses, enrollments, sessions, attendance
(students / face_encodings are written by ml_pipeline/encode_faces.py,
this module only READS from students to validate enrollment.)

Run this file directly to execute a self-test (see bottom of file):
    python app\\db_handler.py
"""

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "attendance.db"
REPORTS_DIR = PROJECT_ROOT / "data" / "attendance_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    """
    Single place all DB connections are created. Centralizing this means
    the path/connection logic can later be swapped (e.g. a hosted DB) by
    changing only this function.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run database/init_db.py first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ======================================================================
# COURSES
# ======================================================================
def add_course(course_code: str, course_name: str) -> int:
    """Insert a new course. Returns the course_id. Raises ValueError if
    course_code already exists (UNIQUE constraint)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO courses (course_code, course_name) VALUES (?, ?)",
            (course_code, course_name),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"Course code '{course_code}' already exists.")
    finally:
        conn.close()


def get_all_courses():
    """Returns list of dicts: [{"course_id", "course_code", "course_name"}, ...]"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT course_id, course_code, course_name FROM courses ORDER BY course_code")
        rows = cursor.fetchall()
        return [{"course_id": r[0], "course_code": r[1], "course_name": r[2]} for r in rows]
    finally:
        conn.close()


def get_course_by_code(course_code: str):
    """Returns a single course dict, or None if not found."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT course_id, course_code, course_name FROM courses WHERE course_code = ?",
            (course_code,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {"course_id": row[0], "course_code": row[1], "course_name": row[2]}
    finally:
        conn.close()


# ======================================================================
# ENROLLMENTS
# ======================================================================
def enroll_student(student_id: int, course_id: int):
    """
    Enroll a student in a course. Silently does nothing if already
    enrolled (relies on the UNIQUE(student_id, course_id) constraint).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)",
                (student_id, course_id),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Already enrolled — not an error, just a no-op
            return False
    finally:
        conn.close()


def get_enrolled_students(course_id: int):
    """Returns list of dicts: [{"student_id", "student_name"}, ...] for a course."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.student_id, s.student_name
            FROM enrollments e
            JOIN students s ON e.student_id = s.student_id
            WHERE e.course_id = ?
            ORDER BY s.student_name
            """,
            (course_id,),
        )
        rows = cursor.fetchall()
        return [{"student_id": r[0], "student_name": r[1]} for r in rows]
    finally:
        conn.close()


def is_student_enrolled(student_id: int, course_id: int) -> bool:
    """Check whether a student is enrolled in a specific course."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM enrollments WHERE student_id = ? AND course_id = ?",
            (student_id, course_id),
        )
        return cursor.fetchone() is not None
    finally:
        conn.close()


def get_student_by_name(student_name: str):
    """Returns {"student_id", "student_name"} or None. Used to resolve the
    name returned by the SVM/KNN classifier into a student_id."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, student_name FROM students WHERE student_name = ?",
            (student_name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {"student_id": row[0], "student_name": row[1]}
    finally:
        conn.close()


# ======================================================================
# SESSIONS  (start / stop attendance)
# ======================================================================
def start_session(course_id: int) -> int:
    """
    Creates a new session row for this course with start_time set to now.
    Returns the new session_id. This is what "Start Attendance" in the
    Streamlit app will call.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        cursor.execute(
            "INSERT INTO sessions (course_id, start_time) VALUES (?, ?)",
            (course_id, now),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def stop_session(session_id: int):
    """
    Sets end_time on the session to now. This is what "Stop Attendance"
    in the Streamlit app will call.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        cursor.execute(
            "UPDATE sessions SET end_time = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_session(session_id: int):
    """Returns session details as a dict, or None if not found."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.session_id, s.course_id, c.course_code, c.course_name,
                   s.session_date, s.start_time, s.end_time
            FROM sessions s
            JOIN courses c ON s.course_id = c.course_id
            WHERE s.session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0], "course_id": row[1], "course_code": row[2],
            "course_name": row[3], "session_date": row[4],
            "start_time": row[5], "end_time": row[6],
        }
    finally:
        conn.close()


# ======================================================================
# ATTENDANCE  (mark present during a live session)
# ======================================================================
def mark_attendance(session_id: int, student_name: str, confidence: float = None):
    """
    Marks a recognized student present for a session.

    Enforces TWO rules:
      1. The student must be ENROLLED in the session's course — a
         recognized-but-unenrolled face is rejected, not logged.
      2. A student can only be marked present ONCE per session — repeat
         detections across frames are silently ignored (handled by the
         UNIQUE(session_id, student_id) constraint), not an error.

    Returns a dict describing what happened:
        {"status": "marked", "student_id": int, "student_name": str}
        {"status": "already_marked", ...}
        {"status": "not_enrolled", ...}
        {"status": "unknown_student", "student_name": str}
    """
    student = get_student_by_name(student_name)
    if student is None:
        return {"status": "unknown_student", "student_name": student_name}

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} does not exist.")

    if not is_student_enrolled(student["student_id"], session["course_id"]):
        return {
            "status": "not_enrolled",
            "student_id": student["student_id"],
            "student_name": student_name,
            "course_code": session["course_code"],
        }

    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO attendance (session_id, student_id, confidence) VALUES (?, ?, ?)",
                (session_id, student["student_id"], confidence),
            )
            conn.commit()
            return {
                "status": "marked",
                "student_id": student["student_id"],
                "student_name": student_name,
            }
        except sqlite3.IntegrityError:
            # UNIQUE(session_id, student_id) violated -> already marked present
            return {
                "status": "already_marked",
                "student_id": student["student_id"],
                "student_name": student_name,
            }
    finally:
        conn.close()


def get_attendance_for_session(session_id: int):
    """
    Returns list of dicts for everyone marked present in a session:
    [{"student_id", "student_name", "timestamp_marked", "confidence"}, ...]
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.student_id, s.student_name, a.timestamp_marked, a.confidence
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.session_id = ?
            ORDER BY a.timestamp_marked
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        return [
            {"student_id": r[0], "student_name": r[1], "timestamp_marked": r[2], "confidence": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def get_attendance_count(session_id: int) -> int:
    """Returns the count of unique students marked present in a session —
    this is the live 'number of faces recognised' counter for the UI."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE session_id = ?", (session_id,))
        return cursor.fetchone()[0]
    finally:
        conn.close()


# ======================================================================
# CSV EXPORT
# ======================================================================
def export_session_to_csv(session_id: int) -> Path:
    """
    Writes a CSV report for a session: every enrolled student, whether
    they were present or absent, and their marked timestamp/confidence
    if present. Returns the path to the written file.

    Including ABSENT students (not just present ones) is deliberate —
    a usable attendance report needs the full roster, not just a list
    of who showed up.
    """
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} does not exist.")

    enrolled = get_enrolled_students(session["course_id"])
    present_records = {r["student_id"]: r for r in get_attendance_for_session(session_id)}

    filename = (
        f"{session['course_code']}_session{session_id}_"
        f"{session['session_date']}.csv"
    )
    output_path = REPORTS_DIR / filename

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "course_code", "course_name", "session_id", "session_date",
            "student_name", "status", "timestamp_marked", "confidence",
        ])

        for student in enrolled:
            record = present_records.get(student["student_id"])
            if record:
                writer.writerow([
                    session["course_code"], session["course_name"], session_id,
                    session["session_date"], student["student_name"], "present",
                    record["timestamp_marked"], f"{record['confidence']:.4f}" if record["confidence"] else "",
                ])
            else:
                writer.writerow([
                    session["course_code"], session["course_name"], session_id,
                    session["session_date"], student["student_name"], "absent",
                    "", "",
                ])

    print(f"[SAVED] Attendance report: {output_path}")
    return output_path


# ======================================================================
# SELF-TEST
# ======================================================================
if __name__ == "__main__":
    """
    Quick end-to-end self-test of this module against the REAL database.
    Run with: python app\\db_handler.py

    NOTE: This test requires at least one student to already exist in the
    `students` table (i.e. you've already run encode_faces.py). It creates
    a temporary test course, enrolls a real student, starts a session,
    marks attendance, stops the session, and exports a CSV — then leaves
    everything in place so you can inspect it (it does NOT clean up after
    itself, since you may want to see the resulting rows/CSV).
    """
    print("=" * 60)
    print("db_handler.py SELF-TEST")
    print("=" * 60)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_name FROM students LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row is None:
        print("[ABORTED] No students found in the database. "
              "Run ml_pipeline/encode_faces.py first.")
        raise SystemExit(1)

    test_student_name = row[0]
    print(f"[INFO] Using existing student for test: '{test_student_name}'")

    test_course_code = "TEST101"
    existing = get_course_by_code(test_course_code)
    if existing:
        course = existing
        print(f"[INFO] Test course '{test_course_code}' already exists, reusing it.")
    else:
        course_id = add_course(test_course_code, "Self-Test Course")
        course = get_course_by_code(test_course_code)
        print(f"[OK] Created test course: {course}")

    student = get_student_by_name(test_student_name)
    enrolled_already = is_student_enrolled(student["student_id"], course["course_id"])
    if not enrolled_already:
        enroll_student(student["student_id"], course["course_id"])
        print(f"[OK] Enrolled '{test_student_name}' in {test_course_code}")
    else:
        print(f"[INFO] '{test_student_name}' already enrolled in {test_course_code}")

    session_id = start_session(course["course_id"])
    print(f"[OK] Started session {session_id} for {test_course_code}")

    result = mark_attendance(session_id, test_student_name, confidence=0.93)
    print(f"[OK] mark_attendance result: {result}")

    result_duplicate = mark_attendance(session_id, test_student_name, confidence=0.93)
    print(f"[OK] mark_attendance (duplicate call) result: {result_duplicate}")

    result_unknown = mark_attendance(session_id, "Totally_Fake_Name_XYZ")
    print(f"[OK] mark_attendance (unknown student) result: {result_unknown}")

    count = get_attendance_count(session_id)
    print(f"[OK] get_attendance_count: {count}")

    stop_session(session_id)
    print(f"[OK] Stopped session {session_id}")

    csv_path = export_session_to_csv(session_id)
    print(f"[OK] CSV exported to: {csv_path}")

    print("\n[DONE] Self-test completed successfully. "
          f"Inspect the CSV at {csv_path} to verify the report looks correct.")