"""
run_attendance.py
==================
Full end-to-end attendance flow, run from the terminal — this is the
integration point between the ML side (ml_pipeline/recognize_faces.py)
and the database side (app/db_handler.py), before any Streamlit UI exists.

What it does:
    1. Lists existing courses, lets you pick one (or create a new one)
    2. Starts a new session for that course (sessions table)
    3. Opens the webcam, detects + classifies faces frame-by-frame using
       the trained SVM model
    4. For each recognized face, calls db_handler.mark_attendance(), which
       enforces: must be enrolled in the course, and only counted ONCE
       per session no matter how many frames they appear in
    5. Shows a live on-screen count of unique students marked present
    6. On pressing 'q': stops the session, exports the CSV report,
       prints a summary

Usage (from project root):
    python app\\run_attendance.py
"""

import sys
from pathlib import Path

import cv2

# Make ml_pipeline importable from here, since this file lives in app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml_pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from recognize_faces import load_model_and_labels, recognize_frame  # noqa: E402
import db_handler as db  # noqa: E402


def choose_course():
    """
    Prints existing courses and lets the user pick one by number, or
    create a new course on the fly. Returns a course dict.
    """
    courses = db.get_all_courses()

    if courses:
        print("\nExisting courses:")
        for i, course in enumerate(courses, start=1):
            print(f"  {i}. {course['course_code']} — {course['course_name']}")
        print(f"  {len(courses) + 1}. Create a new course")

        choice = input("\nSelect a course number: ").strip()
        try:
            choice_idx = int(choice)
        except ValueError:
            print("Invalid input.")
            return choose_course()

        if 1 <= choice_idx <= len(courses):
            return courses[choice_idx - 1]
        elif choice_idx == len(courses) + 1:
            return create_new_course()
        else:
            print("Invalid selection.")
            return choose_course()
    else:
        print("\nNo courses exist yet.")
        return create_new_course()


def create_new_course():
    """Prompts for course code/name, creates it, returns the course dict."""
    course_code = input("Enter new course code (e.g. CSC301): ").strip()
    course_name = input("Enter course name (e.g. Computer Vision): ").strip()
    try:
        db.add_course(course_code, course_name)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return choose_course()
    return db.get_course_by_code(course_code)


def ensure_enrollment_setup(course):
    """
    Checks if the course has any enrolled students. If not, offers to
    enroll all currently registered students (useful for first-time setup
    / thesis demo) since otherwise EVERY recognized face would be
    rejected as not_enrolled and nothing would ever get logged.
    """
    enrolled = db.get_enrolled_students(course["course_id"])
    if enrolled:
        print(f"\n{len(enrolled)} student(s) already enrolled in {course['course_code']}: "
              f"{[s['student_name'] for s in enrolled]}")
        return

    print(f"\n[INFO] No students are enrolled in {course['course_code']} yet.")
    answer = input("Enroll ALL registered students in this course now? (y/n): ").strip().lower()
    if answer == "y":
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, student_name FROM students")
        all_students = cursor.fetchall()
        conn.close()

        for student_id, student_name in all_students:
            db.enroll_student(student_id, course["course_id"])
        print(f"[OK] Enrolled {len(all_students)} student(s): "
              f"{[s[1] for s in all_students]}")
    else:
        print("[WARNING] Proceeding with no enrolled students — "
              "all recognized faces will be rejected as not_enrolled.")


def run_attendance_session(course):
    """
    Starts a session, runs the live webcam recognition + attendance
    marking loop, stops the session on 'q', exports CSV, prints summary.
    """
    svm_model, label_encoder = load_model_and_labels()
    print(f"\n[INFO] Loaded SVM model. Known students: {list(label_encoder.classes_)}")

    session_id = db.start_session(course["course_id"])
    print(f"[OK] Started session {session_id} for {course['course_code']}")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        db.stop_session(session_id)
        raise RuntimeError("Could not open webcam (index 0). Check camera connection/permissions.")

    print("\n[INFO] Attendance started. Press 'q' to STOP attendance.\n")

    marked_present = set()      # student_names successfully marked this session
    rejected_not_enrolled = set()  # recognized but not enrolled — logged once, not spammed

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[WARNING] Failed to read frame from webcam.")
                break

            results = recognize_frame(frame, svm_model, label_encoder)

            for result in results:
                name = result["name"]
                if name == "Unknown":
                    continue

                outcome = db.mark_attendance(session_id, name, confidence=result["confidence"])

                if outcome["status"] == "marked":
                    marked_present.add(name)
                    print(f"[PRESENT] {name} (confidence: {result['confidence']:.2f})")
                elif outcome["status"] == "not_enrolled" and name not in rejected_not_enrolled:
                    rejected_not_enrolled.add(name)
                    print(f"[REJECTED] {name} recognized but NOT enrolled in "
                          f"{course['course_code']} — not marked present.")
                # "already_marked" -> silent, this is the expected common case

            # ---- Draw boxes + labels on the frame ----
            for result in results:
                top, right, bottom, left = result["box"]
                name = result["name"]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({result['confidence']:.2f})"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(
                frame, f"{course['course_code']} | Present: {len(marked_present)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
            )
            cv2.putText(
                frame, "Press 'q' to stop attendance",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            cv2.imshow("Attendance — Live Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    db.stop_session(session_id)
    print(f"\n[OK] Attendance stopped for session {session_id}.")

    csv_path = db.export_session_to_csv(session_id)

    print(f"\n{'=' * 60}")
    print("SESSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Course        : {course['course_code']} — {course['course_name']}")
    print(f"Session ID    : {session_id}")
    print(f"Present count : {len(marked_present)}")
    print(f"Present       : {sorted(marked_present) if marked_present else '(none)'}")
    if rejected_not_enrolled:
        print(f"Recognized but NOT enrolled (excluded): {sorted(rejected_not_enrolled)}")
    print(f"CSV report    : {csv_path}")


def main():
    print("=" * 60)
    print("FACE ATTENDANCE — Live Session")
    print("=" * 60)

    course = choose_course()
    ensure_enrollment_setup(course)
    run_attendance_session(course)


if __name__ == "__main__":
    main()