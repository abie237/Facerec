"""
flask_app.py
=============
Main Flask application for the Face Attendance System.

Routes:
    /                          -> home page
    /attendance                -> live attendance page
    /video_feed                -> MJPEG video stream
    /start_session             -> POST: start attendance session
    /stop_session              -> POST: stop attendance session
    /get_status                -> GET:  live attendance status (JSON)
    /reports                   -> reports page
    /reports/download/<id>     -> download CSV for a session
    /manage                    -> manage courses and enrollments
    /manage/add_course         -> POST: add a new course
    /manage/enroll             -> POST: enroll students

Usage (from project root):
    python app/flask_app.py
"""

import sys
import json
import threading
from pathlib import Path
from datetime import datetime

import cv2
from flask import (
    Flask, Response, render_template, request,
    redirect, url_for, flash, jsonify, send_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml_pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import db_handler as db
from recognize_faces import load_model_and_labels, recognize_frame

app = Flask(__name__)
app.secret_key = "face_attendance_secret_key"
app.template_folder = str(Path(__file__).parent / "templates")
app.static_folder = str(Path(__file__).parent / "static")

# ------------------------------------------------------------------
# Global state (shared between video thread and Flask routes)
# ------------------------------------------------------------------
camera_lock = threading.Lock()
state_lock = threading.Lock()

_camera = None
_knn_model = None
_label_encoder = None

# Shared session state (written by video thread, read by routes)
current_state = {
    "session_id": None,
    "course_code": None,
    "course_name": None,
    "marked_present": set(),
    "running": False,
}


# ------------------------------------------------------------------
# Model + camera helpers
# ------------------------------------------------------------------
def get_model():
    global _knn_model, _label_encoder
    if _knn_model is None:
        _knn_model, _label_encoder = load_model_and_labels()
    return _knn_model, _label_encoder


def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
    return _camera


def release_camera():
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


# ------------------------------------------------------------------
# MJPEG video generator
# ------------------------------------------------------------------
def generate_frames():
    """
    Generator that yields MJPEG frames to the browser.
    Each frame is:
      1. Read from the webcam
      2. Run through face detection + KNN classification (if session active)
      3. Annotated with bounding boxes and names
      4. Encoded as JPEG and yielded as a multipart HTTP response
    """
    knn_model, label_encoder = get_model()

    while True:
        camera = get_camera()
        with camera_lock:
            success, frame = camera.read()

        if not success:
            break

        with state_lock:
            session_id = current_state["session_id"]
            running = current_state["running"]

        if running and session_id is not None:
            results = recognize_frame(frame, knn_model, label_encoder)

            for result in results:
                name = result["name"]
                confidence = result["confidence"]
                distance = result["distance"]
                top, right, bottom, left = result["box"]

                if name != "Unknown":
                    outcome = db.mark_attendance(
                        session_id, name, confidence=confidence
                    )
                    if outcome["status"] == "marked":
                        with state_lock:
                            current_state["marked_present"].add(name)

                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({confidence:.2f}, d={distance:.2f})"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(
                    frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED
                )
                cv2.putText(
                    frame, label, (left + 4, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1,
                )

            # Overlay session info on frame
            with state_lock:
                present_count = len(current_state["marked_present"])
                course_code = current_state["course_code"]
            overlay = f"{course_code} | Present: {present_count}"
            cv2.putText(
                frame, overlay, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
            )
        else:
            # No active session — show standby message
            cv2.putText(
                frame, "Attendance not started", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2,
            )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/attendance")
def attendance():
    courses = db.get_all_courses()
    with state_lock:
        state = {
            "session_id": current_state["session_id"],
            "course_code": current_state["course_code"],
            "course_name": current_state["course_name"],
            "running": current_state["running"],
            "present_count": len(current_state["marked_present"]),
            "marked_present": sorted(current_state["marked_present"]),
        }
    return render_template("attendance.html", courses=courses, state=state)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint — used as <img src='/video_feed'> in the template."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/start_session", methods=["POST"])
def start_session():
    course_id = int(request.form.get("course_id"))
    course = None
    for c in db.get_all_courses():
        if c["course_id"] == course_id:
            course = c
            break

    if course is None:
        flash("Course not found.", "error")
        return redirect(url_for("attendance"))

    enrolled = db.get_enrolled_students(course_id)
    if not enrolled:
        flash(
            f"No students enrolled in {course['course_code']}. "
            "Enroll students first.", "warning"
        )
        return redirect(url_for("attendance"))

    session_id = db.start_session(course_id)
    with state_lock:
        current_state["session_id"] = session_id
        current_state["course_code"] = course["course_code"]
        current_state["course_name"] = course["course_name"]
        current_state["marked_present"] = set()
        current_state["running"] = True

    flash(
        f"Session {session_id} started for {course['course_code']}.", "success"
    )
    return redirect(url_for("attendance"))


@app.route("/stop_session", methods=["POST"])
def stop_session():
    with state_lock:
        session_id = current_state["session_id"]
        current_state["running"] = False

    if session_id is None:
        flash("No active session.", "warning")
        return redirect(url_for("attendance"))

    db.stop_session(session_id)
    csv_path = db.export_session_to_csv(session_id)

    with state_lock:
        current_state["session_id"] = None
        current_state["course_code"] = None
        current_state["course_name"] = None
        current_state["marked_present"] = set()

    flash(
        f"Session stopped. Attendance saved. "
        f"CSV exported to {csv_path.name}.", "success"
    )
    return redirect(url_for("attendance"))


@app.route("/get_status")
def get_status():
    """JSON endpoint polled by the attendance page every 2 seconds
    to update the present count and student list without a full page reload."""
    with state_lock:
        return jsonify({
            "session_id": current_state["session_id"],
            "running": current_state["running"],
            "present_count": len(current_state["marked_present"]),
            "marked_present": sorted(current_state["marked_present"]),
            "course_code": current_state["course_code"],
        })


# ------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------
@app.route("/reports")
def reports():
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.session_id, c.course_code, c.course_name,
               s.session_date, s.start_time, s.end_time
        FROM sessions s
        JOIN courses c ON s.course_id = c.course_id
        ORDER BY s.session_id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    sessions = [
        {
            "session_id": r[0],
            "course_code": r[1],
            "course_name": r[2],
            "session_date": r[3],
            "start_time": r[4],
            "end_time": r[5] or "In progress",
        }
        for r in rows
    ]
    return render_template("reports.html", sessions=sessions)


@app.route("/reports/<int:session_id>")
def report_detail(session_id):
    session = db.get_session(session_id)
    if session is None:
        flash("Session not found.", "error")
        return redirect(url_for("reports"))

    course = db.get_course_by_code(session["course_code"])
    enrolled = db.get_enrolled_students(course["course_id"])
    present_records = {
        r["student_id"]: r
        for r in db.get_attendance_for_session(session_id)
    }

    rows = []
    for student in enrolled:
        record = present_records.get(student["student_id"])
        rows.append({
            "student_name": student["student_name"],
            "status": "Present" if record else "Absent",
            "timestamp": record["timestamp_marked"] if record else "—",
            "confidence": f"{record['confidence']:.4f}"
            if record and record["confidence"] else "—",
        })

    present_count = sum(1 for r in rows if r["status"] == "Present")

    return render_template(
        "report_detail.html",
        session=session,
        rows=rows,
        present_count=present_count,
        total=len(rows),
    )


@app.route("/reports/download/<int:session_id>")
def download_report(session_id):
    session = db.get_session(session_id)
    if session is None:
        flash("Session not found.", "error")
        return redirect(url_for("reports"))

    REPORTS_DIR = PROJECT_ROOT / "data" / "attendance_reports"
    csv_filename = (
        f"{session['course_code']}_session{session_id}_"
        f"{session['session_date']}.csv"
    )
    csv_path = REPORTS_DIR / csv_filename

    if not csv_path.exists():
        csv_path = db.export_session_to_csv(session_id)

    return send_file(csv_path, as_attachment=True, download_name=csv_filename)


# ------------------------------------------------------------------
# Manage courses and enrollments
# ------------------------------------------------------------------
@app.route("/manage")
def manage():
    courses = db.get_all_courses()
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT student_id, student_name FROM students ORDER BY student_name"
    )
    all_students = [
        {"student_id": r[0], "student_name": r[1]}
        for r in cursor.fetchall()
    ]
    conn.close()

    enrollments = {}
    for course in courses:
        enrolled = db.get_enrolled_students(course["course_id"])
        enrollments[course["course_id"]] = [
            s["student_id"] for s in enrolled
        ]

    return render_template(
        "manage_courses.html",
        courses=courses,
        all_students=all_students,
        enrollments=enrollments,
    )


@app.route("/manage/add_course", methods=["POST"])
def add_course():
    course_code = request.form.get("course_code", "").strip().upper()
    course_name = request.form.get("course_name", "").strip()

    if not course_code or not course_name:
        flash("Both course code and name are required.", "error")
        return redirect(url_for("manage"))

    try:
        db.add_course(course_code, course_name)
        flash(f"Course {course_code} — {course_name} added.", "success")
    except ValueError as e:
        flash(str(e), "error")

    return redirect(url_for("manage"))


@app.route("/manage/enroll", methods=["POST"])
def enroll():
    course_id = int(request.form.get("course_id"))
    student_ids = request.form.getlist("student_ids")

    if not student_ids:
        flash("No students selected.", "warning")
        return redirect(url_for("manage"))

    count = 0
    for sid in student_ids:
        result = db.enroll_student(int(sid), course_id)
        if result:
            count += 1

    course = None
    for c in db.get_all_courses():
        if c["course_id"] == course_id:
            course = c
            break

    if count > 0:
        flash(
            f"Enrolled {count} new student(s) in {course['course_code']}.",
            "success",
        )
    else:
        flash("Selected students were already enrolled.", "info")

    return redirect(url_for("manage"))


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Face Attendance System — Flask")
    print("=" * 60)
    print(f"Open http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    try:
        app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
    finally:
        release_camera()