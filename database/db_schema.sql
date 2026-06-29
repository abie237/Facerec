-- ============================================================
-- Face Attendance System — Database Schema (SQLite)
-- ============================================================
-- Relationship summary:
--   students  <--one-to-many-->    face_encodings (each row = one
--                                   training image's 128-d encoding)
--   students  <--many-to-many-->  courses   (via enrollments)
--   courses   <--one-to-many-->   sessions  (each session = one class meeting)
--   sessions  <--many-to-many-->  students  (via attendance, but constrained
--                                             so a student can only be marked
--                                             present ONCE per session)
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- 1. STUDENTS
-- One row per registered student. face_encoding stores the
-- AVERAGED 128-d face_recognition embedding (across all of that
-- student's training images), used for fast single-vector lookups.
-- The individual per-image encodings used for SVM/KNN TRAINING live
-- in the face_encodings table below (one-to-many).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    student_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name    TEXT NOT NULL UNIQUE,     -- matches dataset folder name
    face_encoding   BLOB NOT NULL,            -- pickled AVERAGED 128-d numpy vector
    date_registered TEXT DEFAULT (datetime('now'))
);

-- ------------------------------------------------------------
-- 1b. FACE_ENCODINGS  (one-to-many: students -> face_encodings)
-- One row per TRAINING IMAGE encoding. This is the table
-- train_classifier.py reads from to build X (encodings) and y
-- (student_name labels) for the train/test split -- a single
-- averaged vector per student (as stored in `students`) cannot be
-- split into train/test, so the raw per-image vectors are kept here.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_encodings (
    encoding_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL,
    image_filename  TEXT NOT NULL,            -- e.g. "1.jpg", for traceability
    face_encoding   BLOB NOT NULL,            -- pickled 128-d numpy vector for THIS image
    date_created    TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 2. COURSES
-- One row per course offered.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS courses (
    course_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code   TEXT NOT NULL UNIQUE,       -- e.g. "CSC301"
    course_name   TEXT NOT NULL               -- e.g. "Computer Vision"
);

-- ------------------------------------------------------------
-- 3. ENROLLMENTS  (resolves student <-> course many-to-many)
-- A student can be enrolled in many courses; a course has many students.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    INTEGER NOT NULL,
    course_id     INTEGER NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id)  REFERENCES courses(course_id)  ON DELETE CASCADE,
    UNIQUE (student_id, course_id)             -- prevents duplicate enrollment
);

-- ------------------------------------------------------------
-- 4. SESSIONS
-- One row per class meeting/lesson for a course (created when
-- attendance is "started" in the Streamlit app for that course).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id     INTEGER NOT NULL,
    session_date  TEXT NOT NULL DEFAULT (date('now')),
    start_time    TEXT,                       -- set when attendance "Start" pressed
    end_time      TEXT,                       -- set when attendance "Stop" pressed
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 5. ATTENDANCE  (resolves student <-> session many-to-many)
-- One row per student marked present in a given session.
-- UNIQUE constraint prevents the same student being logged twice
-- in one session (e.g. detected across multiple frames).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL,
    student_id      INTEGER NOT NULL,
    timestamp_marked TEXT NOT NULL DEFAULT (datetime('now')),
    confidence      REAL,                     -- classifier confidence/probability
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    UNIQUE (session_id, student_id)            -- duplicate-attendance prevention
);

-- ------------------------------------------------------------
-- Helpful indexes for common lookups
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_face_encodings_student ON face_encodings(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_student ON enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_course  ON enrollments(course_id);
CREATE INDEX IF NOT EXISTS idx_sessions_course     ON sessions(course_id);
CREATE INDEX IF NOT EXISTS idx_attendance_session  ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student  ON attendance(student_id);