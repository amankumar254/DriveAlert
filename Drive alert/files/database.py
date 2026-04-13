"""
╔══════════════════════════════════════════════════════════════════╗
║   DRIVEALERT — database.py                                       ║
║   SQLite database for users, sessions, and driving stats         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import hashlib
import secrets
import os
import datetime

DB_FILE = "drivealert.db"


# ══════════════════════════════════════════════════════════════════
#  CONNECTION
# ══════════════════════════════════════════════════════════════════

def get_db():
    """Get a database connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ══════════════════════════════════════════════════════════════════
#  SCHEMA — create tables on first run
# ══════════════════════════════════════════════════════════════════

def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    cur  = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            full_name     TEXT    DEFAULT '',
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT,
            total_trips   INTEGER DEFAULT 0,
            total_alerts  INTEGER DEFAULT 0
        )
    """)

    # Sessions (driving trips) table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS driving_sessions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            date           TEXT    NOT NULL,
            duration_min   REAL    DEFAULT 0,
            blink_count    INTEGER DEFAULT 0,
            yawn_count     INTEGER DEFAULT 0,
            alert_count    INTEGER DEFAULT 0,
            safe_score     INTEGER DEFAULT 100,
            avg_fatigue    REAL    DEFAULT 0,
            avg_ear        REAL    DEFAULT 0,
            avg_mar        REAL    DEFAULT 0,
            max_fatigue    REAL    DEFAULT 0,
            perclos_avg    REAL    DEFAULT 0,
            head_tilt_avg  REAL    DEFAULT 0,
            status         TEXT    DEFAULT 'COMPLETED',
            notes          TEXT    DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Alert events table (each individual drowsiness alert)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alert_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            timestamp   TEXT    NOT NULL,
            ear_value   REAL    DEFAULT 0,
            mar_value   REAL    DEFAULT 0,
            fatigue_val REAL    DEFAULT 0,
            yawn_count  INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES driving_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id)    REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Auth tokens table (for session management)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token      TEXT    PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            created_at TEXT    DEFAULT (datetime('now')),
            expires_at TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized →", DB_FILE)


# ══════════════════════════════════════════════════════════════════
#  PASSWORD HASHING
# ══════════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: str) -> str:
    """SHA-256 hash with salt."""
    return hashlib.sha256((salt + password).encode()).hexdigest()


def _make_salt() -> str:
    return secrets.token_hex(16)


# ══════════════════════════════════════════════════════════════════
#  USER OPERATIONS
# ══════════════════════════════════════════════════════════════════

def create_user(username: str, email: str, password: str, full_name: str = "") -> dict:
    """
    Register a new user.
    Returns {'success': True, 'user_id': int} or {'success': False, 'error': str}
    """
    conn = get_db()
    try:
        salt          = _make_salt()
        password_hash = _hash_password(password, salt)
        cur = conn.execute(
            """INSERT INTO users (username, email, password_hash, salt, full_name)
               VALUES (?, ?, ?, ?, ?)""",
            (username.strip(), email.strip().lower(), password_hash, salt, full_name.strip())
        )
        conn.commit()
        return {'success': True, 'user_id': cur.lastrowid}
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return {'success': False, 'error': 'Username already taken'}
        if 'email' in str(e):
            return {'success': False, 'error': 'Email already registered'}
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def verify_user(username_or_email: str, password: str) -> dict:
    """
    Verify login credentials.
    Returns {'success': True, 'user': {...}} or {'success': False, 'error': str}
    """
    conn = get_db()
    try:
        val = username_or_email.strip()
        row = conn.execute(
            "SELECT * FROM users WHERE username=? OR email=?",
            (val, val.lower())
        ).fetchone()

        if not row:
            return {'success': False, 'error': 'User not found'}

        expected = _hash_password(password, row['salt'])
        if expected != row['password_hash']:
            return {'success': False, 'error': 'Incorrect password'}

        # Update last login
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), row['id'])
        )
        conn.commit()
        return {'success': True, 'user': dict(row)}
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
#  AUTH TOKEN OPERATIONS
# ══════════════════════════════════════════════════════════════════

def create_token(user_id: int, days: int = 7) -> str:
    """Create a login session token valid for `days` days."""
    token      = secrets.token_urlsafe(32)
    expires_at = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at)
        )
        conn.commit()
        return token
    finally:
        conn.close()


def validate_token(token: str) -> dict | None:
    """
    Validate a session token.
    Returns user dict if valid, None if expired/not found.
    """
    if not token:
        return None
    conn = get_db()
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        row = conn.execute(
            """SELECT u.* FROM auth_tokens t
               JOIN users u ON u.id = t.user_id
               WHERE t.token=? AND t.expires_at > ?""",
            (token, now)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_token(token: str):
    """Logout — delete token."""
    conn = get_db()
    try:
        conn.execute("DELETE FROM auth_tokens WHERE token=?", (token,))
        conn.commit()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
#  DRIVING SESSION OPERATIONS
# ══════════════════════════════════════════════════════════════════

def save_driving_session(user_id: int, summary: dict) -> int:
    """
    Save a completed driving session.
    Returns the new session id.
    """
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO driving_sessions
               (user_id, date, duration_min, blink_count, yawn_count,
                alert_count, safe_score, avg_fatigue, avg_ear, avg_mar,
                max_fatigue, perclos_avg, head_tilt_avg, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                summary.get('date',          datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
                summary.get('duration_min',  0),
                summary.get('blink_count',   0),
                summary.get('yawn_count',    0),
                summary.get('alert_count',   0),
                summary.get('safe_score',    100),
                summary.get('avg_fatigue',   0),
                summary.get('avg_ear',       0),
                summary.get('avg_mar',       0),
                summary.get('max_fatigue',   0),
                summary.get('perclos_avg',   0),
                summary.get('head_tilt_avg', 0),
                'COMPLETED',
            )
        )
        session_id = cur.lastrowid

        # Update user totals
        conn.execute(
            """UPDATE users SET
               total_trips  = total_trips  + 1,
               total_alerts = total_alerts + ?
               WHERE id = ?""",
            (summary.get('alert_count', 0), user_id)
        )
        conn.commit()
        return session_id
    finally:
        conn.close()


def save_alert_event(session_id: int, user_id: int, event: dict):
    """Save an individual drowsiness alert event."""
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO alert_events
               (session_id, user_id, timestamp, ear_value, mar_value, fatigue_val, yawn_count)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session_id,
                user_id,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event.get('ear',    0),
                event.get('mar',    0),
                event.get('fatigue_score', 0),
                event.get('yawns',  0),
            )
        )
        conn.commit()
    finally:
        conn.close()


def get_user_sessions(user_id: int, limit: int = 50) -> list:
    """Get all driving sessions for a user, newest first."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM driving_sessions
               WHERE user_id=? ORDER BY date DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_user_stats(user_id: int) -> dict:
    """Aggregate stats for a user's profile/dashboard."""
    conn = get_db()
    try:
        row = conn.execute(
            """SELECT
               COUNT(*)                      AS total_trips,
               SUM(alert_count)              AS total_alerts,
               SUM(yawn_count)               AS total_yawns,
               SUM(blink_count)              AS total_blinks,
               ROUND(AVG(safe_score),   1)   AS avg_safe_score,
               ROUND(AVG(avg_fatigue),  1)   AS avg_fatigue,
               ROUND(MAX(max_fatigue),  1)   AS max_fatigue_ever,
               ROUND(AVG(perclos_avg),  4)   AS perclos_avg,
               ROUND(SUM(duration_min), 1)   AS total_drive_min,
               MAX(safe_score)               AS best_score,
               MIN(safe_score)               AS worst_score
               FROM driving_sessions WHERE user_id=?""",
            (user_id,)
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def get_weekly_scores(user_id: int) -> list:
    """Last 7 sessions' safe scores for the bar chart."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT date, safe_score, alert_count, duration_min
               FROM driving_sessions
               WHERE user_id=? ORDER BY date DESC LIMIT 7""",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()
