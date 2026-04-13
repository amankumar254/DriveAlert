"""
╔══════════════════════════════════════════════════════════════════╗
║   DRIVEALERT — Driver.py                                         ║
║   All detection logic imported by app.py                        ║
║                                                                  ║
║   Features:                                                      ║
║   • EAR  — Eye Aspect Ratio (eye closure detection)             ║
║   • MAR  — Mouth Aspect Ratio (yawn detection)                  ║
║   • Head Pose — pitch / yaw (nodding detection)                 ║
║   • PERCLOS — % eye closure over 60s rolling window             ║
║   • Blink rate — blinks per minute                              ║
║   • Fatigue Score — composite 0–100                             ║
║   • Session Logger — saves trips to JSON                        ║
║                                                                  ║
║   Compatible with mediapipe >= 0.10.13                          ║
║   Run directly: python driver.py                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import json
import os
import datetime
import threading
import math
import sys

# ── MediaPipe import — works with new 0.10.x API ──
try:
    import mediapipe as mp
    import mediapipe.python.solutions.face_mesh as _fm
    import mediapipe.python.solutions.drawing_utils as _du
    if not hasattr(mp, 'solutions'):
        mp.solutions = type('solutions', (), {})()
    mp.solutions.face_mesh     = _fm
    mp.solutions.drawing_utils = _du
    print("[OK] MediaPipe loaded")
except Exception as e:
    print(f"[ERROR] MediaPipe import failed: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

class Config:
    CAMERA_INDEX        = 0
    FRAME_WIDTH         = 640
    FRAME_HEIGHT        = 480
    FPS_TARGET          = 30

    EAR_THRESHOLD       = 0.25
    EAR_CONSEC_FRAMES   = 3
    EAR_DROWSY_FRAMES   = 15

    MAR_THRESHOLD       = 0.55

    HEAD_TILT_THRESHOLD = 15.0

    PERCLOS_WINDOW_SEC  = 60
    PERCLOS_THRESHOLD   = 0.15

    WEIGHT_EAR          = 0.40
    WEIGHT_MAR          = 0.20
    WEIGHT_PERCLOS      = 0.25
    WEIGHT_HEAD         = 0.15

    ALERT_COOLDOWN_SEC  = 10

    LOG_FILE            = "drivealert_sessions.json"

    SHOW_LANDMARKS      = True
    SHOW_FPS            = True


# ══════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════

LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH        = [61,  291, 39,  181, 0,   17,  269, 405]
NOSE_TIP     = 1
CHIN         = 152
LEFT_EAR_PT  = 234
RIGHT_EAR_PT = 454


# ══════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    A = dist.euclidean(pts[2], pts[6])
    B = dist.euclidean(pts[3], pts[7])
    C = dist.euclidean(pts[0], pts[4])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def head_pose_angles(landmarks, w, h):
    nose  = landmarks[NOSE_TIP]
    chin  = landmarks[CHIN]
    l_ear = landmarks[LEFT_EAR_PT]
    r_ear = landmarks[RIGHT_EAR_PT]

    pitch = math.degrees(math.atan2(
        (chin.x - nose.x) * w,
        (chin.y - nose.y) * h
    ))

    l_dist = abs(nose.x - l_ear.x)
    r_dist = abs(nose.x - r_ear.x)
    ratio  = (l_dist - r_dist) / (l_dist + r_dist + 1e-6)
    yaw    = ratio * 45.0

    return round(pitch, 1), round(yaw, 1)


# ══════════════════════════════════════════════════════════════════
#  DERIVED METRICS
# ══════════════════════════════════════════════════════════════════

def compute_perclos(eye_closed_log, window_sec):
    now    = time.time()
    cutoff = now - window_sec
    recent = [(t, c) for t, c in eye_closed_log if t >= cutoff]
    if len(recent) < 2:
        return 0.0
    return sum(1 for _, c in recent if c) / len(recent)


def compute_blink_rate(blink_timestamps, window_sec=60):
    now    = time.time()
    cutoff = now - window_sec
    recent = [t for t in blink_timestamps if t >= cutoff]
    return round(len(recent) / (window_sec / 60.0), 1)


def fatigue_score(ear, mar, perclos, head_pitch, cfg=Config):
    ear_score = max(0.0, min(100.0,
        (cfg.EAR_THRESHOLD - ear) / max(cfg.EAR_THRESHOLD, 1e-6) * 200
    ))

    mar_score = max(0.0, min(100.0,
        (mar - cfg.MAR_THRESHOLD) / (1.0 - cfg.MAR_THRESHOLD + 1e-6) * 100
    )) if mar > cfg.MAR_THRESHOLD else 0.0

    perclos_score = min(100.0, perclos / 0.30 * 100)
    head_score    = min(100.0, abs(head_pitch) / 30.0 * 100)

    score = (
        cfg.WEIGHT_EAR     * ear_score     +
        cfg.WEIGHT_MAR     * mar_score     +
        cfg.WEIGHT_PERCLOS * perclos_score +
        cfg.WEIGHT_HEAD    * head_score
    )
    return round(min(100.0, max(0.0, score)), 1)


def drowsiness_status(score, ear, ear_consec_count, cfg=Config):
    if ear_consec_count >= cfg.EAR_DROWSY_FRAMES:
        return 'DROWSY',  (0,  40, 255)
    if score >= 60 or ear < cfg.EAR_THRESHOLD:
        return 'FATIGUE', (20, 176, 255)
    if score >= 30:
        return 'MILD',    (0, 200, 255)
    return 'ALERT',       (0, 232, 122)


# ══════════════════════════════════════════════════════════════════
#  HUD DRAWING
# ══════════════════════════════════════════════════════════════════

def draw_hud(frame, state):
    h, w        = frame.shape[:2]
    status      = state.get('status',       'NO FACE')
    status_col  = state.get('status_color', (60, 60, 60))
    ear         = state.get('ear',          0.0)
    mar         = state.get('mar',          0.0)
    score       = state.get('fatigue_score',0.0)
    perclos     = state.get('perclos',      0.0)
    blink_rate  = state.get('blink_rate',   0.0)
    yawns       = state.get('yawn_count',   0)
    blinks      = state.get('blink_count',  0)
    pitch       = state.get('pitch',        0.0)
    yaw         = state.get('yaw',          0.0)
    fps         = state.get('fps',          0.0)
    alert_on    = state.get('alert_active', False)
    face_found  = state.get('face_found',   False)

    if alert_on:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    col = status_col if face_found else (60, 60, 60)
    sz  = 24
    for (bx, by) in [(0, 0), (w, 0), (0, h), (w, h)]:
        sx = 1 if bx == 0 else -1
        sy = 1 if by == 0 else -1
        ox, oy = bx + sx * 8, by + sy * 8
        cv2.line(frame, (ox, oy), (ox + sx * sz, oy), col, 2, cv2.LINE_AA)
        cv2.line(frame, (ox, oy), (ox, oy + sy * sz), col, 2, cv2.LINE_AA)

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    top_text = (f"DRIVEALERT  |  EAR:{ear:.3f}  "
                f"MAR:{mar:.3f}  PERCLOS:{perclos*100:.1f}%  [{status}]")
    cv2.putText(frame, top_text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 1, cv2.LINE_AA)

    if Config.SHOW_FPS:
        cv2.putText(frame, f"{fps:.0f}fps", (w - 60, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    py = h - 92
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, py), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.60, frame, 0.40, 0, frame)

    cw = w // 4
    stats = [
        ("FATIGUE",   f"{score:.0f}/100", (0, 200, 255) if score < 30 else
                                           (20, 176, 255) if score < 60 else
                                           (0, 40, 255)),
        ("BLINK/MIN", f"{blink_rate}",    (0, 232, 122)),
        ("YAWNS",     str(yawns),         (0, 200, 255)),
        ("PITCH",     f"{pitch}°",        (0, 232, 122) if abs(pitch) < Config.HEAD_TILT_THRESHOLD
                                           else (20, 176, 255)),
    ]
    for i, (label, value, vcol) in enumerate(stats):
        x = i * cw + 10
        cv2.putText(frame, label, (x, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x, py + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, vcol, 1, cv2.LINE_AA)

    stats2 = [
        ("BLINKS", str(blinks)),
        ("EYE",    "CLOSED" if ear < Config.EAR_THRESHOLD else "OPEN"),
        ("YAW",    f"{yaw}°"),
        ("MARKS",  "ON" if state.get('show_landmarks', True) else "OFF"),
    ]
    for i, (label, value) in enumerate(stats2):
        x = i * cw + 10
        cv2.putText(frame, label, (x, py + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x, py + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (140, 140, 140), 1, cv2.LINE_AA)

    bx2, bt = 12, 55
    bh = h - 155
    filled  = int(bh * (score / 100.0))
    bar_col = (0, 232, 122) if score < 30 else (20, 176, 255) if score < 60 else (0, 40, 255)
    cv2.rectangle(frame, (bx2, bt), (bx2 + 10, bt + bh), (30, 30, 30), -1)
    if filled > 0:
        cv2.rectangle(frame, (bx2, bt + bh - filled), (bx2 + 10, bt + bh), bar_col, -1)
    cv2.rectangle(frame, (bx2, bt), (bx2 + 10, bt + bh), (55, 55, 55), 1)

    if alert_on:
        msg = "! DROWSINESS DETECTED — TAKE A BREAK !"
        ts  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
        tx  = (w - ts[0]) // 2
        ty  = h // 2 + 10
        cv2.putText(frame, msg, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 40, 255), 2, cv2.LINE_AA)

    if not face_found:
        msg = "NO FACE DETECTED"
        ts  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
        cv2.putText(frame, msg, ((w - ts[0]) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (55, 55, 55), 2, cv2.LINE_AA)

    return frame


def draw_landmarks(frame, landmarks, w, h, ear, mar):
    all_idx = LEFT_EYE + RIGHT_EYE + MOUTH + [NOSE_TIP, CHIN, LEFT_EAR_PT, RIGHT_EAR_PT]

    for idx in all_idx:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, (0, 200, 255), -1, cv2.LINE_AA)

    eye_col = (0, 200, 255) if ear >= Config.EAR_THRESHOLD else (0, 60, 255)
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cv2.rectangle(frame,
                      (min(xs) - 6, min(ys) - 6),
                      (max(xs) + 6, max(ys) + 6),
                      eye_col, 1, cv2.LINE_AA)

    mouth_col = (0, 60, 255) if mar > Config.MAR_THRESHOLD else (0, 200, 255)
    m_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH]
    m_xs, m_ys = [p[0] for p in m_pts], [p[1] for p in m_pts]
    cv2.rectangle(frame,
                  (min(m_xs) - 4, min(m_ys) - 4),
                  (max(m_xs) + 4, max(m_ys) + 4),
                  mouth_col, 1, cv2.LINE_AA)

    nose_pt = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
    chin_pt = (int(landmarks[CHIN].x     * w), int(landmarks[CHIN].y     * h))
    cv2.line(frame, nose_pt, chin_pt, (70, 70, 70), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════
#  ALERT
# ══════════════════════════════════════════════════════════════════

def trigger_alert(alert_count):
    print(f"\n{'='*60}")
    print(f"  DROWSINESS ALERT #{alert_count}")
    print(f"  Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"  >>> PULL OVER AND TAKE A BREAK <<<")
    print(f"{'='*60}\n")
    try:
        sys.stdout.write('\a')
        sys.stdout.flush()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  SESSION LOGGER
# ══════════════════════════════════════════════════════════════════

class SessionLogger:
    def __init__(self, log_file=Config.LOG_FILE):
        self.log_file = log_file
        self.sessions = self._load()

    def _load(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def save_session(self, summary):
        self.sessions.append(summary)
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
            print(f"[LOG] Session saved → {self.log_file}")
        except Exception as e:
            print(f"[LOG] Save failed: {e}")

    def print_history(self):
        if not self.sessions:
            print("\n[HISTORY] No sessions yet.")
            return
        print(f"\n{'='*60}")
        print(f"  TRIP HISTORY  ({len(self.sessions)} sessions)")
        print(f"{'='*60}")
        for i, s in enumerate(self.sessions[-10:], 1):
            print(f"  {i}. {s.get('date','?')}  "
                  f"Duration:{s.get('duration_min','?')}min  "
                  f"Alerts:{s.get('alert_count','?')}  "
                  f"Score:{s.get('safe_score','?')}%  "
                  f"Yawns:{s.get('yawn_count','?')}")
        print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
#  DRIVEALERT CLASS
# ══════════════════════════════════════════════════════════════════

class DriveAlert:
    def __init__(self):
        self.cfg = Config()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = mp.solutions.face_mesh.FaceMesh(
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )

        self.logger = SessionLogger()
        self.reset_session()

        self.fps_start = time.time()
        self.fps_count = 0
        self.fps       = 0.0

        self.show_landmarks = Config.SHOW_LANDMARKS

    def reset_session(self):
        self.ear_consec_count = 0
        self.blink_count      = 0
        self.yawn_count       = 0
        self.alert_count      = 0
        self.last_alert_time  = 0.0
        self.alert_active     = False

        self.eye_closed_log   = []
        self.blink_timestamps = []
        self.fatigue_history  = []

        self.yawn_active      = False
        self.session_start    = time.time()
        self.frame_count      = 0

        print("[SESSION] Counters reset")

    def _update_fps(self):
        self.fps_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.fps       = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_start = time.time()

    def _process_ear(self, ear):
        is_closed = ear < self.cfg.EAR_THRESHOLD
        self.eye_closed_log.append((time.time(), is_closed))

        cutoff = time.time() - self.cfg.PERCLOS_WINDOW_SEC
        self.eye_closed_log = [(t, c) for t, c in self.eye_closed_log if t >= cutoff]

        if is_closed:
            self.ear_consec_count += 1
        else:
            if self.ear_consec_count >= self.cfg.EAR_CONSEC_FRAMES:
                self.blink_count += 1
                self.blink_timestamps.append(time.time())
            self.ear_consec_count = 0

    def _process_mar(self, mar):
        if mar > self.cfg.MAR_THRESHOLD:
            self.yawn_active = True
        else:
            if self.yawn_active:
                self.yawn_active = False
                self.yawn_count += 1

    def _check_alert(self, score, ear):
        now   = time.time()
        drowsy = self.ear_consec_count >= self.cfg.EAR_DROWSY_FRAMES
        if drowsy and (now - self.last_alert_time) > self.cfg.ALERT_COOLDOWN_SEC:
            self.alert_active    = True
            self.last_alert_time = now
            self.alert_count    += 1
            threading.Thread(
                target = trigger_alert,
                args   = (self.alert_count,),
                daemon = True
            ).start()
        elif not drowsy and score < 60:
            self.alert_active = False

    def _safe_score(self):
        if self.alert_count == 0:
            return 100
        duration_min = max(1, (time.time() - self.session_start) / 60.0)
        alert_rate   = self.alert_count / duration_min
        score        = 100 - int(alert_rate * 40) - self.yawn_count * 2
        return max(0, min(100, score))

    def print_stats(self):
        duration = (time.time() - self.session_start) / 60.0
        print(f"\n{'─'*50}")
        print(f"  SESSION STATS")
        print(f"{'─'*50}")
        print(f"  Duration   : {duration:.1f} min")
        print(f"  Blinks     : {self.blink_count}")
        print(f"  Blink rate : {compute_blink_rate(self.blink_timestamps):.1f} /min")
        print(f"  Yawns      : {self.yawn_count}")
        print(f"  Alerts     : {self.alert_count}")
        print(f"  Safe score : {self._safe_score()}%")
        print(f"{'─'*50}\n")

    def save_and_show_history(self):
        duration_min = round((time.time() - self.session_start) / 60.0, 1)
        summary = {
            "date"        : datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration_min": duration_min,
            "blink_count" : self.blink_count,
            "yawn_count"  : self.yawn_count,
            "alert_count" : self.alert_count,
            "safe_score"  : self._safe_score(),
            "avg_fatigue" : round(float(np.mean(self.fatigue_history)), 1)
                            if self.fatigue_history else 0,
        }
        self.logger.save_session(summary)
        self.logger.print_history()


# ══════════════════════════════════════════════════════════════════
#  MAIN — runs when you press ▶ in VS Code or: python driver.py
# ══════════════════════════════════════════════════════════════════

def main():
    da  = DriveAlert()
    cfg = Config()

    # ── Open camera ──────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cfg.CAMERA_INDEX}.")
        print("        Try changing CAMERA_INDEX to 1 or 2 in Config above.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          cfg.FPS_TARGET)

    print("\n[DRIVEALERT] Camera opened!")
    print("  Q = quit  |  R = reset session  |  L = toggle landmarks\n")

    # ── Main loop ────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame — camera disconnected?")
            break

        da._update_fps()
        da.frame_count += 1

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results    = da.face_mesh.process(rgb)
        face_found = results.multi_face_landmarks is not None

        ear = mar = score = perclos = 0.0
        pitch = yaw = 0.0

        if face_found:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0
            mar       = mouth_aspect_ratio(landmarks, MOUTH, w, h)
            pitch, yaw = head_pose_angles(landmarks, w, h)

            da._process_ear(ear)
            da._process_mar(mar)

            perclos    = compute_perclos(da.eye_closed_log, cfg.PERCLOS_WINDOW_SEC)
            blink_rate = compute_blink_rate(da.blink_timestamps)
            score      = fatigue_score(ear, mar, perclos, pitch, cfg)
            da.fatigue_history.append(score)

            da._check_alert(score, ear)

            if da.show_landmarks:
                frame = draw_landmarks(frame, landmarks, w, h, ear, mar)
        else:
            da.ear_consec_count = 0
            da.alert_active     = False
            blink_rate          = compute_blink_rate(da.blink_timestamps)

        status, status_col = drowsiness_status(
            score, ear, da.ear_consec_count, cfg
        ) if face_found else ('NO FACE', (60, 60, 60))

        state = {
            'status'        : status,
            'status_color'  : status_col,
            'ear'           : ear,
            'mar'           : mar,
            'fatigue_score' : score,
            'perclos'       : perclos,
            'blink_rate'    : blink_rate,
            'yawn_count'    : da.yawn_count,
            'blink_count'   : da.blink_count,
            'pitch'         : pitch,
            'yaw'           : yaw,
            'fps'           : da.fps,
            'alert_active'  : da.alert_active,
            'face_found'    : face_found,
            'show_landmarks': da.show_landmarks,
        }

        frame = draw_hud(frame, state)
        cv2.imshow("DriveAlert", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[QUIT] Q pressed.")
            break
        elif key == ord('r'):
            da.reset_session()
        elif key == ord('l'):
            da.show_landmarks = not da.show_landmarks

    # ── Cleanup ───────────────────────────────────────────────────
    da.print_stats()
    da.save_and_show_history()
    cap.release()
    cv2.destroyAllWindows()
    print("[DRIVEALERT] Session ended.\n")


if __name__ == "__main__":
    main()
