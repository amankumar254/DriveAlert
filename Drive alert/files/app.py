"""
DRIVEALERT — app.py
Run:   python app.py
Open:  http://127.0.0.1:5000
"""

import threading
import base64
import time
import os
import re
import sys
import datetime

import cv2
import numpy as np
from flask import Flask, send_from_directory, jsonify, request, redirect
from flask_socketio import SocketIO, emit

# ── Import database ──
try:
    from database import (
        init_db, create_user, verify_user, get_user_by_id,
        create_token, validate_token, delete_token,
        save_driving_session, save_alert_event,
        get_user_sessions, get_user_stats, get_weekly_scores,
    )
    print("[OK] Database module loaded")
except ImportError as e:
    print(f"[ERROR] database.py not found: {e}")
    sys.exit(1)

# ── Import Driver.py ──
try:
    from Driver import (
        Config, DriveAlert,
        eye_aspect_ratio, mouth_aspect_ratio, head_pose_angles,
        compute_perclos, compute_blink_rate, fatigue_score,
        drowsiness_status, draw_hud, draw_landmarks,
        LEFT_EYE, RIGHT_EYE, MOUTH,
    )
    print("[OK] Driver.py imported")
except ImportError as e:
    print(f"[ERROR] Could not import Driver.py: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
#  FLASK + SOCKETIO
# ══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=BASE_DIR, static_folder=BASE_DIR)
app.config['SECRET_KEY'] = 'drivealert-secret-2025'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# ── Init DB on startup ──
init_db()

# ══════════════════════════════════════════════════════════════════
#  AUTH HELPER
# ══════════════════════════════════════════════════════════════════

def get_current_user():
    """Extract and validate token from Authorization header or cookie."""
    token = None
    auth  = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        token = auth[7:]
    if not token:
        token = request.cookies.get('da_token')
    if not token:
        token = request.args.get('token')
    return validate_token(token) if token else None

# ══════════════════════════════════════════════════════════════════
#  STATIC ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/login')
def login_page():
    return send_from_directory(BASE_DIR, 'login.html')

@app.route('/style.css')
def styles():
    return send_from_directory(BASE_DIR, 'style.css')

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'monitoring': monitoring})

# ══════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/auth/signup', methods=['POST'])
def signup():
    data     = request.get_json()
    username = (data.get('username') or '').strip()
    email    = (data.get('email')    or '').strip()
    password = (data.get('password') or '')
    fullname = (data.get('full_name')or '').strip()

    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'All fields are required'})
    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'})
    if len(username) < 3:
        return jsonify({'success': False, 'error': 'Username must be at least 3 characters'})

    result = create_user(username, email, password, fullname)
    if not result['success']:
        return jsonify(result)

    user  = get_user_by_id(result['user_id'])
    token = create_token(result['user_id'])
    print(f"[AUTH] New user registered: {username}")
    return jsonify({'success': True, 'token': token, 'user': {
        'id': user['id'], 'username': user['username'],
        'full_name': user['full_name'], 'email': user['email'],
    }})


@app.route('/auth/login', methods=['POST'])
def login():
    data     = request.get_json()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '')

    result = verify_user(username, password)
    if not result['success']:
        return jsonify(result)

    user  = result['user']
    token = create_token(user['id'])
    print(f"[AUTH] Login: {user['username']}")
    return jsonify({'success': True, 'token': token, 'user': {
        'id': user['id'], 'username': user['username'],
        'full_name': user['full_name'], 'email': user['email'],
        'total_trips': user['total_trips'], 'total_alerts': user['total_alerts'],
    }})


@app.route('/auth/logout', methods=['POST'])
def logout():
    auth  = request.headers.get('Authorization', '')
    token = auth[7:] if auth.startswith('Bearer ') else None
    if token:
        delete_token(token)
    return jsonify({'success': True})


@app.route('/auth/verify')
def verify():
    user = get_current_user()
    if not user:
        return jsonify({'valid': False})
    return jsonify({'valid': True, 'user': {
        'id': user['id'], 'username': user['username'],
        'full_name': user['full_name'], 'email': user['email'],
        'total_trips': user['total_trips'], 'total_alerts': user['total_alerts'],
    }})

# ══════════════════════════════════════════════════════════════════
#  USER DATA ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/api/history')
def api_history():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    sessions = get_user_sessions(user['id'])
    return jsonify({'sessions': sessions})


@app.route('/api/stats')
def api_stats():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    stats   = get_user_stats(user['id'])
    weekly  = get_weekly_scores(user['id'])
    return jsonify({'stats': stats, 'weekly': weekly})


@app.route('/api/profile')
def api_profile():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    stats = get_user_stats(user['id'])
    return jsonify({
        'user'  : {k: user[k] for k in ('id','username','full_name','email','created_at','last_login','total_trips','total_alerts')},
        'stats' : stats,
    })

# ══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════════════════════

try:
    da = DriveAlert()
    print("[OK] DriveAlert initialized")
except Exception as e:
    print(f"[ERROR] DriveAlert() failed: {e}")
    sys.exit(1)

monitoring          = False
use_client_camera   = False   # True when browser is sending frames (mobile)
_cap                = None
current_user_id     = None   # set when socket connects with token
current_session_id  = None   # set after saving session
session_ear_vals = []
session_mar_vals = []
session_perclos_vals  = []
session_headtilt_vals = []
pending_alerts   = []     # alert events queued during session

# ══════════════════════════════════════════════════════════════════
#  CAMERA LOOP
# ══════════════════════════════════════════════════════════════════

def camera_loop():
    global monitoring, _cap, da
    global session_ear_vals, session_mar_vals
    global session_perclos_vals, session_headtilt_vals

    try:
        _cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    except Exception as e:
        socketio.emit('error', {'message': f'Camera error: {e}'})
        monitoring = False
        return

    if not _cap.isOpened():
        socketio.emit('error', {'message': 'Camera not found.'})
        monitoring = False
        return

    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    _cap.set(cv2.CAP_PROP_FPS,          Config.FPS_TARGET)
    print(f"[CAMERA] Started {int(_cap.get(3))}x{int(_cap.get(4))}")

    while monitoring:
        try:
            ret, frame = _cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = da.face_mesh.process(rgb)

            ear = mar = pitch = yaw = perclos = blink_rate = f_score = 0.0
            status     = 'NO FACE'
            status_col = (60, 60, 60)
            face_found = False
            is_yawning = False

            if results.multi_face_landmarks:
                face_found = True
                lm         = results.multi_face_landmarks[0].landmark

                ear_l      = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
                ear_r      = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                ear        = round((ear_l + ear_r) / 2.0, 4)
                mar        = round(mouth_aspect_ratio(lm, MOUTH, w, h), 4)
                pitch, yaw = head_pose_angles(lm, w, h)

                da._process_ear(ear)
                da._process_mar(mar)
                is_yawning = da.yawn_active

                perclos    = compute_perclos(da.eye_closed_log, Config.PERCLOS_WINDOW_SEC)
                blink_rate = compute_blink_rate(da.blink_timestamps)
                f_score    = fatigue_score(ear, mar, perclos, pitch)

                da.fatigue_history.append(f_score)

                # Collect for session averages
                session_ear_vals.append(ear)
                session_mar_vals.append(mar)
                session_perclos_vals.append(perclos)
                session_headtilt_vals.append(abs(pitch))

                status, status_col = drowsiness_status(f_score, ear, da.ear_consec_count)
                da._check_alert(f_score, ear)

                if Config.SHOW_LANDMARKS:
                    frame = draw_landmarks(frame, lm, w, h, ear, mar)
            else:
                da.eye_closed_log.append((time.time(), False))
                da.ear_consec_count = 0
                da.alert_active     = False

            hud_state = {
                'status': status, 'status_color': status_col,
                'ear': ear, 'mar': mar, 'fatigue_score': f_score,
                'perclos': perclos, 'blink_rate': blink_rate,
                'blink_count': da.blink_count, 'yawn_count': da.yawn_count,
                'alert_count': da.alert_count, 'pitch': pitch, 'yaw': yaw,
                'fps': da.fps, 'face_found': face_found,
                'alert_active': da.alert_active, 'show_landmarks': Config.SHOW_LANDMARKS,
            }
            frame     = draw_hud(frame, hud_state)
            da._update_fps()
            alertness = max(0, min(100, int(100 - f_score))) if face_found else 0

            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64_frame  = base64.b64encode(buffer).decode('utf-8') if ok else ''

            socketio.emit('frame_data', {
                'image': b64_frame, 'ear': ear, 'mar': mar,
                'head_tilt': pitch, 'yaw': yaw,
                'fatigue_score': round(f_score, 1),
                'perclos': round(perclos, 4), 'alertness': alertness,
                'blinks': da.blink_count, 'yawns': da.yawn_count,
                'alert_count': da.alert_count, 'status': status,
                'face_found': face_found, 'yawning': is_yawning,
                'alert': da.alert_active and status == 'DROWSY',
            })

        except Exception as e:
            print(f"[CAMERA LOOP ERROR] {e}")

        time.sleep(0.04)

    if _cap:
        _cap.release()
        _cap = None
    print("[CAMERA] Released")

# ══════════════════════════════════════════════════════════════════
#  SOCKETIO EVENTS
# ══════════════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    print("[WS] Frontend connected")
    emit('server_ready', {'message': 'DriveAlert backend ready'})


@socketio.on('disconnect')
def on_disconnect():
    global monitoring
    monitoring = False
    print("[WS] Frontend disconnected")


@socketio.on('authenticate')
def handle_auth(data):
    """Client sends token after connect so we know which user is driving."""
    global current_user_id
    token = data.get('token')
    user  = validate_token(token) if token else None
    if user:
        current_user_id = user['id']
        emit('auth_ok', {
            'username'    : user['username'],
            'full_name'   : user['full_name'],
            'total_trips' : user['total_trips'],
            'total_alerts': user['total_alerts'],
        })
        print(f"[WS] Authenticated: {user['username']}")
    else:
        current_user_id = None
        emit('auth_fail', {'message': 'Invalid or expired token'})


@socketio.on('start_monitoring')
def handle_start(data=None):
    global monitoring, da, pending_alerts, use_client_camera
    global session_ear_vals, session_mar_vals
    global session_perclos_vals, session_headtilt_vals

    if monitoring:
        return
    da.reset_session()
    monitoring            = True
    pending_alerts        = []
    session_ear_vals      = []
    session_mar_vals      = []
    session_perclos_vals  = []
    session_headtilt_vals = []

    # If the browser will send frames (mobile), skip the server camera loop
    use_client_camera = bool((data or {}).get('client_camera', False))

    if not use_client_camera:
        threading.Thread(target=camera_loop, daemon=True).start()
    emit('monitoring_started', {'message': 'Camera starting...', 'client_camera': use_client_camera})
    print(f"[WS] Monitoring started (client_camera={use_client_camera})")


@socketio.on('client_frame')
def handle_client_frame(data):
    """
    Receive a raw JPEG frame (base64) from the browser camera (mobile).
    Run the same drowsiness pipeline and emit frame_data back.
    """
    global monitoring, da, session_ear_vals, session_mar_vals
    global session_perclos_vals, session_headtilt_vals

    if not monitoring:
        return

    try:
        # Decode base64 → numpy frame
        img_bytes = base64.b64decode(data.get('image', ''))
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = da.face_mesh.process(rgb)

        ear = mar = pitch = yaw = perclos = blink_rate = f_score = 0.0
        status     = 'NO FACE'
        status_col = (60, 60, 60)
        face_found = False
        is_yawning = False

        if results.multi_face_landmarks:
            face_found = True
            lm         = results.multi_face_landmarks[0].landmark

            ear_l      = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            ear_r      = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear        = round((ear_l + ear_r) / 2.0, 4)
            mar        = round(mouth_aspect_ratio(lm, MOUTH, w, h), 4)
            pitch, yaw = head_pose_angles(lm, w, h)

            da._process_ear(ear)
            da._process_mar(mar)
            is_yawning = da.yawn_active

            perclos    = compute_perclos(da.eye_closed_log, Config.PERCLOS_WINDOW_SEC)
            blink_rate = compute_blink_rate(da.blink_timestamps)
            f_score    = fatigue_score(ear, mar, perclos, pitch)

            da.fatigue_history.append(f_score)

            session_ear_vals.append(ear)
            session_mar_vals.append(mar)
            session_perclos_vals.append(perclos)
            session_headtilt_vals.append(abs(pitch))

            status, status_col = drowsiness_status(f_score, ear, da.ear_consec_count)
            da._check_alert(f_score, ear)

            if Config.SHOW_LANDMARKS:
                frame = draw_landmarks(frame, lm, w, h, ear, mar)
        else:
            da.eye_closed_log.append((time.time(), False))
            da.ear_consec_count = 0
            da.alert_active     = False

        hud_state = {
            'status': status, 'status_color': status_col,
            'ear': ear, 'mar': mar, 'fatigue_score': f_score,
            'perclos': perclos, 'blink_rate': blink_rate,
            'blink_count': da.blink_count, 'yawn_count': da.yawn_count,
            'alert_count': da.alert_count, 'pitch': pitch, 'yaw': yaw,
            'fps': da.fps, 'face_found': face_found,
            'alert_active': da.alert_active, 'show_landmarks': Config.SHOW_LANDMARKS,
        }
        frame     = draw_hud(frame, hud_state)
        da._update_fps()
        alertness = max(0, min(100, int(100 - f_score))) if face_found else 0

        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        b64_frame  = base64.b64encode(buffer).decode('utf-8') if ok else ''

        emit('frame_data', {
            'image': b64_frame, 'ear': ear, 'mar': mar,
            'head_tilt': pitch, 'yaw': yaw,
            'fatigue_score': round(f_score, 1),
            'perclos': round(perclos, 4), 'alertness': alertness,
            'blinks': da.blink_count, 'yawns': da.yawn_count,
            'alert_count': da.alert_count, 'status': status,
            'face_found': face_found, 'yawning': is_yawning,
            'alert': da.alert_active and status == 'DROWSY',
        })

    except Exception as e:
        print(f"[CLIENT FRAME ERROR] {e}")



def handle_stop():
    global monitoring, da, current_user_id, current_session_id
    global session_ear_vals, session_mar_vals
    global session_perclos_vals, session_headtilt_vals

    monitoring = False
    time.sleep(0.2)



    # Build session summary
    summary = {
        'date'         : datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        'duration_min' : round((time.time() - da.session_start) / 60.0, 1),
        'blink_count'  : da.blink_count,
        'yawn_count'   : da.yawn_count,
        'alert_count'  : da.alert_count,
        'safe_score'   : da._safe_score(),
        'avg_fatigue'  : round(float(np.mean(da.fatigue_history)), 1) if da.fatigue_history else 0,
        'max_fatigue'  : round(float(max(da.fatigue_history)),      1) if da.fatigue_history else 0,
        'avg_ear'      : round(float(np.mean(session_ear_vals)),      3) if session_ear_vals      else 0,
        'avg_mar'      : round(float(np.mean(session_mar_vals)),      3) if session_mar_vals      else 0,
        'perclos_avg'  : round(float(np.mean(session_perclos_vals)),  4) if session_perclos_vals  else 0,
        'head_tilt_avg': round(float(np.mean(session_headtilt_vals)), 1) if session_headtilt_vals else 0,
    }

    # Save to DB if user is logged in
    if current_user_id:
        try:
            current_session_id = save_driving_session(current_user_id, summary)
            # Save queued alert events
            for evt in pending_alerts:
                save_alert_event(current_session_id, current_user_id, evt)
            print(f"[DB] Session saved for user {current_user_id}, id={current_session_id}")
        except Exception as e:
            print(f"[DB] Save error: {e}")

    emit('monitoring_stopped', {**summary, 'saved_to_db': current_user_id is not None})
    print(f"[WS] Monitoring stopped — score: {summary['safe_score']}%")


@socketio.on('dismiss_alert')
def handle_dismiss():
    da.alert_active    = False
    da.last_alert_time = time.time()
    emit('alert_dismissed')


@socketio.on('log_alert_event')
def handle_alert_event(data):
    """Frontend sends alert details to be stored."""
    global pending_alerts
    pending_alerts.append(data)


@socketio.on('update_settings')
def handle_settings(data):
    if 'ear_threshold' in data:
        Config.EAR_THRESHOLD     = float(data['ear_threshold'])
    if 'mar_threshold' in data:
        Config.MAR_THRESHOLD     = float(data['mar_threshold'])
    if 'closed_frames' in data:
        Config.EAR_DROWSY_FRAMES = int(data['closed_frames'])
    emit('settings_updated', {
        'ear_threshold': Config.EAR_THRESHOLD,
        'mar_threshold': Config.MAR_THRESHOLD,
        'closed_frames': Config.EAR_DROWSY_FRAMES,
    })

# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  DRIVEALERT — Running")
    print("  PC:    http://127.0.0.1:5000")
    print("  Login: http://127.0.0.1:5000/login")
    print("=" * 60)
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
    )
