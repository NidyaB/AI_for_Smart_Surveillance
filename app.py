import cv2
import time
import math
import numpy as np
import json
import os
import threading
from collections import deque
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from flask import Flask, Response, render_template, session, redirect, url_for, request, jsonify
from flask_cors import CORS
from functools import wraps
from datetime import datetime

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "eyeq_super_secure_key")
app.config['TEMPLATES_AUTO_RELOAD'] = False  # Disable in production
CORS(app)  # Allow React / Flutter to access video streams

try:
    cred = credentials.Certificate("../serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Connected to Firebase Firestore")
except Exception as e:
    db = None
    print(f"⚠️ Firebase initialization failed: {e}")

# --- CLOUDINARY CONFIG ---
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )
    # Test the configuration
    cloudinary.api.ping()
    cloudinary_enabled = True
    print("✅ Cloudinary configured successfully")
except Exception as e:
    cloudinary_enabled = False
    print(f"⚠️ Cloudinary configuration failed: {e}")
    print("📝 Image uploads will be disabled. Configure CLOUDINARY_* variables in .env file")

# --- ML & DETECTION SETUP ---
# Load YOLO model ONLY ONCE
MODEL_PATH = "../yolo11n-pose.pt" # Pose model for fall/fight
model = YOLO(MODEL_PATH)

# Alert configs
HORIZONTAL_RATIO = 1.3
FALL_VELOCITY_THRESH = 12
FIGHT_VELOCITY_THRESH = 0.25
CLOSE_DISTANCE_RATIO = 1.2
ALERT_COOLDOWN = 5  # 5s cooldown per zone-event (Updated per requirements)

# Mapping multi-zone feeds
ZONE_VIDEOS = {
    "entrance": "../videos/boxing.mp4",
    "reception": "../videos/falling8.mp4",
    "corridor": "../videos/sleep.mp4",
    "parking": "../videos/walking.mp4"
}

# State variables
last_alert_time = {}
person_loiter_tracker = {k: {} for k in ZONE_VIDEOS}  # Tracker state per zone
person_buffers = {k: {} for k in ZONE_VIDEOS} # Impact/Velocity state per zone
active_frames = {k: None for k in ZONE_VIDEOS}
active_persons_data = {k: {} for k in ZONE_VIDEOS}

def upload_alert_background(event_type, zone, frame_snapshot=None):
    """ Uploads image to Cloudinary (if applicable) and stores doc in Firestore """
    if db is None: return
        
    image_url = None
    
    # Only upload images for INTRUSION or LOITERING
    if event_type in ["INTRUSION", "LOITERING"] and frame_snapshot is not None and cloudinary_enabled:
        try:
            _, buffer = cv2.imencode('.jpg', frame_snapshot, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            upload_result = cloudinary.uploader.upload(
                buffer.tobytes(), 
                folder="surveillance_alerts",
                resource_type="image"
            )
            image_url = upload_result.get("secure_url")
            print(f"☁️ Uploaded image to Cloudinary: {image_url}")
        except Exception as e:
            # Fix error requirement: add try-except catching upload failure gracefully
            print(f"❌ Failed to upload error: {e}")
            image_url = None
    elif event_type in ["INTRUSION", "LOITERING"] and frame_snapshot is not None and not cloudinary_enabled:
        print("📷 Cloudinary not configured - skipping image upload")

    try:
        doc_ref = db.collection('alerts').document()
        alert_data = {
            'timestamp': firestore.SERVER_TIMESTAMP,
            'event': event_type,
            'zone_name': zone,
            'status': 'active',
            'confidence': 0.95
        }
        if image_url: alert_data['image_url'] = image_url
        doc_ref.set(alert_data)
        print(f"📄 Saved {event_type} alert ({zone}) to Firebase successfully.")
    except Exception as e:
        print(f"❌ Failed to save alert to Firebase: {e}")

def trigger_alert(event_type, zone_name, frame):
    """ Handle 5-second cooldown and trigger background upload """
    key = f"{event_type}_{zone_name}"
    current_time = time.time()
    
    if key in last_alert_time and (current_time - last_alert_time[key] < ALERT_COOLDOWN):
        return # Skip due to 5 second cooldown
        
    last_alert_time[key] = current_time
    print(f"🚨 ALERT TRIGGERED: {event_type} in {zone_name}")
    
    threading.Thread(target=upload_alert_background, args=(event_type, zone_name, frame.copy())).start()

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def process_zone_results(zone_id, result, frame):
    """ Post-processing tracking logic applied per zone result """
    active_persons = {}
    
    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().numpy()
        kpts_all = result.keypoints.xy.cpu().numpy()

        for box, pid, kpts in zip(boxes, ids, kpts_all):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if pid not in person_buffers[zone_id]:
                person_buffers[zone_id][pid] = {"center_history": deque(maxlen=10), "prev_hands": None, "impact_detected": False, "was_vertical": True}

            nose_y = kpts[0][1]
            avg_hip_y = (kpts[11][1] + kpts[12][1]) / 2 if kpts[11][1] > 0 else 0
            vel_y = cy - person_buffers[zone_id][pid]["center_history"][-1][1] if person_buffers[zone_id][pid]["center_history"] else 0
            person_buffers[zone_id][pid]["center_history"].append((cx, cy))

            ratio = w / h if h > 0 else 0
            if ratio < 0.8: person_buffers[zone_id][pid]["was_vertical"] = True
            
            if (vel_y > FALL_VELOCITY_THRESH or (nose_y > avg_hip_y > 0)) and person_buffers[zone_id][pid]["was_vertical"]:
                person_buffers[zone_id][pid]["impact_detected"] = True
                person_buffers[zone_id][pid]["was_vertical"] = False

            is_fall = (ratio > HORIZONTAL_RATIO and person_buffers[zone_id][pid]["impact_detected"])
            
            speed = 0
            if person_buffers[zone_id][pid]["prev_hands"] is not None and h > 0:
                speed = (dist(kpts[9], person_buffers[zone_id][pid]["prev_hands"][0]) + dist(kpts[10], person_buffers[zone_id][pid]["prev_hands"][1])) / h
            person_buffers[zone_id][pid]["prev_hands"] = (kpts[9], kpts[10])

            active_persons[pid] = {"box": (x1, y1, x2, y2), "center": (cx, cy), "w": w, "is_fall": is_fall, "speed": speed}

            # Intrusion Logic
            # If the zone is strictly "restricted", any entry is an Intrusion
            if zone_id == "restricted":
                if pid not in person_loiter_tracker[zone_id]:
                    person_loiter_tracker[zone_id][pid] = {"enter_time": time.time()}
                    trigger_alert("INTRUSION", zone_id, frame)
            else:
                # Loitering logic for normal zones
                if pid not in person_loiter_tracker[zone_id]:
                    person_loiter_tracker[zone_id][pid] = {"enter_time": time.time()}
                else:
                    if time.time() - person_loiter_tracker[zone_id][pid]["enter_time"] > 60:
                        trigger_alert("LOITERING", zone_id, frame)
                        person_loiter_tracker[zone_id][pid]["enter_time"] = time.time()

            if is_fall:
                trigger_alert("FALL", zone_id, frame)

        pids = list(active_persons.keys())
        is_fighting = False
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p1, p2 = active_persons[pids[i]], active_persons[pids[j]]
                if dist(p1["center"], p2["center"]) < (CLOSE_DISTANCE_RATIO * max(p1["w"], p2["w"])):
                    if (p1["speed"] + p2["speed"]) > FIGHT_VELOCITY_THRESH:
                        is_fighting = True

        if is_fighting:
            trigger_alert("FIGHT", zone_id, frame)

        # Draw Overlay
        for pid, data in active_persons.items():
            x1, y1, x2, y2 = data["box"]
            color = (0, 0, 255) if (data["is_fall"] or is_fighting) else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{zone_id.upper()} ID:{pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    # Purge old loitering configs for people no longer present
    to_del = [p for p in person_loiter_tracker[zone_id] if p not in active_persons]
    for p in to_del: del person_loiter_tracker[zone_id][p]

    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    active_frames[zone_id] = buffer.tobytes()

def multi_zone_detection_loop():
    """ Optimized Single-Thread Batch Loop """
    zone_keys = list(ZONE_VIDEOS.keys())
    caps = [cv2.VideoCapture(ZONE_VIDEOS[k]) for k in zone_keys]
    
    while True:
        frames = []
        valid_indices = []
        
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                ret, frame = cap.read()
            if ret:
                frames.append(frame)
                valid_indices.append(i)
                
        if not frames:
            time.sleep(0.5)
            continue
            
        # Run batched inference maintaining 1 model to eliminate threading collisions!
        results = model.track(frames, persist=True, verbose=False)
        
        for idx, result in enumerate(results):
            zone_id = zone_keys[valid_indices[idx]]
            process_zone_results(zone_id, result, frames[idx])
            
        time.sleep(0.01)

def generate_frames(zone_id):
    while True:
        frame_bytes = active_frames.get(zone_id)
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

ZONE_NAMES = {
    "entrance": "Entrance",
    "reception": "Reception Area",
    "corridor": "Corridor",
    "parking": "Parking Area"
}

# --- AUTHENTICATION MIDDLEWARE ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- WEB UI ROUTES ---
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid administrative credentials")
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/live')
@login_required
def live():
    return render_template('live.html')
    
@app.route('/alerts')
@login_required
def alerts():
    return render_template('alerts.html')

@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/help')
@login_required
def help_center():
    return render_template('help.html')

@app.route('/zone/<zone_id>')
@login_required
def zone_detail(zone_id):
    if zone_id not in ZONE_VIDEOS:
        return "Invalid Zone", 404
    return render_template('zone.html', zone_id=zone_id, zone_name=ZONE_NAMES.get(zone_id, zone_id))

@app.route('/video_feed/<zone_id>')
def video_feed(zone_id):
    if zone_id not in ZONE_VIDEOS:
        return "Invalid Zone", 404
    return Response(generate_frames(zone_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- DATA APIs ---
def serialize_firestore_doc(doc):
    data = doc.to_dict()
    data['id'] = doc.id
    if 'timestamp' in data and data['timestamp']:
        # Firestore Timestamp -> Standard ISO String
        data['timestamp'] = data['timestamp'].isoformat()
    return data

@app.route('/api/alerts')
@login_required
def api_alerts():
    if db is None: return jsonify([])
    try:
        zone_filter = request.args.get('zone')
        query = db.collection('alerts').order_by('timestamp', direction=firestore.Query.DESCENDING)
        if zone_filter and zone_filter != 'all':
            query = query.where('zone_name', '==', zone_filter)
            
        docs = query.limit(20).stream()
        return jsonify([serialize_firestore_doc(doc) for doc in docs])
    except Exception as e:
        print(f"API Fetch Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs')
@login_required
def api_logs():
    if db is None: return jsonify([])
    try:
        # For historical logs, pull a larger batch
        docs = db.collection('alerts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100).stream()
        return jsonify([serialize_firestore_doc(doc) for doc in docs])
    except Exception as e:
        print(f"API Fetch Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def api_settings():
    if request.method == 'POST':
        try:
            settings_data = request.get_json()
            # In a real app, you'd save this to a database
            # For now, we'll just return success
            print(f"Settings updated: {settings_data}")
            return jsonify({"success": True, "message": "Settings saved successfully"})
        except Exception as e:
            print(f"Settings save error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    else:
        # Return current settings (placeholder for now)
        return jsonify({
            "autoDetection": True,
            "recordingQuality": "1080p",
            "motionSensitivity": "medium",
            "emailAlerts": True,
            "pushNotifications": True,
            "alertSound": False,
            "darkMode": True,
            "autoBackup": True,
            "dataRetention": "30"
        })

@app.route('/api/profile', methods=['GET', 'POST'])
@login_required
def api_profile():
    if request.method == 'POST':
        try:
            profile_data = request.get_json()
            # In a real app, you'd validate password and save to database
            # For now, we'll just return success
            print(f"Profile updated: {profile_data}")
            return jsonify({"success": True, "message": "Profile updated successfully"})
        except Exception as e:
            print(f"Profile save error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    else:
        # Return current profile (placeholder for now)
        return jsonify({
            "name": "Administrator",
            "email": "admin@eyeq.com",
            "role": "System Owner",
            "phone": "+1 (555) 123-4567",
            "bio": "Security system administrator with 5+ years of experience in surveillance and system management."
        })

@app.route('/api/system/troubleshoot')
@login_required
def api_troubleshoot():
    try:
        # Simulate system diagnostics
        diagnostics = {
            "camera_status": "All cameras online",
            "network_connectivity": "Good",
            "storage_space": "2.4 GB used of 50 GB",
            "cpu_usage": "15%",
            "memory_usage": "45%",
            "last_backup": "2026-04-04 14:30:00"
        }
        return jsonify({"success": True, "diagnostics": diagnostics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/system/backup')
@login_required
def api_backup():
    try:
        # Simulate backup process
        import time
        time.sleep(1)  # Simulate backup time
        return jsonify({"success": True, "message": "System backup completed successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/system/health')
@login_required
def api_system_health():
    try:
        # Simulate system health check
        health_data = {
            "overall_status": "Healthy",
            "cameras_online": 4,
            "total_cameras": 4,
            "uptime": "7 days, 14 hours",
            "alerts_today": 12,
            "storage_used": "2.4 GB",
            "last_maintenance": "2026-04-01"
        }
        return jsonify({"success": True, "health": health_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable for production deployment
    port = int(os.getenv("PORT", 5000))

    print(f"\n🌐 Starting Multi-Zone Backend on port {port}...")
    t = threading.Thread(target=multi_zone_detection_loop)
    t.daemon = True
    t.start()

    # Production-ready server configuration
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
