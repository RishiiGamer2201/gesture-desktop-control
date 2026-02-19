"""
Web Application for Jarvis + Gesture Control - FINAL VERSION
Includes process management for Jarvis and Gesture Desktop Control
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import json
import os
import subprocess
import pickle
from datetime import datetime
from threading import Thread, Lock
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import webbrowser
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'jarvis-gesture-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration  
CUSTOM_GESTURES_FILE = 'custom_gestures.json'
TRAINING_DATA_DIR = 'training_data'
MODEL_PATH = 'models/gesture_knn_model_updated.pkl'
CUSTOM_DATA_FILE = 'training_data/custom_gesture_data.csv'

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables with locks
camera_lock = Lock()
collection_lock = Lock()
current_frame = None
current_landmarks = None
collection_active = False
current_gesture_samples = []

# Process management
jarvis_process = None
gesture_process = None
process_lock = Lock()

# Load custom gestures
def load_custom_gestures():
    if os.path.exists(CUSTOM_GESTURES_FILE):
        with open(CUSTOM_GESTURES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_custom_gestures(gestures):
    with open(CUSTOM_GESTURES_FILE, 'w') as f:
        json.dump(gestures, indent=2, fp=f)

# Default gestures
DEFAULT_GESTURES = {
    0: {"name": "PALM", "action": "halt", "description": "Halt cursor"},
    1: {"name": "V_SIGN", "action": "move", "description": "Move cursor"},
    2: {"name": "MIDDLE_ONLY", "action": "left_click", "description": "Left click"},
    3: {"name": "INDEX_ONLY", "action": "right_click", "description": "Right click"},
    4: {"name": "JOINED_FINGERS", "action": "double_click", "description": "Double click"},
    5: {"name": "PINKY_ONLY", "action": "scroll_down", "description": "Scroll down / Volume down"},
    6: {"name": "PINKY_THUMB", "action": "scroll_up", "description": "Scroll up / Volume up"},
    7: {"name": "THUMBS_UP", "action": "volume_toggle", "description": "Volume mode toggle"},
    8: {"name": "THUMBS_DOWN", "action": "open_spotify", "description": "Open Spotify"},
    9: {"name": "ROCK_SIGN", "action": "reserved", "description": "Reserved"}
}

# Jarvis commands
JARVIS_COMMANDS = [
    {"command": "Jarvis hello", "description": "Greet Jarvis"},
    {"command": "Jarvis what's the time", "description": "Get current time"},
    {"command": "Jarvis search [query]", "description": "Google search"},
    {"command": "Jarvis youtube [query]", "description": "YouTube search"},
    {"command": "Jarvis wikipedia [query]", "description": "Wikipedia search"},
    {"command": "Jarvis location [place]", "description": "Find location on maps"},
    {"command": "Jarvis launch gesture", "description": "Start gesture control"},
    {"command": "Jarvis stop gesture", "description": "Stop gesture control"},
    {"command": "Jarvis volume up/down", "description": "Control volume"},
    {"command": "Jarvis screenshot", "description": "Take screenshot"},
    {"command": "Jarvis help", "description": "Show all commands"},
    {"command": "Jarvis exit", "description": "Quit Jarvis"}
]

# Process Management Functions
def is_process_running(process):
    if process is None:
        return False
    return process.poll() is None

def get_jarvis_status():
    with process_lock:
        return is_process_running(jarvis_process)

def get_gesture_status():
    with process_lock:
        return is_process_running(gesture_process)

def start_jarvis():
    global jarvis_process
    
    with process_lock:
        if is_process_running(jarvis_process):
            return {'status': 'error', 'message': 'Jarvis is already running'}
        
        try:
            jarvis_process = subprocess.Popen(
                [sys.executable, 'jarvis.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            time.sleep(2)
            
            if is_process_running(jarvis_process):
                return {'status': 'success', 'message': 'Jarvis started successfully'}
            else:
                return {'status': 'error', 'message': 'Jarvis failed to start'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

def stop_jarvis():
    global jarvis_process
    
    with process_lock:
        if not is_process_running(jarvis_process):
            return {'status': 'error', 'message': 'Jarvis is not running'}
        
        try:
            jarvis_process.terminate()
            jarvis_process.wait(timeout=5)
            jarvis_process = None
            return {'status': 'success', 'message': 'Jarvis stopped'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

def start_gesture_control():
    global gesture_process
    
    with process_lock:
        if is_process_running(gesture_process):
            return {'status': 'error', 'message': 'Gesture control is already running'}
        
        try:
            gesture_process = subprocess.Popen(
                [sys.executable, 'main_knn_updated.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            time.sleep(2)
            
            if is_process_running(gesture_process):
                return {'status': 'success', 'message': 'Gesture control started successfully'}
            else:
                return {'status': 'error', 'message': 'Gesture control failed to start'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

def stop_gesture_control():
    global gesture_process
    
    with process_lock:
        if not is_process_running(gesture_process):
            return {'status': 'error', 'message': 'Gesture control is not running'}
        
        try:
            gesture_process.terminate()
            gesture_process.wait(timeout=5)
            gesture_process = None
            return {'status': 'success', 'message': 'Gesture control stopped'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/jarvis')
def jarvis_page():
    jarvis_running = get_jarvis_status()
    return render_template('jarvis.html', commands=JARVIS_COMMANDS, jarvis_running=jarvis_running)

@app.route('/gestures')
def gestures_page():
    custom_gestures = load_custom_gestures()
    all_gestures = {**DEFAULT_GESTURES}
    
    for gesture_id, gesture in custom_gestures.items():
        all_gestures[int(gesture_id)] = gesture
    
    gesture_running = get_gesture_status()
    return render_template('gestures.html', gestures=all_gestures, gesture_running=gesture_running)

@app.route('/add-gesture')
def add_gesture_page():
    return render_template('add_gesture.html')

@app.route('/api/gestures')
def get_gestures():
    custom_gestures = load_custom_gestures()
    all_gestures = {**DEFAULT_GESTURES}
    
    for gesture_id, gesture in custom_gestures.items():
        all_gestures[int(gesture_id)] = gesture
    
    return jsonify(all_gestures)

# Process control routes
@app.route('/api/jarvis/start', methods=['POST'])
def api_start_jarvis():
    return jsonify(start_jarvis())

@app.route('/api/jarvis/stop', methods=['POST'])
def api_stop_jarvis():
    return jsonify(stop_jarvis())

@app.route('/api/jarvis/status', methods=['GET'])
def api_jarvis_status():
    return jsonify({'running': get_jarvis_status()})

@app.route('/api/gesture/start', methods=['POST'])
def api_start_gesture():
    return jsonify(start_gesture_control())

@app.route('/api/gesture/stop', methods=['POST'])
def api_stop_gesture():
    return jsonify(stop_gesture_control())

@app.route('/api/gesture/status', methods=['GET'])
def api_gesture_status():
    return jsonify({'running': get_gesture_status()})

# Camera feed
def generate_frames():
    global current_frame, current_landmarks
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            with camera_lock:
                current_frame = frame.copy()
                if results.multi_hand_landmarks:
                    current_landmarks = results.multi_hand_landmarks[0]
                else:
                    current_landmarks = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            cv2.putText(frame, "Gesture Collection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if collection_active:
                cv2.putText(frame, "RECORDING", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Samples: {len(current_gesture_samples)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)
            
    finally:
        cap.release()
        hands.close()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket events
@socketio.on('start_camera')
def handle_start_camera():
    emit('camera_status', {'status': 'started'})

@socketio.on('stop_camera')
def handle_stop_camera():
    global collection_active
    collection_active = False
    emit('camera_status', {'status': 'stopped'})

@socketio.on('start_collection')
def handle_start_collection():
    global collection_active, current_gesture_samples
    with collection_lock:
        collection_active = True
        current_gesture_samples = []
    emit('collection_status', {'status': 'started', 'samples': 0})

@socketio.on('capture_sample')
def handle_capture_sample():
    global current_landmarks, collection_active, current_gesture_samples
    
    if not collection_active:
        return
    
    with camera_lock:
        if current_landmarks is not None:
            wrist = current_landmarks.landmark[0]
            features = []
            for landmark in current_landmarks.landmark:
                features.append(landmark.x - wrist.x)
                features.append(landmark.y - wrist.y)
            
            with collection_lock:
                current_gesture_samples.append(features)
                sample_count = len(current_gesture_samples)
            
            socketio.emit('collection_status', {
                'status': 'capturing',
                'samples': sample_count
            })

@socketio.on('stop_collection')
def handle_stop_collection(data):
    global collection_active, current_gesture_samples
    
    with collection_lock:
        collection_active = False
        samples = current_gesture_samples.copy()
    
    gesture_name = data.get('name', 'Custom Gesture')
    action_type = data.get('action_type', 'open_app')
    action_value = data.get('action_value', '')
    
    if len(samples) < 20:
        emit('training_log', {
            'message': f'❌ Not enough samples! Need at least 20, got {len(samples)}',
            'type': 'error'
        })
        return
    
    custom_gestures = load_custom_gestures()
    if custom_gestures:
        next_id = max([int(k) for k in custom_gestures.keys()]) + 1
    else:
        next_id = 10
    
    custom_gestures[str(next_id)] = {
        'name': gesture_name,
        'action': action_type,
        'action_value': action_value,
        'description': f'{action_type}: {action_value}',
        'samples': len(samples),
        'created': datetime.now().isoformat()
    }
    save_custom_gestures(custom_gestures)
    
    emit('training_log', {
        'message': f'✓ Saved {len(samples)} samples for gesture ID {next_id}',
        'type': 'success'
    })
    
    save_training_data(samples, next_id)
    retrain_model()
    
    emit('collection_complete', {
        'gesture_id': next_id,
        'gesture_name': gesture_name,
        'samples': len(samples)
    })

def save_training_data(samples, label):
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    data = np.array(samples)
    labels = np.full((len(samples), 1), label)
    dataset = np.hstack((data, labels))
    
    if os.path.exists(CUSTOM_DATA_FILE):
        existing_data = np.loadtxt(CUSTOM_DATA_FILE, delimiter=',')
        if existing_data.ndim == 1:
            existing_data = existing_data.reshape(1, -1)
        combined = np.vstack((existing_data, dataset))
        np.savetxt(CUSTOM_DATA_FILE, combined, delimiter=',')
    else:
        np.savetxt(CUSTOM_DATA_FILE, dataset, delimiter=',')
    
    socketio.emit('training_log', {
        'message': f'✓ Saved training data to {CUSTOM_DATA_FILE}',
        'type': 'success'
    })

def retrain_model():
    socketio.emit('training_log', {
        'message': '🔄 Starting model training...',
        'type': 'info'
    })
    
    try:
        import glob
        csv_files = glob.glob(os.path.join(TRAINING_DATA_DIR, "*.csv"))
        
        if not csv_files:
            socketio.emit('training_log', {
                'message': '❌ No training data found!',
                'type': 'error'
            })
            return
        
        socketio.emit('training_log', {
            'message': f'📂 Loading {len(csv_files)} data file(s)...',
            'type': 'info'
        })
        
        all_data = []
        for f in csv_files:
            data = np.loadtxt(f, delimiter=',')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            all_data.append(data)
        
        combined_data = np.vstack(all_data)
        X = combined_data[:, :-1]
        y = combined_data[:, -1]
        
        socketio.emit('training_log', {
            'message': f'✓ Loaded {len(X)} total samples',
            'type': 'success'
        })
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        socketio.emit('training_log', {
            'message': f'📊 Train: {len(X_train)} | Test: {len(X_test)}',
            'type': 'info'
        })
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        socketio.emit('training_log', {
            'message': '🧠 Model trained successfully!',
            'type': 'success'
        })
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        socketio.emit('training_log', {
            'message': f'🎯 Accuracy: {accuracy:.2f}%',
            'type': 'success'
        })
        
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        socketio.emit('training_log', {
            'message': f'💾 Model saved to {MODEL_PATH}',
            'type': 'success'
        })
        
        socketio.emit('training_complete', {
            'accuracy': accuracy,
            'total_samples': len(X)
        })
        
    except Exception as e:
        socketio.emit('training_log', {
            'message': f'❌ Training error: {str(e)}',
            'type': 'error'
        })

@app.route('/api/execute-action', methods=['POST'])
def execute_action():
    data = request.json
    action_type = data.get('action_type')
    action_value = data.get('action_value')
    
    try:
        if action_type == 'open_app':
            subprocess.Popen([action_value], shell=True)
            return jsonify({'status': 'success', 'message': f'Opened {action_value}'})
        
        elif action_type == 'search_web':
            url = f'https://www.google.com/search?q={action_value}'
            webbrowser.open(url)
            return jsonify({'status': 'success', 'message': f'Searching: {action_value}'})
        
        else:
            return jsonify({'status': 'error', 'message': 'Unknown action type'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/delete-gesture', methods=['POST'])
def delete_gesture():
    data = request.json
    gesture_id = str(data.get('gesture_id'))
    
    custom_gestures = load_custom_gestures()
    
    if gesture_id in custom_gestures:
        del custom_gestures[gesture_id]
        save_custom_gestures(custom_gestures)
        return jsonify({'status': 'success', 'message': 'Gesture deleted'})
    
    return jsonify({'status': 'error', 'message': 'Gesture not found'})

if __name__ == '__main__':
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('screenshots', exist_ok=True)
    print("=" * 60)
    print("Jarvis + Gesture Control Web Application")
    print("=" * 60)
    print("Server starting at http://localhost:5000")
    print("=" * 60)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
