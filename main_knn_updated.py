"""
Gesture Control with KNN Model - UPDATED
New gesture set: Pinky gestures for scroll/volume, no drag/fist
Relative cursor movement
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import pickle
import os
import subprocess
import platform
import json
import webbrowser

pyautogui.FAILSAFE = False

MODEL_PATH = 'models/gesture_knn_model_updated.pkl'

# NEW GESTURE DEFINITIONS
GESTURES = {
    0: "PALM",
    1: "V_SIGN",
    2: "MIDDLE_ONLY",
    3: "INDEX_ONLY",
    4: "JOINED_FINGERS",
    5: "PINKY_ONLY",      # NEW: Scroll down / Volume down
    6: "PINKY_THUMB",     # NEW: Scroll up / Volume up
    7: "THUMBS_UP",
    8: "THUMBS_DOWN",
    9: "ROCK_SIGN"
}

# Load custom gestures
def load_custom_gestures():
    """Load custom gestures from JSON file"""
    gesture_file = 'webapp/custom_gestures.json'
    if os.path.exists(gesture_file):
        try:
            with open(gesture_file, 'r') as f:
                custom = json.load(f)
                print(f" Loaded {len(custom)} custom gesture(s)")
                return custom
        except:
            pass
    return {}

# Execute custom gesture action
def execute_custom_action(gesture_id, custom_gestures):
    """Execute the action assigned to a custom gesture"""
    gesture_id_str = str(gesture_id)
    
    if gesture_id_str not in custom_gestures:
        return False
    
    gesture = custom_gestures[gesture_id_str]
    action_type = gesture.get('action')
    action_value = gesture.get('action_value')
    gesture_name = gesture.get('name', 'Custom')
    
    try:
        if action_type == 'open_app':
            print(f" {gesture_name}: Opening {action_value}")
            if os.name == 'nt':  # Windows
                subprocess.Popen([action_value], shell=True)
            else:  # Mac/Linux
                subprocess.Popen([action_value])
            return True
            
        elif action_type == 'search_web':
            print(f" {gesture_name}: Searching '{action_value}'")
            url = f'https://www.google.com/search?q={action_value}'
            webbrowser.open(url)
            return True
    
    except Exception as e:
        print(f" Error executing {gesture_name}: {e}")
    
    return False

class KNNGestureController:
    def __init__(self):
        self.model = self.load_model()
        
        # Load custom gestures
        self.custom_gestures = load_custom_gestures()
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Screen
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cam_w = 1280
        self.cam_h = 720
        
        # FPS
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Gesture state
        self.prev_gesture_id = None
        self.gesture_state = "neutral"
        
        # Relative cursor movement
        self.cursor_active = False
        self.cursor_initialized = False
        self.prev_hand_x = None
        self.prev_hand_y = None
        self.movement_sensitivity = 2.5
        
        # Volume mode
        self.volume_mode = False
        self.last_thumbs_up_time = 0
        
        # Click cooldown
        self.last_click_time = 0
        self.click_cooldown = 0.5
        
        # Scroll/volume action cooldown
        self.last_action_time = 0
        self.action_cooldown = 0.15  # Faster repeat for smooth scrolling
        
        # Confidence threshold
        self.confidence_threshold = 0.70
    
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f" Model not found: {MODEL_PATH}")
            print("   Run 'python train_knn_updated.py' first!")
            return None
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f" Loaded KNN model from {MODEL_PATH}")
        return model
    
    def extract_features(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
        return np.array(features).reshape(1, -1)
    
    def predict_gesture(self, hand_landmarks):
        if self.model is None:
            return None, "NO_MODEL", 0.0
        features = self.extract_features(hand_landmarks)
        gesture_id = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[int(gesture_id)]
        gesture_name = GESTURES.get(int(gesture_id), "UNKNOWN")
        return int(gesture_id), gesture_name, confidence
    
    def calculate_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def get_midpoint(self, p1, p2):
        return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
    
    def open_spotify(self):
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(['spotify'])
            elif system == "Darwin":
                subprocess.Popen(['open', '-a', 'Spotify'])
            else:
                subprocess.Popen(['spotify'])
            print(" Opening Spotify...")
        except:
            print("  Spotify not found")
    
    def execute_gesture(self, gesture_id, hand_landmarks):
        now = time.time()
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # 0. PALM — Neutral/Halt
        if gesture_id == 0:
            self.cursor_active = False
            self.prev_hand_x = None
            self.prev_hand_y = None
            self.gesture_state = "neutral"
        
        # 1. V_SIGN — Relative cursor movement
        elif gesture_id == 1:
            mid_x, mid_y = self.get_midpoint(index_tip, middle_tip)
            
            if not self.cursor_initialized:
                pyautogui.moveTo(self.screen_w // 2, self.screen_h // 2)
                self.cursor_initialized = True
                self.prev_hand_x = mid_x
                self.prev_hand_y = mid_y
                self.cursor_active = True
                print(" Cursor initialized to center")
                return
            
            if self.prev_hand_x is not None and self.cursor_active:
                dx = mid_x - self.prev_hand_x
                dy = mid_y - self.prev_hand_y
                
                cur_x, cur_y = pyautogui.position()
                new_x = cur_x + dx * self.screen_w * self.movement_sensitivity
                new_y = cur_y + dy * self.screen_h * self.movement_sensitivity
                new_x = max(0, min(self.screen_w - 1, new_x))
                new_y = max(0, min(self.screen_h - 1, new_y))
                pyautogui.moveTo(int(new_x), int(new_y))
            
            self.prev_hand_x = mid_x
            self.prev_hand_y = mid_y
            self.cursor_active = True
            self.gesture_state = "moving"
        
        # 2. MIDDLE_ONLY — Left click
        elif gesture_id == 2:
            if now - self.last_click_time > self.click_cooldown:
                pyautogui.click()
                self.last_click_time = now
                print(" Left Click")
        
        # 3. INDEX_ONLY — Right click
        elif gesture_id == 3:
            if now - self.last_click_time > self.click_cooldown:
                pyautogui.rightClick()
                self.last_click_time = now
                print(" Right Click")
        
        # 4. JOINED_FINGERS — Double click
        elif gesture_id == 4:
            if now - self.last_click_time > self.click_cooldown:
                pyautogui.doubleClick()
                self.last_click_time = now
                print(" Double Click")
        
        # 5. PINKY_ONLY — Scroll down / Volume down
        elif gesture_id == 5:
            if now - self.last_action_time > self.action_cooldown:
                if self.volume_mode:
                    # Volume down
                    pyautogui.press('volumedown')
                    print(" Volume Down")
                else:
                    # Scroll down
                    pyautogui.scroll(-5)  # Negative = scroll down
                    print(" Scroll Down")
                self.last_action_time = now
        
        # 6. PINKY_THUMB — Scroll up / Volume up
        elif gesture_id == 6:
            if now - self.last_action_time > self.action_cooldown:
                if self.volume_mode:
                    # Volume up
                    pyautogui.press('volumeup')
                    print(" Volume Up")
                else:
                    # Scroll up
                    pyautogui.scroll(5)  # Positive = scroll up
                    print(" Scroll Up")
                self.last_action_time = now
        
        # 7. THUMBS_UP — Volume mode toggle
        elif gesture_id == 7:
            if not self.volume_mode and (now - self.last_thumbs_up_time) > 5:
                self.volume_mode = True
                self.last_thumbs_up_time = now
                print(" Volume mode ON")
            elif self.volume_mode and (now - self.last_thumbs_up_time) > 5:
                self.volume_mode = False
                self.last_thumbs_up_time = now
                print(" Volume mode OFF")
        
        # 8. THUMBS_DOWN — Open Spotify
        elif gesture_id == 8:
            if self.prev_gesture_id != 8:
                self.open_spotify()
        
        # 9. ROCK_SIGN — Reserved
        elif gesture_id == 9:
            pass
        
        # 10+. CUSTOM GESTURES
        elif gesture_id >= 10:
            # Check if this is a new custom gesture
            if self.prev_gesture_id != gesture_id:
                if now - self.last_action_time > self.action_cooldown:
                    # Execute custom action
                    if execute_custom_action(gesture_id, self.custom_gestures):
                        self.last_action_time = now
                    else:
                        # Unknown custom gesture
                        gesture_name = self.custom_gestures.get(str(gesture_id), {}).get('name', f'Gesture {gesture_id}')
                        print(f" Custom gesture {gesture_id} ({gesture_name}) not configured")
        
        self.prev_gesture_id = gesture_id
    
    def draw_ui(self, frame, gesture_name, confidence, hand_detected):
        h, w = frame.shape[:2]
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "KNN GESTURE CONTROL - UPDATED", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if hand_detected:
            color = (0, 255, 0) if confidence >= 0.8 else \
                    (0, 255, 255) if confidence >= 0.7 else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {gesture_name}", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (400, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No hand detected", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Status badges
        y = 150
        if not self.cursor_active and self.cursor_initialized:
            cv2.putText(frame, "CURSOR PAUSED", (w - 260, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            y += 30
        
        if self.volume_mode:
            cv2.putText(frame, "VOLUME MODE", (w - 260, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y += 30
        
        return frame
    
    def run(self):
        if self.model is None:
            print(" Cannot run without model!")
            return
        
        print("=" * 60)
        print(" KNN GESTURE CONTROL - UPDATED")
        print("=" * 60)
        print("\n Model loaded")
        print(" Camera ready")
        print(" Relative cursor movement")
        print("\n NEW GESTURES:")
        print("   PINKY_ONLY   → Scroll down / Volume down")
        print("   PINKY_THUMB  → Scroll up / Volume up")
        print("\n Gesture Guide:")
        print("    PALM           → Halt cursor")
        print("    V_SIGN         → Move cursor (relative)")
        print("    MIDDLE_ONLY    → Left click")
        print("    INDEX_ONLY     → Right click")
        print("    JOINED_FINGERS → Double click")
        print("    THUMBS_UP      → Volume mode toggle")
        print("    THUMBS_DOWN    → Open Spotify")
        print("\n REMOVED:")
        print("\nPress 'q' to quit")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.calculate_fps()
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            gesture_name = "NONE"
            confidence = 0.0
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                gesture_id, gesture_name, confidence = self.predict_gesture(hand_landmarks)
                
                if confidence >= self.confidence_threshold:
                    self.execute_gesture(gesture_id, hand_landmarks)
            
            frame = self.draw_ui(frame, gesture_name, confidence, hand_detected)
            cv2.imshow("KNN Gesture Control - Updated", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n Exiting...")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        KNNGestureController().run()
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
