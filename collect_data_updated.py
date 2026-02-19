"""
Gesture Data Collection for KNN Model
Updated gesture set - no drag/fist, new pinky gestures
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# Configuration
DATA_DIR = 'training_data'
os.makedirs(DATA_DIR, exist_ok=True)

# NEW GESTURE DEFINITIONS
GESTURES = {
    0: "PALM",           # Neutral/Halt
    1: "V_SIGN",         # Move cursor
    2: "MIDDLE_ONLY",    # Left click
    3: "INDEX_ONLY",     # Right click
    4: "JOINED_FINGERS", # Double click
    5: "PINKY_ONLY",     # Scroll down / Volume down
    6: "PINKY_THUMB",    # Scroll up / Volume up
    7: "THUMBS_UP",      # Volume mode toggle
    8: "THUMBS_DOWN",    # Open Spotify
    9: "ROCK_SIGN"       # Optional
}

class GestureDataCollector:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.data = []
        self.labels = []
        self.counter = {i: 0 for i in range(10)}
    
    def extract_features(self, hand_landmarks):
        """Extract 42 features (21 landmarks × 2 coordinates, normalized to wrist)"""
        wrist = hand_landmarks.landmark[0]
        features = []
        
        for landmark in hand_landmarks.landmark:
            features.append(landmark.x - wrist.x)
            features.append(landmark.y - wrist.y)
        
        return features
    
    def draw_ui(self, frame, hand_detected=False, last_captured=None):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "GESTURE DATA COLLECTOR", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        y = 75
        cv2.putText(frame, "Press 0-9 to capture gesture", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, "Press 's' to save data", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, "Press 'q' to quit", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gesture list with updated names
        y = 145
        cv2.putText(frame, "Gestures:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 30
        
        for i in range(10):
            count = self.counter[i]
            if count >= 50:
                color = (0, 255, 0)
            elif count >= 20:
                color = (0, 255, 255)
            else:
                color = (128, 128, 128)
            
            gesture_text = f"{i}: {GESTURES[i][:15]:15s} {count:3d}"
            cv2.putText(frame, gesture_text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 22
        
        # Total count
        cv2.putText(frame, f"TOTAL: {len(self.data)}", (20, 375),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Hand detection
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if hand_detected else "NO HAND"
        cv2.circle(frame, (w - 150, 30), 10, status_color, -1)
        cv2.putText(frame, status_text, (w - 130, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Last captured
        if last_captured is not None:
            cv2.putText(frame, f"Captured: {last_captured}", (w - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def save_data(self):
        """Save collected data to CSV"""
        if len(self.data) == 0:
            print("⚠️  No data to save!")
            return False
        
        # Combine features and labels
        dataset = np.hstack((
            np.array(self.data),
            np.array(self.labels).reshape(-1, 1)
        ))
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_data_{timestamp}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        np.savetxt(filepath, dataset, delimiter=',')
        
        print("\n" + "=" * 60)
        print("DATA SAVED SUCCESSFULLY")
        print("=" * 60)
        print(f"File: {filepath}")
        print(f"Total samples: {len(self.data)}")
        print("\nSamples per gesture:")
        for i, name in GESTURES.items():
            print(f"  {name:15s}: {self.counter[i]:3d}")
        print("=" * 60)
        
        return True
    
    def run(self):
        """Main collection loop"""
        print("=" * 60)
        print("GESTURE DATA COLLECTOR - NEW GESTURE SET")
        print("=" * 60)
        print("\n📋 Gestures to collect:")
        for key, name in GESTURES.items():
            print(f"  Press '{key}' for {name}")
        print("\n💡 Recommendation: 50-100 samples per gesture")
        print("\n⌨️  Controls:")
        print("  's' - Save data")
        print("  'q' - Quit")
        print("\n🆕 NEW GESTURES:")
        print("  5: PINKY_ONLY   - Only pinky finger up (scroll/volume down)")
        print("  6: PINKY_THUMB  - Pinky + thumb up (scroll/volume up)")
        print("\n❌ REMOVED GESTURES:")
        print("  - FIST (no longer used)")
        print("  - PINCH (replaced by pinky gestures)")
        print("\n" + "=" * 60 + "\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_captured = None
        last_capture_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract features
                features = self.extract_features(hand_landmarks)
                
                # Check for keypress
                key = cv2.waitKey(1) & 0xFF
                
                # Capture gesture (0-9)
                if ord('0') <= key <= ord('9'):
                    label = key - ord('0')
                    
                    self.data.append(features)
                    self.labels.append(label)
                    self.counter[label] += 1
                    
                    last_captured = GESTURES[label]
                    last_capture_time = cv2.getTickCount()
                    
                    print(f"✓ Captured {GESTURES[label]}: {self.counter[label]} samples")
                
                # Save data
                elif key == ord('s'):
                    if self.save_data():
                        break
                
                # Quit
                elif key == ord('q'):
                    print("\n⚠️  Quitting without saving!")
                    break
            
            # Clear "last captured" message after 1 second
            if last_captured and (cv2.getTickCount() - last_capture_time) / cv2.getTickFrequency() > 1.0:
                last_captured = None
            
            # Draw UI
            frame = self.draw_ui(frame, hand_detected, last_captured)
            
            cv2.imshow("Gesture Data Collector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        if len(self.data) > 0:
            print(f"\n📊 Collection Summary:")
            print(f"Total samples: {len(self.data)}")
            for i, name in GESTURES.items():
                status = "✓" if self.counter[i] >= 50 else "⚠" if self.counter[i] >= 20 else "❌"
                print(f"  {status} {name:15s}: {self.counter[i]:3d} samples")

if __name__ == "__main__":
    try:
        collector = GestureDataCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
