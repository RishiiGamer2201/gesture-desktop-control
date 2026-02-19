"""
Jarvis System Test
Verifies all components are working
"""

import sys

print("=" * 60)
print("JARVIS SYSTEM TEST")
print("=" * 60)
print()

# Test 1: Python Version
print("1. Checking Python version...")
if sys.version_info >= (3, 7):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor} (OK)")
else:
    print(f"   ❌ Python {sys.version_info.major}.{sys.version_info.minor} (Need 3.7+)")
print()

# Test 2: Required Modules
print("2. Checking required modules...")
modules_status = []

required_modules = {
    'pyttsx3': 'Text-to-speech',
    'speech_recognition': 'Speech recognition',
    'pynput': 'Keyboard control',
    'eel': 'Web interface',
    'cv2': 'OpenCV (gesture control)',
    'mediapipe': 'Hand tracking',
    'numpy': 'Numerical computing',
    'pyautogui': 'Mouse/keyboard control',
    'sklearn': 'Machine learning'
}

for module, description in required_modules.items():
    try:
        __import__(module)
        print(f"   ✓ {module:20s} - {description}")
        modules_status.append(True)
    except ImportError:
        print(f"   ❌ {module:20s} - {description} (MISSING)")
        modules_status.append(False)
print()

# Test 3: Files
print("3. Checking required files...")
import os

required_files = [
    'jarvis.py',
    'app.py',
    'main_knn_updated.py',
    'web/index.html',
    'web/style.css',
    'web/script.js'
]

files_status = []
for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
        files_status.append(True)
    else:
        print(f"   ❌ {file} (MISSING)")
        files_status.append(False)
print()

# Test 4: Optional - Microphone
print("4. Checking microphone (optional)...")
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"   ✓ Microphone found")
        mic_ok = True
except:
    print(f"   ⚠️ Microphone not available (Chat will still work)")
    mic_ok = False
print()

# Test 5: Optional - Gesture Model
print("5. Checking gesture model (optional)...")
if os.path.exists('models/gesture_knn_model_updated.pkl'):
    print(f"   ✓ Gesture model found")
    model_ok = True
else:
    print(f"   ⚠️ Gesture model not found")
    print(f"      Run: python train_knn_updated.py")
    model_ok = False
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

all_modules = all(modules_status)
all_files = all(files_status)

if all_modules and all_files:
    print("✅ All required components are installed!")
    print()
    print("You can start Jarvis:")
    print("   python jarvis.py")
    print()
    if not mic_ok:
        print("Note: Microphone not found, but chat interface will work!")
    if not model_ok:
        print("Note: Train gesture model to use hand control")
else:
    print("❌ Some components are missing")
    print()
    if not all_modules:
        print("Install missing modules:")
        print("   pip install -r requirements_jarvis.txt")
        print()
    if not all_files:
        print("Make sure all files from the zip are extracted!")
        print()

print("=" * 60)
