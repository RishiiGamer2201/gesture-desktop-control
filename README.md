# 🤖 Jarvis + Gesture Control System

<div align="center">

**AI-Powered Voice Assistant with Custom Hand Gesture Recognition**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-lightgrey.svg)]()

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-demo) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

A complete hands-free computer control system combining **voice recognition** and **computer vision** to control your desktop using voice commands and hand gestures. Features a beautiful web interface for managing custom gestures with real-time training capabilities.

### 🎯 Key Highlights

- 🎤 **Voice Assistant (Jarvis)** - Natural language voice control
- ✋ **Gesture Recognition** - Control mouse/keyboard with hand gestures
- 🌐 **Web Dashboard** - Manage everything from your browser
- 🧠 **Custom Gestures** - Train AI to recognize your own gestures
- 📊 **Real-time Training** - See model training logs live
- 🎯 **99%+ Accuracy** - KNN-based gesture classification

---

## ✨ Features

### Voice Assistant (Jarvis)
- 🔍 Web search (Google, YouTube, Wikipedia)
- ⏰ Time and date queries
- 🗺️ Location search
- 📁 File navigation
- 🔊 Volume control
- 📸 Screenshots (saved to `screenshots/` folder)
- ✋ Launch/stop gesture control via voice

### Gesture Control
- 🖱️ Mouse control (move, click, scroll)
- ⌨️ Keyboard shortcuts
- 🎨 10 built-in gestures
- ➕ Unlimited custom gestures
- 🎯 Real-time hand tracking
- 🚀 30 FPS performance

### Web Dashboard
- 🏠 Project overview
- 📋 View all gestures
- ➕ Add custom gestures with camera
- 🎓 Automatic model training
- 📊 Live training logs
- ▶️ Start/stop Jarvis & gesture control

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_jarvis.txt
cd webapp && pip install -r requirements.txt && cd ..
```

### 2. Start Web Interface

```bash
cd webapp
python app.py
```

Open: `http://localhost:5000`

### 3. Use the System

**Option A: Via Web Interface**
- Go to Jarvis page → Click "Start Jarvis"
- Go to Gestures page → Click "Start Gesture Control"

**Option B: Standalone**
```bash
# Voice only
python jarvis.py

# Gestures only  
python main_knn_updated.py
```

---

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam
- Microphone (for voice control)
- Internet connection (for voice recognition)

### Step-by-Step

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/jarvis-gesture-control.git
cd jarvis-gesture-control
```

2. **Install dependencies**
```bash
# Core dependencies
pip install -r requirements_jarvis.txt

# Web app dependencies
cd webapp
pip install -r requirements.txt
cd ..
```

3. **Test installation**
```bash
python test_system.py
```

4. **Run!**
```bash
# Start web interface
cd webapp && python app.py

# Or use standalone
python jarvis.py
```

---

## 💡 Usage

### Creating Custom Gestures

1. Open web app: `http://localhost:5000/add-gesture`
2. Click "Start Camera"
3. Show your gesture
4. Click "Start Collection"
5. Hold gesture for 50-100 samples
6. Click "Done Collecting"
7. Configure:
   - Name: "My Gesture"
   - Action: "Open App" or "Search Web"
   - Value: `notepad` or `youtube`
8. Click "Train Model"
9. Watch live training logs!
10. Use your gesture in gesture control mode

### Voice Commands

Say "Jarvis" followed by:
- `search Python tutorial` - Google search
- `youtube relaxing music` - YouTube
- `what's the time` - Current time
- `launch gesture` - Start gesture control
- `screenshot` - Save screenshot
- `help` - Show all commands

### Built-in Gestures

| Gesture | Action |
|---------|--------|
| PALM | Halt cursor |
| V_SIGN | Move cursor |
| MIDDLE_ONLY | Left click |
| INDEX_ONLY | Right click |
| JOINED_FINGERS | Double click |
| PINKY_ONLY | Scroll down / Volume down |
| PINKY_THUMB | Scroll up / Volume up |
| THUMBS_UP | Volume mode toggle |
| THUMBS_DOWN | Open Spotify |
| ROCK_SIGN | Reserved |

## 🛠️ Technology Stack

### AI & Machine Learning
- MediaPipe - Hand tracking
- Scikit-learn - KNN classifier
- OpenCV - Computer vision
- TensorFlow - Backend

### Backend
- Python 3.11
- Flask - Web framework
- Flask-SocketIO - Real-time updates
- PyAutoGUI - System control

### Frontend
- Bootstrap 5
- JavaScript (ES6+)
- Socket.IO - WebSockets
- HTML5/CSS3

---

## 📊 Project Structure

```
jarvis-gesture-control/
├── jarvis.py                    # Voice assistant
├── main_knn_updated.py          # Gesture controller
├── app.py                       # Jarvis chat handler
├── test_system.py               # System verification
├── collect_data_updated.py      # Data collection
├── train_knn_updated.py         # Model training
├── requirements_jarvis.txt      # Dependencies
│
├── webapp/                      # Web application
│   ├── app.py                   # Flask server
│   ├── templates/               # HTML pages
│   ├── requirements.txt         # Web dependencies
│   └── custom_gestures.json     # User gestures
│
├── models/                      # Trained models
│   └── gesture_knn_model_updated.pkl
│
├── training_data/               # Training datasets
├── screenshots/                 # Saved screenshots
└── web/                         # Jarvis chat UI
```

---

## 🎯 Performance

- **Model Accuracy:** 99%+
- **FPS:** 25-30
- **Latency:** <50ms
- **Gesture Recognition:** Real-time
- **Custom Gestures:** Unlimited

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- Scikit-learn for machine learning
- Flask team for the web framework
- Bootstrap for UI components

---

## 📞 Support

- 📧 Email: rishiikumarsingh2201@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/RishiiGamer2201/gesture-desktop-control/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/RishiiGamer2201/gesture-desktop-control/discussions)

<div align="center">

**Made with ❤️ by Rishii Kumar Singh**

[⬆ Back to Top](#-jarvis--gesture-control-system)

</div>
