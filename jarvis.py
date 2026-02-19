"""
Jarvis - Voice Assistant with Gesture Control Integration
Integrates KNN gesture recognition system
"""

import pyttsx3
import speech_recognition as sr
from datetime import date
import time
import webbrowser
import datetime
from pynput.keyboard import Key, Controller
import pyautogui
import sys
import os
from os import listdir
from os.path import isfile, join
import wikipedia
import app
from threading import Thread

# Import gesture controller
try:
    import main_knn_updated as GestureController
except ImportError:
    print("  Gesture controller not found. Voice-only mode.")
    GestureController = None

# -------------Object Initialization---------------
today = date.today()
r = sr.Recognizer()
keyboard = Controller()
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# ----------------Variables------------------------
file_exp_status = False
files = []
path = ''
is_awake = True
gesture_active = False
gesture_controller = None

# ------------------Functions----------------------
def reply(audio):
    """Speak and display response"""
    app.ChatBot.addAppMsg(audio)
    print(audio)
    engine.say(audio)
    engine.runAndWait()


def wish():
    """Greet based on time of day"""
    hour = int(datetime.datetime.now().hour)

    if hour >= 0 and hour < 12:
        reply("Good Morning!")
    elif hour >= 12 and hour < 18:
        reply("Good Afternoon!")   
    else:
        reply("Good Evening!")  
        
    reply("I am Jarvis, your AI assistant. How may I help you?")


# Set Microphone parameters
try:
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        r.adjust_for_ambient_noise(source, duration=1)
        r.energy_threshold = 300  # Adjusted for better recognition
        r.dynamic_energy_threshold = True
        r.pause_threshold = 1.0
        print(f" Microphone ready (threshold: {r.energy_threshold})")
except Exception as e:
    print(f" Microphone setup warning: {e}")
    print("You can still use chat interface!")


def record_audio():
    """Convert speech to text"""
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        r.pause_threshold = 1.0
        voice_data = ''
        
        try:
            audio = r.listen(source, phrase_time_limit=5, timeout=5)
            print("Processing...")
            voice_data = r.recognize_google(audio)
            print(f"You said: {voice_data}")
        except sr.WaitTimeoutError:
            print("Listening timeout")
            return ''
        except sr.RequestError:
            print(' Service down. Check internet.')
            return ''
        except sr.UnknownValueError:
            print('...')  # Silent, just listening
            return ''
        except Exception as e:
            print(f"Error: {e}")
            return ''
        
        return voice_data.lower()


def start_gesture_control():
    """Start gesture recognition in separate thread"""
    global gesture_active, gesture_controller
    
    if GestureController is None:
        reply("Gesture control module not available.")
        return False
    
    if gesture_active:
        reply("Gesture control is already active.")
        return False
    
    try:
        gesture_controller = GestureController.KNNGestureController()
        gesture_thread = Thread(target=gesture_controller.run)
        gesture_thread.daemon = True
        gesture_thread.start()
        gesture_active = True
        reply("Gesture control activated successfully.")
        return True
    except Exception as e:
        reply(f"Failed to start gesture control: {str(e)}")
        return False


def stop_gesture_control():
    """Stop gesture recognition"""
    global gesture_active, gesture_controller
    
    if not gesture_active:
        reply("Gesture control is already inactive.")
        return False
    
    gesture_active = False
    gesture_controller = None
    reply("Gesture control deactivated.")
    return True


def respond(voice_data):
    """Process and execute commands"""
    global file_exp_status, files, is_awake, path
    
    print(f"Command: {voice_data}")
    voice_data = voice_data.replace('jarvis', '')
    app.ChatBot.addUserMsg(voice_data)

    # Wake up command
    if not is_awake:
        if 'wake up' in voice_data:
            is_awake = True
            wish()
        return

    # ============ BASIC COMMANDS ============
    
    if 'hello' in voice_data or 'hi' in voice_data:
        wish()

    elif 'what is your name' in voice_data or 'who are you' in voice_data:
        reply('I am Jarvis, your AI assistant!')

    elif 'date' in voice_data:
        reply(today.strftime("%B %d, %Y"))

    elif 'time' in voice_data:
        reply(str(datetime.datetime.now()).split(" ")[1].split('.')[0])

    elif 'search' in voice_data:
        query = voice_data.split('search')[1].strip()
        reply(f'Searching for {query}')
        url = 'https://google.com/search?q=' + query
        try:
            webbrowser.get().open(url)
            reply('Here are the results')
        except:
            reply('Please check your internet connection')

    elif 'wikipedia' in voice_data:
        query = voice_data.replace('wikipedia', '').strip()
        try:
            reply(f'Searching Wikipedia for {query}')
            result = wikipedia.summary(query, sentences=2)
            reply(result)
        except:
            reply('Could not find information on Wikipedia')

    elif 'location' in voice_data:
        reply('Which place are you looking for?')
        temp_audio = record_audio()
        app.ChatBot.addUserMsg(temp_audio)
        reply('Locating...')
        url = 'https://google.nl/maps/place/' + temp_audio
        try:
            webbrowser.get().open(url)
            reply('This is what I found')
        except:
            reply('Please check your internet connection')

    elif 'youtube' in voice_data:
        query = voice_data.replace('youtube', '').strip()
        if query:
            url = 'https://www.youtube.com/results?search_query=' + query
            webbrowser.get().open(url)
            reply(f'Playing {query} on YouTube')
        else:
            webbrowser.get().open('https://www.youtube.com')
            reply('Opening YouTube')

    elif ('bye' in voice_data) or ('goodbye' in voice_data):
        reply("Goodbye! Have a nice day.")
        is_awake = False

    elif ('exit' in voice_data) or ('terminate' in voice_data) or ('quit' in voice_data):
        if gesture_active:
            stop_gesture_control()
        app.ChatBot.close()
        reply("Shutting down. Goodbye!")
        sys.exit()

    # ============ GESTURE CONTROL ============
    
    elif 'launch gesture' in voice_data or 'start gesture' in voice_data:
        start_gesture_control()

    elif 'stop gesture' in voice_data or 'close gesture' in voice_data:
        stop_gesture_control()

    elif 'gesture status' in voice_data:
        if gesture_active:
            reply('Gesture control is currently active')
        else:
            reply('Gesture control is currently inactive')

    # ============ KEYBOARD CONTROLS ============
    
    elif 'copy' in voice_data:
        with keyboard.pressed(Key.ctrl):
            keyboard.press('c')
            keyboard.release('c')
        reply('Copied')
          
    elif 'paste' in voice_data:
        with keyboard.pressed(Key.ctrl):
            keyboard.press('v')
            keyboard.release('v')
        reply('Pasted')

    elif 'cut' in voice_data:
        with keyboard.pressed(Key.ctrl):
            keyboard.press('x')
            keyboard.release('x')
        reply('Cut')

    elif 'undo' in voice_data:
        with keyboard.pressed(Key.ctrl):
            keyboard.press('z')
            keyboard.release('z')
        reply('Undone')

    elif 'select all' in voice_data:
        with keyboard.pressed(Key.ctrl):
            keyboard.press('a')
            keyboard.release('a')
        reply('Selected all')

    # ============ FILE NAVIGATION ============
    
    elif 'list files' in voice_data or 'show files' in voice_data:
        counter = 0
        path = 'C://'
        files = listdir(path)
        filestr = ""
        for f in files:
            counter += 1
            print(f"{counter}: {f}")
            filestr += f"{counter}: {f}<br>"
        file_exp_status = True
        reply('These are the files in your root directory')
        app.ChatBot.addAppMsg(filestr)
        
    elif file_exp_status:
        counter = 0   
        if 'open' in voice_data:
            try:
                file_num = int(voice_data.split(' ')[-1]) - 1
                if isfile(join(path, files[file_num])):
                    os.startfile(path + files[file_num])
                    file_exp_status = False
                    reply('File opened')
                else:
                    path = path + files[file_num] + '//'
                    files = listdir(path)
                    filestr = ""
                    for f in files:
                        counter += 1
                        filestr += f"{counter}: {f}<br>"
                    reply('Folder opened')
                    app.ChatBot.addAppMsg(filestr)
            except:
                reply('Could not open that file')
                                    
        elif 'back' in voice_data:
            filestr = ""
            if path == 'C://':
                reply('This is the root directory')
            else:
                path_parts = path.split('//')[:-2]
                path = '//'.join(path_parts) + '//'
                files = listdir(path)
                for f in files:
                    counter += 1
                    filestr += f"{counter}: {f}<br>"
                reply('Going back')
                app.ChatBot.addAppMsg(filestr)

    # ============ SYSTEM CONTROLS ============
    
    elif 'volume up' in voice_data:
        for _ in range(5):
            pyautogui.press('volumeup')
        reply('Volume increased')

    elif 'volume down' in voice_data:
        for _ in range(5):
            pyautogui.press('volumedown')
        reply('Volume decreased')

    elif 'mute' in voice_data:
        pyautogui.press('volumemute')
        reply('Muted')

    elif 'screenshot' in voice_data:
        # Create screenshots folder if it doesn't exist
        os.makedirs('screenshots', exist_ok=True)
        # Save with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'screenshots/screenshot_{timestamp}.png'
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        reply(f'Screenshot saved to {filepath}')

    # ============ HELP ============
    
    elif 'help' in voice_data or 'what can you do' in voice_data:
        help_text = """
        I can help you with:
        - Web search (say 'search [query]')
        - Time and date
        - YouTube videos
        - Wikipedia information
        - File navigation
        - Gesture control (say 'launch gesture')
        - System controls (volume, screenshots)
        - And much more!
        """
        reply(help_text)

    else: 
        reply('I am not programmed to do that yet!')


# ------------------Driver Code--------------------

def main():
    """Main program loop"""
    global is_awake
    
    # Start GUI in separate thread
    gui_thread = Thread(target=app.ChatBot.start)
    gui_thread.start()

    # Wait for GUI to start
    print("Starting Jarvis...")
    while not app.ChatBot.started:
        time.sleep(0.5)

    # Initial greeting
    wish()
    
    print("\n" + "=" * 50)
    print("Jarvis is ready!")
    print("- Type in the chat window, OR")
    print("- Say 'Jarvis' + your command")
    print("=" * 50 + "\n")
    
    # Main loop
    while True:
        voice_data = None
        from_chat = False
        
        # Check for GUI input first
        if app.ChatBot.isUserInput():
            voice_data = app.ChatBot.popUserInput().lower()
            from_chat = True
            print(f"Chat input: {voice_data}")
        else:
            # Voice input
            voice_data = record_audio()
            if voice_data:
                print(f"Voice input: {voice_data}")

        # Process command
        if voice_data:
            # For chat input: process directly
            # For voice input: require "jarvis" wake word
            if from_chat:
                # Chat input - process directly
                try:
                    respond(voice_data)
                except SystemExit:
                    reply("Shutting down successfully")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    reply("An error occurred while processing your command")
            else:
                # Voice input - require wake word
                if 'jarvis' in voice_data:
                    try:
                        respond(voice_data)
                    except SystemExit:
                        reply("Shutting down successfully")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        reply("An error occurred while processing your command")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Jarvis shutting down...")
        if gesture_active:
            stop_gesture_control()
        app.ChatBot.close()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
