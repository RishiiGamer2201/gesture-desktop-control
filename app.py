"""
ChatBot GUI Handler for Jarvis
Web-based interface using Eel
"""

import eel
import os
from queue import Queue

class ChatBot:
    started = False
    userinputQueue = Queue()
    
    @staticmethod
    def isUserInput():
        return not ChatBot.userinputQueue.empty()
    
    @staticmethod
    def popUserInput():
        return ChatBot.userinputQueue.get()
    
    @staticmethod
    def close_callback(route, websockets):
        """Called when browser window is closed"""
        exit()
    
    @staticmethod
    @eel.expose
    def getUserInput(msg):
        """Receive input from web interface"""
        ChatBot.userinputQueue.put(msg)
        print(f"User: {msg}")
    
    @staticmethod
    def close():
        ChatBot.started = False
    
    @staticmethod
    def addUserMsg(msg):
        """Add user message to chat"""
        eel.addUserMsg(msg)
    
    @staticmethod
    def addAppMsg(msg):
        """Add Jarvis's response to chat"""
        eel.addAppMsg(msg)
    
    @staticmethod
    def start():
        """Start the web-based chat interface"""
        path = os.path.dirname(os.path.abspath(__file__))
        eel.init(path + r'\web', allowed_extensions=['.js', '.html'])
        
        try:
            eel.start('index.html', 
                     mode='chrome',
                     host='localhost',
                     port=27007,  # Changed port to avoid conflicts
                     block=False,
                     size=(400, 600),
                     position=(10, 100),
                     disable_cache=True,
                     close_callback=ChatBot.close_callback)
            
            ChatBot.started = True
            
            while ChatBot.started:
                try:
                    eel.sleep(10.0)
                except:
                    # Main thread exited
                    break
        
        except Exception as e:
            print(f"Error starting ChatBot: {e}")
            pass
