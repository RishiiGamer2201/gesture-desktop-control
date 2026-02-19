// Jarvis Chat Interface Script

// Get current time as string
function getTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

// Add user message to chat
eel.expose(addUserMsg);
function addUserMsg(msg) {
    const chatContainer = document.getElementById('chatContainer');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${msg}</p>
            <span class="timestamp">${getTime()}</span>
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Add Jarvis's message to chat
eel.expose(addAppMsg);
function addAppMsg(msg) {
    const chatContainer = document.getElementById('chatContainer');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message app-message';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${msg}</p>
            <span class="timestamp">${getTime()}</span>
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Send message from input field
function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (message === '') return;
    
    // Send to Python backend
    eel.getUserInput(message);
    
    // Clear input
    input.value = '';
    input.focus();
}

// Handle Enter key
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('userInput');
    
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Focus input on load
    input.focus();
});

// Add welcome message timestamp
document.addEventListener('DOMContentLoaded', function() {
    const welcomeMsg = document.querySelector('.app-message .timestamp');
    if (welcomeMsg) {
        welcomeMsg.textContent = getTime();
    }
});
