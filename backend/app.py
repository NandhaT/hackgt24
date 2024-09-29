from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO

@app.route('/')
def index():
    return 'Flask server is running!'

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
