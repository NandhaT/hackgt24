import torch
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask_socketio import SocketIO, emit
import base64
import time
import logging

logging.basicConfig(level=logging.INFO)

# Flask API setup
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
globalFrame = None

# MongoDB connection
mongo_uri = "mongodb+srv://pdiddy:pdiddy!@cluster0.ydaow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["medical_item_tracking"]
collection = db["tracked_items"]

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

def gen_frames():
    while True:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 0]
        frame = globalFrame
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64 to send over WebSocket
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            # Send the frame over WebSocket
            socketio.emit('frame', {'data': frame_base64})
        time.sleep(0.1)

@socketio.on('start_video')
def start_video_stream():
    gen_frames()  # Start capturing video frames

# API route to get items currently inside the zone
@app.route('/api/items/inside', methods=['GET'])
def get_items_inside_zone():
    inside_items = collection.find({"current_status": "inside"})
    item_list = [{"item_id": item["item_id"], "timestamp_entered": item["timestamp_entered"]} for item in inside_items]
    return jsonify(item_list)

def run_flask_api():
    """Run the Flask API in a separate thread."""
    app.run(host='0.0.0.0', port=5000)

def run_socket():
    socketio.run(app, host="0.0.0.0", port=5001)

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return self.grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def isOpened(self):
        return self.cap.isOpened()

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()

class TrackManager:
    def __init__(self, max_age=15):
        self.tracks = {}
        self.max_age = max_age

    def update(self, track_id, bbox, class_name, inside_zone):
        self.tracks[track_id] = {
            'last_seen': datetime.now(),
            'bbox': bbox,
            'class_name': class_name,
            'inside_zone': inside_zone
        }

    def get_active_tracks(self):
        current_time = datetime.now()
        active_tracks = {}
        for track_id, track_info in list(self.tracks.items()):
            if (current_time - track_info['last_seen']).total_seconds() < self.max_age:
                active_tracks[track_id] = track_info
            else:
                del self.tracks[track_id]
        return active_tracks

def run_object_tracking():
    CONFIDENCE_THRESHOLD = 0.4
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    cap = VideoCaptureAsync(0).start()
    if not cap.isOpened():
        logging.error("Failed to open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read frame.")
        return
    frame_height, frame_width = frame.shape[:2]

    # Modify the buffer to make the zone wider
    buffer_vertical = min(120, frame_height // 4)  # Reduced vertical buffer
    buffer_horizontal = min(60, frame_width // 8)  # Reduced horizontal buffer
    ZONE_POLY = np.array([
        [buffer_horizontal, buffer_vertical], 
        [frame_width - buffer_horizontal, buffer_vertical], 
        [frame_width - buffer_horizontal, frame_height - buffer_vertical], 
        [buffer_horizontal, frame_height - buffer_vertical]
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/output7/weights/best.pt').to(device)
    model.conf = CONFIDENCE_THRESHOLD

    tracker = DeepSort(max_age=15, nn_budget=100, embedder="mobilenet", embedder_gpu=True)

    track_manager = TrackManager(max_age=15)
    db_cache = {}
    frame_count = 0
    FRAME_SKIP = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame, breaking the loop")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        input_frame = cv2.resize(frame, (320, 320))

        with torch.no_grad():
            results = model(input_frame)

        dets = results.xyxy[0].cpu().numpy()
        dets = dets[dets[:, 4] > CONFIDENCE_THRESHOLD]

        scale_x, scale_y = frame_width / 320, frame_height / 320
        scaled_dets = dets[:, :4] * np.array([scale_x, scale_y, scale_x, scale_y])
        confidences = dets[:, 4]
        class_ids = dets[:, 5].astype(int)

        tracker_dets = [([x1, y1, x2 - x1, y2 - y1], conf, cls) for (x1, y1, x2, y2), conf, cls in zip(scaled_dets, confidences, class_ids)]

        tracks = tracker.update_tracks(tracker_dets, frame=frame)

        current_frame_items = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            
            box_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            inside_zone = cv2.pointPolygonTest(ZONE_POLY, box_center, False) >= 0
            
            class_name = model.names[track.det_class] if hasattr(track, 'det_class') else "unknown"
            
            unique_id = f"{class_name}_{track_id}"
            current_frame_items.add(unique_id)
            
            track_manager.update(unique_id, (xmin, ymin, xmax, ymax), class_name, inside_zone)

            color = GREEN if inside_zone else (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin), color, -1)
            cv2.putText(frame, unique_id, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        cv2.polylines(frame, [ZONE_POLY], isClosed=True, color=(255, 255, 0), thickness=2)

        global globalFrame
        globalFrame = frame

        active_tracks = track_manager.get_active_tracks()
        for unique_id, track_info in active_tracks.items():
            db_cache[unique_id] = {
                "is_inside": track_info['inside_zone'],
                "last_update": track_info['last_seen']
            }

        if frame_count % 30 == 0:
            update_database(db_cache)
            db_cache.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

def update_database(db_cache):
    for item_id, data in db_cache.items():
        is_inside = data["is_inside"]
        current_time = data["last_update"].isoformat()
        item = collection.find_one({"item_id": item_id})

        if item:
            if not is_inside and item["current_status"] == "inside":
                collection.update_one({"item_id": item_id}, {"$set": {"current_status": "outside", "timestamp_exited": current_time}})
            elif is_inside and item["current_status"] == "outside":
                collection.update_one({"item_id": item_id}, {"$set": {"current_status": "inside", "timestamp_entered": current_time}})
        else:
            if is_inside:
                new_item = {"item_id": item_id, "timestamp_entered": current_time, "timestamp_exited": None, "current_status": "inside"}
                collection.insert_one(new_item)

if __name__ == '__main__':
    api_thread = Thread(target=run_flask_api)
    api_thread.start()

    socket_thread = Thread(target=run_socket)
    socket_thread.start()
    
    run_object_tracking()