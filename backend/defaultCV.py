import torch
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, jsonify
from flask_cors import CORS

# Object Tracking Setup (your current code)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)

class VideoCaptureAsync:
    def __init__(self, src=0):  # Accepting the camera source as a parameter
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
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

# Flask API setup
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

# MongoDB connection
mongo_uri = "mongodb+srv://pdiddy:pdiddy!@cluster0.ydaow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["medical_item_tracking"]
collection = db["tracked_items"]

# API route to get items currently inside the zone
@app.route('/api/items/inside', methods=['GET'])
def get_items_inside_zone():
    inside_items = collection.find({"current_status": "inside"})
    item_list = [{"item_id": item["item_id"], "timestamp_entered": item["timestamp_entered"]} for item in inside_items]
    return jsonify(item_list)

def run_flask_api():
    """Run the Flask API in a separate thread."""
    app.run(host='0.0.0.0', port=5000)

# Object Tracking Loop
def run_object_tracking():
    # Initialize the camera feed
    cap = VideoCaptureAsync(0).start()

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
    else:
        print("Failed to open camera.")
        return

    # Zone setup (your current code)
    buffer = 720
    buffer = min(buffer, frame_width // 2, frame_height // 2)
    ZONE_POLY = np.array([[buffer, 0], [frame_width - buffer, 0], [frame_width - buffer, frame_height], [buffer, frame_height]])

    tracked_items = {}
    class_counters = {}
    class_names = model.names

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        current_frame_items = set()

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            class_id = int(class_id)
            box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            box_center_int = (int(box_center[0]), int(box_center[1]))

            class_name = class_names[class_id]
            if class_name not in class_counters:
                class_counters[class_name] = 0

            item_id = None
            for tracked_id, data in tracked_items.items():
                if data["class_name"] == class_name and not data["inside_zone"]:
                    item_id = tracked_id
                    break

            if not item_id:
                item_id = f'{class_name}_{class_counters[class_name]}'
                class_counters[class_name] += 1

            inside_zone = is_inside_zone(box_center_int, ZONE_POLY)
            current_frame_items.add(item_id)

            if item_id not in tracked_items:
                tracked_items[item_id] = {"class_name": class_name, "inside_zone": inside_zone, "last_seen": datetime.now()}
                track_item(item_id, inside_zone)
            else:
                prev_status = tracked_items[item_id]["inside_zone"]
                if inside_zone != prev_status:
                    track_item(item_id, inside_zone)
                    tracked_items[item_id]["inside_zone"] = inside_zone

            color = (0, 255, 0) if inside_zone else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'ID: {item_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.polylines(frame, [ZONE_POLY], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.imshow('Object Tracking', frame)

        for tracked_item_id in list(tracked_items.keys()):
            if tracked_item_id not in current_frame_items:
                tracked_items[tracked_item_id]["inside_zone"] = False
                track_item(tracked_item_id, False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

# Utility function to check if the box center is inside the zone
def is_inside_zone(box_center, zone_poly):
    return cv2.pointPolygonTest(zone_poly, box_center, False) >= 0

# Utility function to track items in MongoDB
def track_item(item_id, is_inside):
    current_time = datetime.now().isoformat()
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
    # Start the Flask API in a separate thread
    api_thread = Thread(target=run_flask_api)
    api_thread.start()

    # Run object tracking in the main thread
    run_object_tracking()
