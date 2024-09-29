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
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

def run_object_tracking():
    CONFIDENCE_THRESHOLD = 0.4  # Lowered confidence threshold
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    # Initialize the camera feed
    cap = VideoCaptureAsync(0).start()

    if not cap.isOpened():
        logging.error("Failed to open camera.")
        return

    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read frame.")
        return
    frame_height, frame_width = frame.shape[:2]

    # Zone setup
    buffer = min(240, frame_width // 2, frame_height // 2)
    ZONE_POLY = np.array([[buffer, 0], [frame_width - buffer, 0], [frame_width - buffer, frame_height], [buffer, frame_height]])

    # Load YOLO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/output7/weights/best.pt').to(device)
    model.conf = CONFIDENCE_THRESHOLD  # Set confidence threshold
    logging.info("Custom YOLO model loaded successfully")

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30, nn_budget=100, embedder="mobilenet", embedder_gpu=True)

    # Dictionary to store consistent IDs
    track_history = {}

    # Local cache for database operations
    db_cache = {}
    frame_count = 0

    # Frame skip for processing (adjust as needed)
    FRAME_SKIP = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame, breaking the loop")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:  # Process every FRAME_SKIP frames
            continue

        # Resize frame for YOLO input (smaller size for speed)
        input_frame = cv2.resize(frame, (320, 320))

        # Run YOLO detection
        with torch.no_grad():  # Disable gradient calculation for inference
            results = model(input_frame)

        # Process detections
        dets = results.xyxy[0].cpu().numpy()  # Convert to numpy array
        dets = dets[dets[:, 4] > CONFIDENCE_THRESHOLD]  # Filter by confidence

        # Scale detections back to original frame size
        scale_x, scale_y = frame_width / 320, frame_height / 320
        scaled_dets = dets[:, :4] * np.array([scale_x, scale_y, scale_x, scale_y])
        confidences = dets[:, 4]
        class_ids = dets[:, 5].astype(int)

        # Prepare detections for DeepSORT
        tracker_dets = [([x1, y1, x2 - x1, y2 - y1], conf, cls) for (x1, y1, x2, y2), conf, cls in zip(scaled_dets, confidences, class_ids)]

        # Update tracks
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
            
            # Get class name for the track
            if hasattr(track, 'det_class'):
                class_name = model.names[track.det_class]
            else:
                class_name = "unknown"
            
            # Generate or retrieve consistent ID for the track
            if track_id not in track_history:
                track_history[track_id] = f"{class_name}_{len(track_history) + 1}"
            
            unique_id = track_history[track_id]
            current_frame_items.add(unique_id)
            
            # Update local cache
            if unique_id not in db_cache:
                db_cache[unique_id] = {"is_inside": inside_zone, "last_update": datetime.now()}
            elif db_cache[unique_id]["is_inside"] != inside_zone:
                db_cache[unique_id] = {"is_inside": inside_zone, "last_update": datetime.now()}

            color = GREEN if inside_zone else (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin), color, -1)
            cv2.putText(frame, unique_id, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Draw zone
        cv2.polylines(frame, [ZONE_POLY], isClosed=True, color=(255, 255, 0), thickness=2)

        # Display frame
        cv2.imshow('Object Tracking', frame)

        # Batch update database every 30 frames
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
    # Start the Flask API in a separate thread
    api_thread = Thread(target=run_flask_api)
    api_thread.start()

    # Run object tracking in the main thread
    run_object_tracking()