import torch
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import time
from threading import Thread, Lock
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv5 model (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)

# Function to read frames asynchronously using a separate thread
class VideoCaptureAsync:
    def __init__(self, src=0):
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

# Open the camera feed asynchronously
cap = VideoCaptureAsync(0).start()

# Get initial camera resolution
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
        print(f"Camera resolution: {frame_width}x{frame_height}")
else:
    print("Failed to open camera.")

# MongoDB connection string
mongo_uri = "mongodb+srv://pdiddy:pdiddy!@cluster0.ydaow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["item_tracking"]
collection = db["tracked_items"]

# Define operating zone polygon (central zone with buffer of 720 units)
buffer = 720
buffer = min(buffer, frame_width // 2, frame_height // 2)
ZONE_POLY = np.array([
    [buffer, 0],
    [frame_width - buffer, 0],
    [frame_width - buffer, frame_height],
    [buffer, frame_height]
])

# Dictionary to track items across frames based on their class
tracked_items = {}
class_counters = {}  # Counter to assign unique IDs for each class

# Map YOLOv5 class IDs to object names (like 'person', 'car', etc.)
class_names = model.names  # This will give a list of class names (0 -> 'person', 1 -> 'bicycle', etc.)

# Function to check if a point is inside the polygon zone
def is_inside_zone(box_center, zone_poly):
    return cv2.pointPolygonTest(zone_poly, box_center, False) >= 0

# Function to track the item in MongoDB
def track_item(item_id, is_inside):
    current_time = datetime.now().isoformat()
    item = collection.find_one({"item_id": item_id})

    if item:
        if not is_inside and item["current_status"] == "inside":
            collection.update_one({"item_id": item_id}, {"$set": {"current_status": "outside", "timestamp_exited": current_time}})
            print(f"Item {item_id} has exited the zone.")
        elif is_inside and item["current_status"] == "outside":
            collection.update_one({"item_id": item_id}, {"$set": {"current_status": "inside", "timestamp_entered": current_time}})
            print(f"Item {item_id} has re-entered the zone.")
    else:
        if is_inside:
            new_item = {
                "item_id": item_id,
                "timestamp_entered": current_time,
                "timestamp_exited": None,
                "current_status": "inside"
            }
            collection.insert_one(new_item)
            print(f"New item {item_id} entered the zone.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv5
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Track items in the current frame
    current_frame_items = set()

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_id = int(class_id)  # Ensure class_id is an integer
        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        box_center_int = (int(box_center[0]), int(box_center[1]))

        # Get the object class name
        class_name = class_names[class_id]

        # Initialize the counter for this class if it doesn't exist
        if class_name not in class_counters:
            class_counters[class_name] = 0

        # Check if this object has been tracked before
        item_id = None
        for tracked_id, data in tracked_items.items():
            if data["class_name"] == class_name and not data["inside_zone"]:  # If the same class is re-detected
                item_id = tracked_id
                break

        # If the object is new (not previously tracked), create a new item ID
        if not item_id:
            item_id = f'{class_name}_{class_counters[class_name]}'
            class_counters[class_name] += 1

        # Check if the center of the bounding box is inside the polygon zone
        inside_zone = is_inside_zone(box_center_int, ZONE_POLY)

        # Add the item_id to the set of currently tracked items
        current_frame_items.add(item_id)

        # If this item is not already tracked, track it
        if item_id not in tracked_items:
            tracked_items[item_id] = {"class_name": class_name, "inside_zone": inside_zone, "last_seen": datetime.now()}
            track_item(item_id, inside_zone)
        else:
            # Update tracking information in MongoDB if the status changed
            prev_status = tracked_items[item_id]["inside_zone"]
            if inside_zone != prev_status:
                track_item(item_id, inside_zone)
                tracked_items[item_id]["inside_zone"] = inside_zone

        # Draw bounding box and label (with item_id)
        color = (0, 255, 0) if inside_zone else (0, 0, 255)  # Green if inside the zone, red if outside
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Display the item ID above the bounding box
        cv2.putText(frame, f'ID: {item_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the polygon zone
    cv2.polylines(frame, [ZONE_POLY], isClosed=True, color=(255, 255, 0), thickness=2)

    # Display the frame with bounding boxes and the polygon zone
    cv2.imshow('Object Tracking', frame)

    # Remove items that have not been seen in the current frame
    for tracked_item_id in list(tracked_items.keys()):
        if tracked_item_id not in current_frame_items:
            tracked_items[tracked_item_id]["inside_zone"] = False
            track_item(tracked_item_id, False)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.stop()
cv2.destroyAllWindows()
