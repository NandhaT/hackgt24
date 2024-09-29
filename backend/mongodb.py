import time
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection string
mongo_uri = "mongodb+srv://pdiddy:pdiddy!@cluster0.ydaow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["medical_item_tracking"]
collection = db["tracked_items"]

def track_item(item_id, is_inside):
    current_time = datetime.now().isoformat()

    # Check if the item already exists in the database
    item = collection.find_one({"item_id": item_id})

    if item:
        # Update the current_status and timestamp_exited if outside
        if not is_inside:
            print(f"Item {item_id} is exiting the zone")
            collection.update_one(
                {"item_id": item_id},
                {"$set": {"current_status": "outside", "timestamp_exited": current_time}}
            )
        else:
            print(f"Item {item_id} is still inside")
    else:
        # If the item is new, insert it
        if is_inside:
            print(f"New item detected: {item_id}")
            new_item = {
                "item_id": item_id,
                "timestamp_entered": current_time,
                "timestamp_exited": None,
                "current_status": "inside"
            }
            collection.insert_one(new_item)
        else:
            print(f"Item {item_id} detected outside but not tracked yet")

# Example: Simulating item entry/exit
track_item("item_1", True)  # Item enters the zone
time.sleep(2)
track_item("item_1", False)  # Item exits the zone
