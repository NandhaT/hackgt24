from flask import Flask, jsonify
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__)

# MongoDB connection string
mongo_uri = "mongodb+srv://pdiddy:pdiddy!@cluster0.ydaow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["medical_item_tracking"]
collection = db["tracked_items"]

# API route to get items currently inside the zone
@app.route('/api/items/inside', methods=['GET'])
def get_items_inside_zone():
    # Query the MongoDB collection for items where the current status is 'inside'
    inside_items = collection.find({"current_status": "inside"})
    
    # Create a list of item IDs to return
    item_list = [{"item_id": item["item_id"], "timestamp_entered": item["timestamp_entered"]} for item in inside_items]
    
    # Return the list of items as a JSON response
    return jsonify(item_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)