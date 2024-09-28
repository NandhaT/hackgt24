import torch
import cv2
# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Open camera feed
cap = cv2.VideoCapture(0)  # '0' for the first camera on the device

while cap.isOpened():
    ret, frame = cap.read()  # Read frame-by-frame from the camera
    if not ret:
        break
    
    # Pass the frame to the model

    results = model(frame)
    
    # Render the results on the frame (bounding boxes and labels)
    frame = results.render()[0]
    
    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()
    
    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the image with the bounding box
    cv2.imshow('Face Detector', frame)
    
    # Stop if Q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()