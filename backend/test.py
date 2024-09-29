import cv2

# Open the default camera (usually the first camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame reading was unsuccessful, exit the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the captured frame
    cv2.imshow('Camera Feed', frame)

    # Wait for 1ms and check if 'q' is pressed to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
