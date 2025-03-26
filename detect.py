import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")  # Change to your preferred model

# Open camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform YOLO inference
    results = model(frame)

    # Draw detected objects
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
