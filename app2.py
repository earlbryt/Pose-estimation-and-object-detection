import os
import cv2
from ultralytics import YOLO

print("Current working directory:", os.getcwd())

# Correct model name and path
model_name = "yolov8n.pt"
model_path = os.path.join(os.getcwd(), model_name)

# Attempt to load the model, download if not found
try:
    model = YOLO(model_path)
    print(f"Model loaded from {model_path}")
except FileNotFoundError:
    print(f"Model not found at {model_path}. Attempting to download...")
    model = YOLO("yolov8n")  # This should trigger a download
    print("Model downloaded and loaded successfully")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Perform prediction
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()