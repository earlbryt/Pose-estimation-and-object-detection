import cv2 as cv
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the webcam
cap = cv.VideoCapture(0)

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

    # Convert the annotated frame to BGR color space (OpenCV default)
    annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)

    # Display the annotated frame
    cv.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv.destroyAllWindows()