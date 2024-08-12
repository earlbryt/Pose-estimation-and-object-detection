import cv2
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO("yolov8m.pt")  # Using the nano model for faster inference

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)  # Set frame height

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Perform prediction
    results = model(frame, conf=0.5)  # Increase confidence threshold

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()