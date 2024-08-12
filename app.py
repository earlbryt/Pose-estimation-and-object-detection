import cv2 as cv
from ultralytics import YOLO
import os

print(os.getcwd())


model = YOLO("yolov8s.pt")
model = YOLO('yolov8n-pose.pt')

results = model.predict(source="0", show=True)

print(results)

if cv.waitKey(1) & 0xFF == ord('q'):
  break

